"""Contrôle de lecture, indépendant de la bibliothèque graphique (F6.2, F6.3).

* :class:`LiveSource` : partie en temps réel (un ``Match`` piloté demi-coup par
  demi-coup) ; « nouvelle partie » change de graine et fait tourner le premier joueur.
* :class:`ReplaySource` : rejeu d'un journal JSONL, en avant comme en arrière
  (``undo``), avec accès direct à un demi-coup.
* :class:`PlaybackController` : pause, pas à pas, vitesse réglable, surcouche
  de Voronoi, données du panneau d'information et panneau de décision (P07).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tron.agents.base import DecisionDetails
from tron.core.config import GameConfig
from tron.core.directions import direction_name, parse_direction
from tron.core.rng import derive_seed
from tron.core.state import GameState
from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.match import Match, MoveEvent
from tron.sim.recorder import GameLog, apply_logged
from tron.viz.render import pv_path
from tron.viz.scene import PlyExtras, alliance_pairs, record_extras

UNAVAILABLE = "non disponible"
END_LABELS = {
    "survive": "objectif de survie atteint",
    "target_score": "score cible atteint",
    "max_rounds": "nombre maximal de tours atteint",
}
"""Libellés des causes de fin autres que l'élimination (P09)."""

SPEEDS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
"""Vitesses de lecture disponibles, en demi-coups par seconde."""


@dataclass(frozen=True, slots=True)
class PlayerLine:
    """Une ligne du panneau : identité, algorithme, état, territoire, dernière décision."""

    index: int
    name: str
    team: str
    agent: str
    alive: bool
    rank: int | None
    share: float
    nodes: int | None
    value: float | None
    depth: int | None


@dataclass(frozen=True, slots=True)
class PanelInfo:
    title: str
    ply: int
    moves: int
    current: int
    status: str
    players: tuple[PlayerLine, ...]
    teams: tuple[tuple[str, float, int], ...]
    """(équipe, part de territoire, joueurs vivants)."""


@dataclass(frozen=True, slots=True)
class DecisionPanel:
    """Dernière décision : coup, statistiques, alternatives, variante principale (P07).

    Les valeurs que l'agent n'a pas calculées sont affichées « non disponible ».
    """

    ply: int
    player: int
    name: str
    agent: str
    move: str
    value: float | None
    depth: int | None
    nodes: int | None
    leaves: int | None
    prunes: int | None
    engine: str
    pv: tuple[int, ...]
    alternatives: tuple[tuple[str, str], ...]
    extra: tuple[str, ...]

    def lines(self, top: int = 4) -> list[str]:
        """Texte du panneau (indépendant de la bibliothèque graphique)."""

        def fmt(v: float | int | None, spec: str = "") -> str:
            return UNAVAILABLE if v is None else format(v, spec)

        pv = "".join(direction_name(d) for d in self.pv) or UNAVAILABLE
        out = [
            f"Décision {self.ply} · {self.player} {self.name} ({self.agent})",
            f"coup {self.move} · valeur {fmt(self.value, '.4f')} · moteur {self.engine or '-'}",
            f"profondeur {fmt(self.depth)} · nœuds {fmt(self.nodes)} · "
            f"feuilles {fmt(self.leaves)} · coupes {fmt(self.prunes)}",
            f"variante principale : {pv}",
        ]
        if self.alternatives:
            alt = " · ".join(f"{d} {v}" for d, v in self.alternatives[:top])
            out.append(f"coups : {alt}")
        else:
            out.append(f"coups : {UNAVAILABLE}")
        out.extend(self.extra)
        return out


def _details_lines(details: DecisionDetails | None, names: list[str]) -> tuple[str, ...]:
    if details is None:
        return ()
    out = []
    if details.visits:
        out.append("visites : " + " · ".join(f"{direction_name(m)} {k}" for m, k in details.visits))
    if details.social_weights:
        out.append(
            "poids W : "
            + " · ".join(
                f"{names[j] if j < len(names) else j} {w:+.2f}" for j, w in details.social_weights
            )
        )
    if details.allies:
        out.append(
            "alliés tactiques : "
            + ", ".join(names[a] if a < len(names) else str(a) for a in details.allies)
        )
    if details.heuristic_weights:
        out.append(
            "poids : " + " · ".join(f"{k[:4]} {w:.2f}" for k, w in details.heuristic_weights)
        )
    if details.opponents:
        out.append(
            "profils : "
            + " · ".join(
                f"{names[q] if q < len(names) else q} {k} {p:.0%}" for q, k, p in details.opponents
            )
        )
    if details.note:
        out.append(details.note)
    return tuple(out)


def _alternatives(details: DecisionDetails | None) -> tuple[tuple[str, str], ...]:
    if details is None:
        return ()
    return tuple(
        (direction_name(m), UNAVAILABLE if v is None else f"{v:.4f}") for m, v in details.ranked()
    )


class Source(Protocol):
    @property
    def state(self) -> GameState: ...

    @property
    def config(self) -> GameConfig: ...

    @property
    def finished(self) -> bool: ...

    @property
    def can_go_back(self) -> bool: ...

    @property
    def agent_names(self) -> list[str]: ...

    @property
    def title(self) -> str: ...

    def forward(self) -> MoveEvent | dict[str, object] | None: ...

    def backward(self) -> bool: ...

    def new_game(self) -> None: ...


class LiveSource:
    """Partie jouée en direct par les agents de la configuration."""

    def __init__(self, config: GameConfig) -> None:
        self.base = config
        self.games = 0
        self.match = Match(config)

    @property
    def state(self) -> GameState:
        return self.match.state

    @property
    def config(self) -> GameConfig:
        return self.match.config

    @property
    def finished(self) -> bool:
        return self.match.state.is_terminal()

    @property
    def can_go_back(self) -> bool:
        return False

    @property
    def agent_names(self) -> list[str]:
        return [a.describe() for a in self.match.agents]

    @property
    def title(self) -> str:
        return f"{self.config.name} — partie {self.games + 1} (graine {self.config.seed})"

    def forward(self) -> MoveEvent | None:
        return self.match.step()

    def extras(self, ply: int) -> PlyExtras:
        """Direction jouée, diplomatie, zones et obstacles du demi-coup ``ply`` (thème néon)."""
        events = self.match.events
        i = ply - (self.state.ply - len(events))
        if 0 <= i < len(events):
            e = events[i]
            return PlyExtras(e.direction, e.diplomacy, e.zone, e.obstacles)
        return PlyExtras()

    def backward(self) -> bool:
        return False

    def new_game(self) -> None:
        self.games += 1
        n = self.base.n_players
        cfg = self.base.with_seed(
            derive_seed(self.base.seed, self.games), (self.base.first_player + self.games) % n
        )
        self.match = Match(cfg)


class ReplaySource:
    """Rejeu d'un journal, avec retour arrière et accès direct."""

    def __init__(self, log: GameLog) -> None:
        self.log = log
        self._state = log.initial_state()
        self._dirs = log.directions
        self.index = 0

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def config(self) -> GameConfig:
        return self.log.config

    @property
    def finished(self) -> bool:
        return self.index >= len(self.log.moves)

    @property
    def can_go_back(self) -> bool:
        return self.index > 0

    @property
    def agent_names(self) -> list[str]:
        return self.log.agents or [p.agent for p in self.config.players]

    @property
    def title(self) -> str:
        return f"Rejeu {self.config.name} (graine {self.config.seed})"

    def extras(self, ply: int) -> PlyExtras:
        """Direction jouée, diplomatie, zones et obstacles du demi-coup ``ply`` (thème néon)."""
        if 0 <= ply < len(self.log.moves):
            return record_extras(self.log.moves[ply], self._dirs[ply])
        return PlyExtras()

    def forward(self) -> dict[str, object] | None:
        if self.finished:
            return None
        rec = self.log.moves[self.index]
        apply_logged(self._state, rec, self._dirs[self.index])
        self.index += 1
        return rec

    def backward(self) -> bool:
        if self.index == 0:
            return False
        self._state.undo()
        self.index -= 1
        return True

    def seek(self, ply: int) -> None:
        ply = max(0, min(ply, len(self.log.moves)))
        while self.index > ply:
            self.backward()
        while self.index < ply:
            self.forward()

    def new_game(self) -> None:
        self.seek(0)


class PlaybackController:
    """Pause, pas à pas, vitesse, surcouche de Voronoi et panneau d'information."""

    def __init__(
        self,
        source: Source,
        speed: float = 10.0,
        paused: bool = False,
        voronoi: bool = False,
        max_steps_per_update: int = 200,
    ) -> None:
        self.source = source
        self.paused = paused
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.show_voronoi = voronoi
        self.show_decision = False
        self.max_steps = max_steps_per_update
        self._acc = 0.0
        self._heur: VoronoiHeuristic | None = None
        self._cache_key: tuple[int, int, int] | None = None
        self._shares: list[float] = []
        self._owners: np.ndarray | None = None
        self._last: dict[int, dict[str, float | int | None]] = {}
        self._alliance_key: tuple[int, int] | None = None
        self._alliance_pairs: tuple[tuple[int, int], ...] = ()

    # ------------------------------------------------------------- commandes
    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_index]

    def faster(self) -> None:
        self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)

    def slower(self) -> None:
        self.speed_index = max(self.speed_index - 1, 0)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_voronoi(self) -> None:
        self.show_voronoi = not self.show_voronoi

    def toggle_decision(self) -> None:
        self.show_decision = not self.show_decision

    # ---------------------------------------------------------- décision (P07)
    def decision_panel(self) -> DecisionPanel | None:
        """Explication de la dernière décision jouée (``None`` avant le premier coup)."""
        src = self.source
        names = [p.name for p in src.config.players]
        agents = src.agent_names
        if isinstance(src, LiveSource):
            events = [e for e in src.match.events if e.stats is not None]
            if not events:
                return None
            e = events[-1]
            s = e.stats
            assert s is not None
            return DecisionPanel(
                ply=e.ply,
                player=e.player,
                name=names[e.player],
                agent=agents[e.player] if e.player < len(agents) else "",
                move=direction_name(e.direction),
                value=s.value if math.isfinite(s.value) else None,
                depth=s.depth,
                nodes=s.nodes,
                leaves=s.leaves,
                prunes=s.prunes,
                engine=s.engine,
                pv=tuple(s.pv),
                alternatives=_alternatives(e.details),
                extra=_details_lines(e.details, names),
            )
        if isinstance(src, ReplaySource):
            for i in range(src.index - 1, -1, -1):
                rec = src.log.moves[i]
                if "nodes" not in rec:
                    continue
                player = int(rec["player"])
                details = src.log.details(i)
                pv = tuple(parse_direction(c) for c in str(rec.get("pv", "")))
                return DecisionPanel(
                    ply=int(rec["ply"]),
                    player=player,
                    name=names[player],
                    agent=agents[player] if player < len(agents) else "",
                    move=str(rec["dir"]),
                    value=rec.get("value"),
                    depth=rec.get("depth"),
                    nodes=rec.get("nodes"),
                    leaves=rec.get("leaves"),
                    prunes=rec.get("prunes"),
                    engine=str(rec.get("engine", "")),
                    pv=pv,
                    alternatives=_alternatives(details),
                    extra=_details_lines(details, names),
                )
            return None
        return None  # pragma: no cover - autre source

    def decision_path(self) -> list[tuple[int, int, int]]:
        """Suite de la variante principale de la dernière décision, depuis l'état affiché."""
        panel = self.decision_panel() if self.show_decision else None
        st = self.source.state
        if panel is None or len(panel.pv) < 2 or st.ply != panel.ply + 1:
            return []
        return pv_path(st, panel.pv[1:])

    # ------------------------------------------------- scène (thème néon)
    def ply_extras(self, ply: int) -> PlyExtras:
        """Direction jouée, diplomatie, zones et obstacles du demi-coup ``ply``."""
        extras = getattr(self.source, "extras", None)
        return extras(ply) if extras is not None else PlyExtras()

    def alliances(self) -> tuple[tuple[int, int], ...]:
        """Alliances tactiques actives (diplomatie, P03) dans l'état affiché."""
        src = self.source
        if isinstance(src, LiveSource):
            dip = src.match.diplomacy
            return dip.snapshot(src.state.ply).active_pairs if dip is not None else ()
        if isinstance(src, ReplaySource):
            key = (id(src.log), src.index)
            if key != self._alliance_key:
                events = [
                    ev for rec in src.log.moves[: src.index] for ev in record_extras(rec).diplomacy
                ]
                self._alliance_pairs = alliance_pairs(events)
                self._alliance_key = key
            return self._alliance_pairs
        return ()  # pragma: no cover - autre source

    def step_forward(self) -> bool:
        event = self.source.forward()
        if event is None:
            return False
        self._remember(event)
        return True

    def step_backward(self) -> bool:
        return self.source.backward()

    def new_game(self) -> None:
        self.source.new_game()
        self._acc = 0.0
        self._last.clear()
        self._heur = None
        self._cache_key = None

    def update(self, dt: float) -> int:
        """Avance selon la vitesse et le temps écoulé ; renvoie le nombre de demi-coups joués."""
        if self.paused or self.source.finished:
            self._acc = 0.0
            return 0
        self._acc += dt * self.speed
        steps = 0
        while self._acc >= 1.0 and steps < self.max_steps:
            self._acc -= 1.0
            if not self.step_forward():
                break
            steps += 1
        if steps >= self.max_steps:
            self._acc = 0.0
        return steps

    def _remember(self, event: MoveEvent | dict[str, object]) -> None:
        if isinstance(event, MoveEvent):
            s = event.stats
            if s is not None:
                self._last[event.player] = {
                    "nodes": s.nodes,
                    "value": s.value if math.isfinite(s.value) else None,
                    "depth": s.depth,
                }
        else:
            player = int(event["player"])  # type: ignore[call-overload]
            if "nodes" in event:
                self._last[player] = {
                    "nodes": event.get("nodes"),  # type: ignore[dict-item]
                    "value": event.get("value"),  # type: ignore[dict-item]
                    "depth": event.get("depth"),  # type: ignore[dict-item]
                }

    # -------------------------------------------------------------- données
    def _analyse(self) -> None:
        st = self.source.state
        key = (id(st), st.ply, st.key)
        if key == self._cache_key:
            return
        if self._heur is None or self._heur.width != st.width or self._heur.n_players != st.n:
            self._heur = VoronoiHeuristic(st.width, st.height, st.n)
        if st.alive_players():
            self._shares = self._heur.shares(st)
            self._owners = self._heur.owner_map(st)
        else:  # pragma: no cover - impossible en tour par tour
            self._shares = [0.0] * st.n
            self._owners = None
        self._cache_key = key

    def owners(self) -> np.ndarray | None:
        """Carte des propriétaires de Voronoi si la surcouche est active."""
        if not self.show_voronoi:
            return None
        self._analyse()
        return self._owners

    def panel(self) -> PanelInfo:
        self._analyse()
        st = self.source.state
        cfg = st.config
        out = st.outcome()
        names = self.source.agent_names
        lines = []
        for p, spec in enumerate(cfg.players):
            last = self._last.get(p, {})
            lines.append(
                PlayerLine(
                    index=p,
                    name=spec.name,
                    team=spec.team,
                    agent=names[p] if p < len(names) else spec.agent,
                    alive=st.is_alive(p),
                    rank=out.ranks[p] if out else None,
                    share=self._shares[p] if p < len(self._shares) else 0.0,
                    nodes=last.get("nodes"),  # type: ignore[arg-type]
                    value=last.get("value"),
                    depth=last.get("depth"),  # type: ignore[arg-type]
                )
            )
        teams = []
        for t, name in enumerate(cfg.team_names):
            members = [p for p in range(st.n) if cfg.team_index[p] == t]
            teams.append(
                (
                    name,
                    sum(self._shares[p] for p in members),
                    sum(1 for p in members if st.is_alive(p)),
                )
            )
        if out is not None:
            if out.winning_team < 0:
                status = "Égalité"
            elif cfg.rules.individual:
                status = f"Victoire de {cfg.victory_names[out.winning_team]} (survie individuelle)"
            else:
                status = f"Victoire de l'équipe {cfg.team_names[out.winning_team]}"
            reason = END_LABELS.get(st.end_reason())
            if reason:
                status = f"{status} ({reason})"
        elif self.paused:
            status = "Pause"
        else:
            status = f"Lecture ×{self.speed:g}"
        if cfg.rules.needs_engine:  # règles à état propre (P09) : tour, scores de zone
            status = f"{status} · tour {st.round}"
        if cfg.rules.zones:
            scores = " – ".join(
                f"{name} {score}" for name, score in zip(cfg.victory_names, st.scores, strict=True)
            )
            status = f"{status} · {scores}"
        return PanelInfo(
            title=self.source.title,
            ply=st.ply,
            moves=st.n_moves,
            current=st.current,
            status=status,
            players=tuple(lines),
            teams=tuple(teams),
        )
