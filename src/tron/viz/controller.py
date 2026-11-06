"""Contrôle de lecture, indépendant de la bibliothèque graphique (F6.2, F6.3).

* :class:`LiveSource` : partie en temps réel (un ``Match`` piloté demi-coup par
  demi-coup) ; « nouvelle partie » change de graine et fait tourner le premier joueur.
* :class:`ReplaySource` : rejeu d'un journal JSONL, en avant comme en arrière
  (``undo``), avec accès direct à un demi-coup.
* :class:`PlaybackController` : pause, pas à pas, vitesse réglable, surcouche
  de Voronoi, et données du panneau d'information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tron.core.config import GameConfig
from tron.core.rng import derive_seed
from tron.core.state import GameState
from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.match import Match, MoveEvent
from tron.sim.recorder import GameLog, apply_logged

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
        self.max_steps = max_steps_per_update
        self._acc = 0.0
        self._heur: VoronoiHeuristic | None = None
        self._cache_key: tuple[int, int, int] | None = None
        self._shares: list[float] = []
        self._owners: np.ndarray | None = None
        self._last: dict[int, dict[str, float | int | None]] = {}

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
            status = f"Victoire de l'équipe {cfg.team_names[out.winning_team]}"
        elif self.paused:
            status = "Pause"
        else:
            status = f"Lecture ×{self.speed:g}"
        return PanelInfo(
            title=self.source.title,
            ply=st.ply,
            moves=st.n_moves,
            current=st.current,
            status=status,
            players=tuple(lines),
            teams=tuple(teams),
        )
