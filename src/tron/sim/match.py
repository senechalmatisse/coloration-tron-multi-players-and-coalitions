"""Boucle de partie et observateurs (§6.2, §6.4).

``Match`` demande un coup à l'agent du joueur au trait, l'applique, émet un
événement vers les observateurs (journal, affichage) et détecte la fin. Le
moteur ne dépend jamais de l'affichage (F6.5) : la visualisation n'est qu'un
observateur parmi d'autres.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from tron.agents.base import Agent, DecisionDetails, SearchStats, build_agents
from tron.agents.context import AllianceEvent, DecisionContext
from tron.core.config import GameConfig
from tron.core.directions import PASS
from tron.core.state import GameState
from tron.eval.voronoi import warmup
from tron.sim.diplomacy import DiplomacyState, summarize_diplomacy
from tron.sim.metrics import MetricsCollector
from tron.sim.result import MatchMetrics, MatchResult, PlayerTotals

_WARMED_UP = False


def ensure_warmup() -> None:
    """Compile (ou charge du cache) les noyaux une fois par processus."""
    global _WARMED_UP  # noqa: PLW0603 - drapeau de processus
    if not _WARMED_UP:
        warmup()
        _WARMED_UP = True


@dataclass(frozen=True, slots=True)
class MoveEvent:
    """Un demi-coup joué : déplacement ou élimination, et événements diplomatiques associés.

    ``diplomacy`` contient, dans l'ordre, l'issue de l'intention du joueur (au
    demi-coup ``ply``) puis les dissolutions et expirations constatées après le coup.
    """

    ply: int
    player: int
    direction: int
    origin: tuple[int, int]
    target: tuple[int, int] | None
    eliminated: bool
    stats: SearchStats | None
    diplomacy: tuple[AllianceEvent, ...] = ()
    details: DecisionDetails | None = None
    """Explication de la décision (P07), ``None`` si l'agent n'en fournit pas."""


def accepts_context(agent: Agent) -> bool:
    """Vrai si ``agent.choose_move`` accepte le paramètre ``context`` (agents externes)."""
    try:
        params = inspect.signature(agent.choose_move).parameters
    except (TypeError, ValueError):  # pragma: no cover - objet non introspectable
        return False
    return "context" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class MatchObserver(Protocol):
    """Observateur d'une partie (journal, affichage, statistiques…)."""

    def on_start(self, match: Match) -> None: ...

    def on_event(self, match: Match, event: MoveEvent) -> None: ...

    def on_end(self, match: Match, result: MatchResult) -> None: ...


class Match:
    """Une partie entre agents, pilotable coup par coup (``step``) ou d'un bloc (``run``).

    ``metrics`` active le calcul des métriques de coopération (P02, voir
    :mod:`tron.sim.metrics`) ; ``territory_every`` règle l'observation du
    territoire (``None`` : tous les n demi-coups, 0 : jamais). Les métriques
    n'influencent ni les règles ni les décisions.
    """

    def __init__(
        self,
        config: GameConfig,
        agents: Sequence[Agent] | None = None,
        observers: Sequence[MatchObserver] = (),
        positions: Sequence[tuple[int, int]] | None = None,
        *,
        metrics: bool = True,
        territory_every: int | None = None,
    ) -> None:
        ensure_warmup()
        self.config = config
        self.state = GameState(config, positions=positions)
        self.agents = list(agents) if agents is not None else build_agents(config)
        if len(self.agents) != config.n_players:
            raise ValueError("un agent par joueur est requis")
        for agent in self.agents:
            agent.reset()
        self.observers: list[MatchObserver] = list(observers)
        self.events: list[MoveEvent] = []
        self.result: MatchResult | None = None
        self.metrics: MetricsCollector | None = (
            MetricsCollector(config, territory_every) if metrics else None
        )
        self.diplomacy: DiplomacyState | None = (
            DiplomacyState(config.diplomacy, config.n_players) if config.diplomacy.enabled else None
        )
        self._with_context = [accepts_context(a) for a in self.agents]
        self.prior_moves: list[dict[str, Any]] = []
        """Lignes ``move`` jouées avant une reprise depuis une sauvegarde (P08)."""
        self._resumed = False
        self._started = False
        self._t0 = 0.0
        n = config.n_players
        self._decisions = [0] * n
        self._nodes = [0] * n
        self._time = [0.0] * n
        self._max_depth = [0] * n
        self._sum_depth = [0] * n
        self._leaves = [0] * n

    def add_observer(self, observer: MatchObserver) -> None:
        self.observers.append(observer)

    @property
    def finished(self) -> bool:
        return self.result is not None

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._t0 = perf_counter()
            if self.metrics is not None and not self._resumed:
                self.metrics.start(self.state)
            for obs in self.observers:
                obs.on_start(self)

    def step(self) -> MoveEvent | None:
        """Joue un demi-coup ; renvoie ``None`` si la partie est terminée."""
        self.start()
        st = self.state
        if st.is_terminal():
            self._finish()
            return None
        p = st.current
        stats: SearchStats | None = None
        details: DecisionDetails | None = None
        dipl: list[AllianceEvent] = []
        if st.has_legal_move(p):
            agent = self.agents[p]
            view = st.copy(history=False)
            if self._with_context[p]:
                snap = self.diplomacy.snapshot(st.ply) if self.diplomacy is not None else None
                decision = agent.choose_move(view, p, DecisionContext(st.ply, snap))
            else:
                decision = agent.choose_move(view, p)
            move, stats, details = decision.move, decision.stats, decision.details
            if decision.diplomacy is not None and self.diplomacy is not None:
                dipl += self.diplomacy.apply_intent(p, decision.diplomacy, st.ply)
            self._decisions[p] += 1
            self._nodes[p] += stats.nodes
            self._time[p] += stats.time
            self._max_depth[p] = max(self._max_depth[p], stats.depth)
            self._sum_depth[p] += stats.depth
            self._leaves[p] += stats.leaves
        else:
            move = PASS
        origin = st.head(p)
        moved = st.apply(move)
        if self.diplomacy is not None:
            dipl += self.diplomacy.advance(st.ply, [st.is_alive(q) for q in range(st.n)])
        event = MoveEvent(
            ply=st.ply - 1,
            player=p,
            direction=move,
            origin=origin,
            target=st.head(p) if moved else None,
            eliminated=not moved,
            stats=stats,
            diplomacy=tuple(dipl),
            details=details,
        )
        self.events.append(event)
        target = event.target
        for agent in self.agents:  # crochet d'observation (P05), vide pour les agents historiques
            agent.observe_move(p, move, origin, target, not moved, st)
        if self.metrics is not None:
            self.metrics.record(st, p, origin, move, event.target, event.eliminated, event.ply)
        for obs in self.observers:
            obs.on_event(self, event)
        if st.is_terminal():
            self._finish()
        return event

    def save_checkpoint(self, path: str | Path) -> Path:
        """Sauvegarde exacte de la partie en cours (voir :mod:`tron.sim.checkpoint`)."""
        from tron.sim.checkpoint import save_checkpoint  # noqa: PLC0415 - import circulaire

        return save_checkpoint(self, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, observers: Sequence[MatchObserver] = ()) -> Match:
        """Reprend une partie sauvegardée là où elle s'était arrêtée."""
        from tron.sim.checkpoint import load_checkpoint  # noqa: PLC0415 - import circulaire

        return load_checkpoint(path, observers)

    def run(self) -> MatchResult:
        """Joue la partie jusqu'au bout."""
        while self.step() is not None:
            pass
        assert self.result is not None
        return self.result

    def _finish(self) -> None:
        if self.result is not None:
            return
        out = self.state.outcome()
        assert out is not None
        cfg = self.config
        names = cfg.team_names
        totals = tuple(
            PlayerTotals(
                decisions=self._decisions[i],
                nodes=self._nodes[i],
                think_time=self._time[i],
                max_depth=self._max_depth[i],
                mean_depth=self._sum_depth[i] / self._decisions[i] if self._decisions[i] else 0.0,
                leaves=self._leaves[i],
            )
            for i in range(cfg.n_players)
        )
        self.result = MatchResult(
            name=cfg.name,
            seed=cfg.seed,
            first_player=cfg.first_player,
            width=cfg.width,
            height=cfg.height,
            players=tuple(p.name for p in cfg.players),
            teams=tuple(p.team for p in cfg.players),
            agents=tuple(a.describe() for a in self.agents),
            depths=tuple(p.depth for p in cfg.players),
            winning_team=names[out.winning_team] if out.winning_team >= 0 else "",
            winners=out.winners,
            ranks=out.ranks,
            elimination_order=out.elimination_order,
            n_moves=out.n_moves,
            n_plies=out.n_plies,
            duration=perf_counter() - self._t0,
            totals=totals,
            positions=self.state.positions,
            metrics=self._final_metrics(),
        )
        for obs in self.observers:
            obs.on_end(self, self.result)

    def _final_metrics(self) -> MatchMetrics | None:
        if self.metrics is None:
            return None
        summary = (
            summarize_diplomacy(self.diplomacy.events, self.state.ply)
            if self.diplomacy is not None
            else None
        )
        return self.metrics.finish(self.state, summary)
