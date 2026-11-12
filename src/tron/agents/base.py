"""Interface commune des agents et registre (F3.1, F3.7, extensibilité)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, ClassVar

from tron.agents.context import DecisionContext, DiplomacyIntent
from tron.core.config import ConfigError, GameConfig
from tron.core.directions import PASS
from tron.core.rng import STREAM_AGENT, make_rng
from tron.core.state import GameState
from tron.eval.base import Heuristic, make_heuristic
from tron.eval.valuation import Valuation

TIE_BREAKS = ("random", "fixed")


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Statistiques d'une décision : nœuds, temps, valeur, profondeur atteinte…"""

    nodes: int = 0
    leaves: int = 0
    prunes: int = 0
    time: float = 0.0
    value: float = math.nan
    depth: int = 0
    ties: int = 1
    engine: str = ""
    pv: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pv"] = list(self.pv)
        if not math.isfinite(self.value):
            data["value"] = None
        return data


@dataclass(frozen=True, slots=True)
class Decision:
    """Coup choisi (direction 0 à 3, ou PASS), statistiques et intention diplomatique."""

    move: int
    stats: SearchStats
    diplomacy: DiplomacyIntent | None = None


class Agent(ABC):
    """Agent : ``choose_move(state, player) → Decision`` (F3.1).

    Paramètres communs : ``depth`` (plies), ``rounds`` (tours complets : la
    profondeur vaut alors rounds × joueurs vivants), ``tie_break`` (``random``
    tiré avec la graine de la partie, ou ``fixed`` = ordre N, E, S, O).
    """

    name: ClassVar[str] = "?"

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        depth: int = 1,
        rounds: int | None = None,
        tie_break: str = "random",
        seed: int | None = None,
        heuristic: Heuristic | None = None,
    ) -> None:
        if tie_break not in TIE_BREAKS:
            raise ConfigError(f"départage inconnu : {tie_break} (attendu : {TIE_BREAKS})")
        if depth < 1 or (rounds is not None and rounds < 1):
            raise ConfigError("la profondeur doit être >= 1")
        self.player = player
        self.config = config
        self.depth = depth
        self.rounds = rounds
        self.tie_break = tie_break
        self.rng = make_rng(config.seed if seed is None else seed, STREAM_AGENT, player)
        self.heuristic = heuristic or make_heuristic(
            config.heuristic, config.width, config.height, config.n_players
        )
        self.valuation = Valuation(config, self.heuristic)

    def effective_depth(self, state: GameState) -> int:
        """Profondeur en plies pour l'état courant."""
        if self.rounds is not None:
            return self.rounds * len(state.alive_players())
        return self.depth

    def choose_move(
        self,
        state: GameState,
        player: int | None = None,
        context: DecisionContext | None = None,
    ) -> Decision:
        """Choisit le coup du joueur au trait. ``state`` est restauré à l'identique.

        ``context`` (facultatif) apporte le demi-coup et l'état diplomatique ; les
        agents qui ne s'en servent pas l'ignorent. L'intention diplomatique
        éventuelle est formulée après le choix du coup (crochet :meth:`_diplomacy`).
        """
        p = state.current if player is None else player
        if p != state.current:
            raise ValueError(f"le joueur {p} n'est pas au trait ({state.current})")
        t0 = perf_counter()
        moves = state.legal_moves()
        if not moves:
            return Decision(PASS, SearchStats(time=perf_counter() - t0, engine=self.name))
        move, stats = self._decide_with_context(state, moves, context)
        intent = self._diplomacy(state, move, context) if context is not None else None
        elapsed = perf_counter() - t0
        return Decision(move, _with_time(stats, elapsed), intent)

    def _decide_with_context(
        self, state: GameState, moves: list[int], context: DecisionContext | None
    ) -> tuple[int, SearchStats]:
        """Crochet de décision avec contexte ; par défaut, :meth:`_decide` sans contexte."""
        return self._decide(state, moves)

    def _diplomacy(
        self, state: GameState, move: int, context: DecisionContext
    ) -> DiplomacyIntent | None:
        """Intention diplomatique de fin de décision (aucune par défaut)."""
        return None

    @abstractmethod
    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        """Choisit parmi ``moves`` (au moins un coup légal)."""

    def pick(self, ties: Sequence[int]) -> int:
        """Départage des coups de même valeur (F3.7)."""
        ordered = sorted(ties)
        if self.tie_break == "fixed" or len(ordered) == 1:
            return ordered[0]
        return ordered[int(self.rng.integers(len(ordered)))]

    def reset(self) -> None:  # noqa: B027 - crochet facultatif, vide par défaut
        """Réinitialise la mémoire de l'agent (nouvelle partie)."""

    def describe(self) -> str:
        return f"{self.name}(d={self.depth})"


def _with_time(stats: SearchStats, elapsed: float) -> SearchStats:
    return SearchStats(
        nodes=stats.nodes,
        leaves=stats.leaves,
        prunes=stats.prunes,
        time=elapsed,
        value=stats.value,
        depth=stats.depth,
        ties=stats.ties,
        engine=stats.engine,
        pv=stats.pv,
    )


# --------------------------------------------------------------- registre
AGENTS: dict[str, type[Agent]] = {}


def register_agent(name: str) -> Callable[[type[Agent]], type[Agent]]:
    """Décorateur d'enregistrement d'un agent sous ``name``."""

    def deco(cls: type[Agent]) -> type[Agent]:
        cls.name = name
        AGENTS[name] = cls
        return cls

    return deco


def make_agent(name: str, player: int, config: GameConfig, **params: Any) -> Agent:
    """Instancie l'agent ``name`` pour ``player`` (paramètres issus de la configuration)."""
    try:
        cls = AGENTS[name]
    except KeyError:
        raise ConfigError(f"agent inconnu : {name} (connus : {sorted(AGENTS)})") from None
    try:
        return cls(player, config, **params)
    except TypeError as exc:
        raise ConfigError(f"paramètres invalides pour l'agent {name} : {exc}") from exc


def build_agents(config: GameConfig) -> list[Agent]:
    """Un agent par joueur, selon la configuration de la partie."""
    agents = []
    for i, spec in enumerate(config.players):
        params = dict(spec.params)
        params.setdefault("depth", spec.depth)
        agents.append(make_agent(spec.agent, i, config, **params))
    return agents
