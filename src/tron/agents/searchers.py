"""Agents de recherche : Negamax, Paranoid, MaxN et SocialAgent (F3.3 à F3.6, F4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from tron.agents.base import Agent, SearchStats, register_agent
from tron.agents.search import (
    BoundFn,
    SearchMemory,
    SearchOptions,
    SearchResult,
    search,
)
from tron.agents.social import (
    PROFILES,
    LinearBound,
    Matrix,
    TwoCamp,
    is_identity,
    social_matrix,
    sparse_rows,
    two_camp_reduction,
)
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState


class SearchAgent(Agent):
    """Base des agents de recherche : options, mémoire (table, historique), départage."""

    DEFAULT_ORDERING: ClassVar[str] = "static"
    DEFAULT_ITERATIVE: ClassVar[bool] = False
    DEFAULT_TT: ClassVar[bool] = False

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        ordering: str | None = None,
        iterative_deepening: bool | None = None,
        tt: bool | None = None,
        pruning: bool = True,
        time_limit: float | None = None,
        forced_shortcut: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(player, config, **kwargs)
        try:
            self.options = SearchOptions(
                ordering=ordering or self.DEFAULT_ORDERING,
                iterative=(
                    self.DEFAULT_ITERATIVE if iterative_deepening is None else iterative_deepening
                ),
                tt=self.DEFAULT_TT if tt is None else tt,
                pruning=pruning,
                time_limit=time_limit,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self.forced_shortcut = forced_shortcut
        size = (config.width + 2) * (config.height + 2)
        self.memory = SearchMemory.create(self.options, config.n_players, size)
        self.last_result: SearchResult | None = None

    def reset(self) -> None:
        self.memory.clear()
        self.last_result = None

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        if len(moves) == 1 and self.forced_shortcut:
            return moves[0], SearchStats(nodes=1, depth=0, engine=self.engine_name(state))
        result = self.run_search(state, self.effective_depth(state))
        self.last_result = result
        move = self.pick(result.ties)
        return move, SearchStats(
            nodes=result.nodes,
            leaves=result.leaves,
            prunes=result.prunes,
            value=result.root_values.get(move, result.value),
            depth=result.depth,
            ties=len(result.ties),
            engine=result.engine,
            pv=tuple(result.pv),
        )

    def engine_name(self, state: GameState) -> str:
        return self.name

    def run_search(self, state: GameState, depth: int) -> SearchResult:
        """Recherche brute à la profondeur ``depth`` (sans départage aléatoire)."""
        raise NotImplementedError

    def describe(self) -> str:
        unit = f"r={self.rounds}" if self.rounds is not None else f"d={self.depth}"
        return f"{self.name}({unit})"


@register_agent("negamax")
class NegamaxAgent(SearchAgent):
    """Negamax alpha-bêta à deux joueurs (référence de validation, F3.5)."""

    def __init__(self, player: int, config: GameConfig, **kwargs: Any) -> None:
        if config.n_players != 2:
            raise ConfigError("negamax ne s'applique qu'aux parties à deux joueurs")
        super().__init__(player, config, **kwargs)

    def run_search(self, state: GameState, depth: int) -> SearchResult:
        return search(
            "negamax", state, self.valuation, depth, options=self.options, memory=self.memory
        )


@register_agent("paranoid")
class ParanoidAgent(SearchAgent):
    """Paranoid (F3.3) : le joueur racine maximise sa part, tous les autres la minimisent."""

    DEFAULT_ORDERING = "history"
    DEFAULT_ITERATIVE = True
    DEFAULT_TT = True

    def run_search(self, state: GameState, depth: int) -> SearchResult:
        p = self.player
        signs = tuple(1 if q == p else -1 for q in range(state.n))
        return search(
            "alphabeta",
            state,
            self.valuation,
            depth,
            options=self.options,
            memory=self.memory,
            coeffs=((p, 1.0),),
            signs=signs,
            camp=(p,),
        )

    def engine_name(self, state: GameState) -> str:
        return "alphabeta"


@register_agent("maxn")
class MaxNAgent(SearchAgent):
    """MaxN (F3.4) : chacun maximise sa composante, élagage superficiel de Korf."""

    def run_search(self, state: GameState, depth: int) -> SearchResult:
        val = self.valuation

        def bound(q: int, p: int, y: float) -> float:
            return val.pair_bound(p, q) - y

        return search(
            "maxn",
            state,
            val,
            depth,
            options=self.options,
            memory=self.memory,
            rows=None,
            bound=bound,
        )

    def engine_name(self, state: GameState) -> str:
        return "maxn"


@register_agent("social")
class SocialAgent(SearchAgent):
    """Recherche à préférences sociales (matrice W, F4.2 à F4.6).

    Paramètres : ``profile`` (``maxn``, ``paranoid``, ``team``, ``team_paranoid``,
    ``custom``), ``alpha`` (profil ``team``), ``matrix`` (profil ``custom``),
    ``engine`` (``auto``, ``alphabeta``, ``maxn``).
    """

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        profile: str = "team",
        alpha: float = 1.0,
        matrix: Sequence[Sequence[float]] | None = None,
        engine: str = "auto",
        ordering: str | None = None,
        iterative_deepening: bool | None = None,
        tt: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if profile not in PROFILES:
            raise ConfigError(f"profil inconnu : {profile} (attendu : {PROFILES})")
        if engine not in ("auto", "alphabeta", "maxn"):
            raise ConfigError(f"moteur inconnu : {engine}")
        try:
            w = social_matrix(profile, config.team_index, player, alpha, matrix)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self.profile = profile
        self.alpha = alpha
        self.engine = engine
        self.matrix: Matrix = w
        self._identity = is_identity(w)
        self._rows = sparse_rows(w)
        self._reductions: dict[tuple[bool, ...], TwoCamp | None] = {}
        # Réglages par défaut selon le moteur probable (alpha-bêta : tri et itératif).
        probable_ab = engine == "alphabeta" or (
            engine == "auto"
            and two_camp_reduction(w, config.team_index, player, [True] * config.n_players)
            is not None
        )
        super().__init__(
            player,
            config,
            ordering=ordering or ("history" if probable_ab else "static"),
            iterative_deepening=probable_ab if iterative_deepening is None else iterative_deepening,
            tt=probable_ab if tt is None else tt,
            **kwargs,
        )
        self._bound: BoundFn = self._make_bound()

    def _make_bound(self) -> BoundFn:
        val = self.valuation
        if self._identity:

            def bound(q: int, p: int, y: float) -> float:
                return val.pair_bound(p, q) - y

            return bound
        return LinearBound(self.matrix, val.sum_bound)

    def reduction(self, state: GameState) -> TwoCamp | None:
        """Réduction à deux camps pour l'ensemble actuel de joueurs vivants."""
        key = tuple(bool(state.alive[p]) for p in range(state.n))
        if key not in self._reductions:
            self._reductions[key] = two_camp_reduction(
                self.matrix, self.config.team_index, self.player, key
            )
        return self._reductions[key]

    def engine_name(self, state: GameState) -> str:
        if self.engine == "maxn":
            return "maxn"
        red = self.reduction(state)
        if red is None:
            if self.engine == "alphabeta":
                raise ConfigError("la matrice W ne se ramène pas à deux camps")
            return "maxn"
        return "alphabeta"

    def run_search(self, state: GameState, depth: int) -> SearchResult:
        engine = self.engine_name(state)
        if engine == "alphabeta":
            red = self.reduction(state)
            assert red is not None
            return search(
                "alphabeta",
                state,
                self.valuation,
                depth,
                options=self.options,
                memory=self.memory,
                coeffs=red.coeffs,
                signs=red.signs,
                camp=red.camp,
            )
        return search(
            "maxn",
            state,
            self.valuation,
            depth,
            options=self.options,
            memory=self.memory,
            rows=self._rows,
            bound=self._bound,
        )

    def describe(self) -> str:
        extra = f",α={self.alpha}" if self.profile == "team" else ""
        unit = f"r={self.rounds}" if self.rounds is not None else f"d={self.depth}"
        return f"social[{self.profile}{extra}]({unit})"
