"""Agents de recherche : Negamax, Paranoid, MaxN et SocialAgent (F3.3 à F3.6, F4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from tron.agents.base import Agent, DecisionDetails, SearchStats, register_agent
from tron.agents.context import DecisionContext, DiplomacyIntent
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
    dynamic_matrix,
    is_identity,
    social_matrix,
    sparse_rows,
    tactical_intent,
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
        aspiration: float | None = None,
        reduction: int = 0,
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
                aspiration=aspiration,
                reduction=reduction,
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

    def snapshot_state(self) -> dict[str, Any]:
        data = super().snapshot_state()
        mem = self.memory
        data["tt"] = [[k, v] for k, v in mem.tt.items()] if mem.tt is not None else None
        data["history"] = list(mem.history) if mem.history is not None else None
        return data

    def restore_state(self, data: Mapping[str, Any]) -> None:
        super().restore_state(data)
        mem = self.memory
        if mem.tt is not None:
            mem.tt.clear()
            mem.tt.update((int(k), int(v)) for k, v in data.get("tt") or [])
        hist = data.get("history")
        if mem.history is not None and hist is not None:
            if len(hist) != len(mem.history):
                raise ValueError("mémoire d'historique de taille incohérente")
            mem.history[:] = [int(x) for x in hist]

    def explain(self, result: SearchResult | None, moves: list[int]) -> DecisionDetails:
        """Valeurs racine exactes (``None`` pour un coup seulement borné par l'élagage)."""
        if result is None:
            return DecisionDetails(scores=tuple((m, None) for m in moves), note="coup forcé")
        exact = set(result.exact)
        return DecisionDetails(
            scores=tuple((m, result.root_values.get(m) if m in exact else None) for m in moves),
        )

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        if len(moves) == 1 and self.forced_shortcut:
            self._details = self.explain(None, moves)
            return moves[0], SearchStats(nodes=1, depth=0, engine=self.engine_name(state))
        result = self.run_search(state, self.effective_depth(state))
        self.last_result = result
        self._details = self.explain(result, moves)
        move = self.pick(result.ties)
        # Variante principale du coup réellement choisi (parmi les ex aequo).
        pv = result.pvs.get(move) or (result.pv if move == result.best else [move])
        return move, SearchStats(
            nodes=result.nodes,
            leaves=result.leaves,
            prunes=result.prunes,
            value=result.root_values.get(move, result.value),
            depth=result.depth,
            ties=len(result.ties),
            engine=result.engine,
            pv=tuple(pv),
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
    ``custom``, ``dynamic_team``), ``alpha`` (profils ``team`` et ``dynamic_team``),
    ``matrix`` (profil ``custom``), ``engine`` (``auto``, ``alphabeta``, ``maxn``).

    Diplomatie (P03) : avec ``diplomacy`` vrai (par défaut pour ``dynamic_team``)
    et la diplomatie de la partie activée, chaque décision utilise la matrice
    dynamique W_t (:func:`~tron.agents.social.dynamic_matrix`) et se termine par
    une intention de :func:`~tron.agents.social.tactical_intent` (paramètre
    ``break_margin``). Sinon, la matrice statique historique est utilisée.
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
        diplomacy: bool | None = None,
        break_margin: float = 0.15,
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
        if not 0.0 <= break_margin <= 1.0:
            raise ConfigError("break_margin doit être dans [0, 1]")
        self.profile = profile
        self.alpha = alpha
        self.engine = engine
        self.diplomatic = (profile == "dynamic_team") if diplomacy is None else bool(diplomacy)
        self.break_margin = break_margin
        self.base_matrix: Matrix = [row[:] for row in w]
        self.matrix: Matrix = w
        self._allies: tuple[int, ...] = ()
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

    def reset(self) -> None:
        super().reset()
        self.set_matrix(self.base_matrix)

    # ------------------------------------------------------- matrice dynamique
    def set_matrix(self, w: Matrix) -> None:
        """Remplace W (et les structures dérivées) si elle a changé."""
        if w == self.matrix:
            return
        self.matrix = [row[:] for row in w]
        self._identity = is_identity(self.matrix)
        self._rows = sparse_rows(self.matrix)
        self._reductions = {}
        self._bound = self._make_bound()

    def _diplomacy_active(self, context: DecisionContext | None) -> bool:
        return (
            self.diplomatic
            and self.config.diplomacy.enabled
            and context is not None
            and context.diplomacy is not None
        )

    def explain(self, result: SearchResult | None, moves: list[int]) -> DecisionDetails:
        base = super().explain(result, moves)
        row = self.matrix[self.player]
        return DecisionDetails(
            scores=base.scores,
            social_weights=tuple((j, w) for j, w in enumerate(row) if w != 0.0),
            allies=self._allies,
            note=base.note,
        )

    def _decide_with_context(
        self, state: GameState, moves: list[int], context: DecisionContext | None
    ) -> tuple[int, SearchStats]:
        snap = context.diplomacy if context is not None else None
        self._allies = snap.allies(self.player) if snap is not None else ()
        if self._diplomacy_active(context):
            assert context is not None and context.diplomacy is not None
            spec = self.config.diplomacy
            self.set_matrix(
                dynamic_matrix(
                    self.base_matrix,
                    context.diplomacy,
                    spec.alliance_weight,
                    spec.betrayal_penalty,
                )
            )
        else:
            self.set_matrix(self.base_matrix)
        return self._decide(state, moves)

    def _diplomacy(
        self, state: GameState, move: int, context: DecisionContext
    ) -> DiplomacyIntent | None:
        if not self._diplomacy_active(context):
            return None
        assert context.diplomacy is not None
        return tactical_intent(
            self.player,
            self.config.team_index,
            [state.is_alive(q) for q in range(state.n)],
            self.heuristic.shares(state),
            context.diplomacy,
            self.config.diplomacy.max_allies_per_player,
            self.break_margin,
        )

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
        extra = f",α={self.alpha}" if self.profile in ("team", "dynamic_team") else ""
        unit = f"r={self.rounds}" if self.rounds is not None else f"d={self.depth}"
        return f"social[{self.profile}{extra}]({unit})"
