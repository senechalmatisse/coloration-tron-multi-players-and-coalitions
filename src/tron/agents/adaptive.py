"""Agent adaptatif (P05) : ajustement borné des poids de l'heuristique selon les adversaires.

L'agent délègue la recherche à un agent de repli (``fallback_agent`` : ``social``,
``maxn``, ``paranoid`` ou ``greedy``) qui évalue les feuilles avec une heuristique
composite partagée. Avant chaque décision, il ajuste les poids de cette
heuristique à partir des probabilités de profils estimées par
:class:`~tron.agents.opponents.OpponentModel` :

    w_t = (1 − λ)·w_base + λ·moyenne_q[ w_base + Σ_k e_q(k)·(T_k − w_base) ]

* λ = ``adaptation_rate`` ∈ [0, 1] borne l'ajustement ;
* w_base = poids de la configuration si l'heuristique est ``composite``, sinon le
  seul territoire (l'heuristique vaut alors exactement Voronoi) ;
* e_q(k) = max(0, p_q(k) − p⁰(k)) : excès de probabilité du profil k par rapport
  aux probabilités sans aucune observation p⁰ ; un adversaire non observé ou mal
  identifié ne déplace donc pas les poids. Comme Σ_k e_q(k) ≤ 1, w_t reste une
  combinaison convexe de w_base et des cibles (poids positifs, de somme 1) ;
* T_k (:data:`TARGET_WEIGHTS`) : contre un adversaire ``aggressive``, plus de
  marge et de mobilité (échappatoires) ; ``territorial`` : plus de contrôle des
  passages ; ``survival`` : plus de territoire et de sorties pour couper ses
  échappatoires ; ``cooperative`` et ``random_like`` : poids de base ;
* si aucune confiance n'atteint ``min_confidence`` (ou si λ = 0), w_t = w_base.

Tant que w_t = w_base, les feuilles sont évaluées avec l'heuristique de la
configuration elle-même (:class:`AdaptiveHeuristic`) : l'agent reproduit alors
**exactement** les décisions de son agent de repli (même flux aléatoire). Dès
que les poids s'écartent de w_base, l'heuristique composite ajustée prend le relais.

Diplomatie (P03, si elle est active et ``diplomacy`` vrai) : la politique
:func:`~tron.agents.social.tactical_intent` est suivie ; une rupture reste
prioritaire, sinon l'agent propose une alliance à l'adversaire le plus
probablement ``cooperative`` (probabilité ≥ ``cooperation_threshold``) s'il est
éligible. L'agent de repli ``social`` utilise alors la matrice dynamique W_t.
"""

from __future__ import annotations

from typing import Any

from tron.agents.base import (
    AGENTS,
    Agent,
    DecisionDetails,
    SearchStats,
    make_agent,
    register_agent,
)
from tron.agents.context import DecisionContext, DiplomacyIntent, DiplomacySnapshot
from tron.agents.opponents import OpponentModel
from tron.agents.social import tactical_intent
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState
from tron.eval.base import Heuristic, make_heuristic
from tron.eval.composite import COMPONENTS, CompositeHeuristic, normalize_weights

FALLBACK_AGENTS = ("social", "maxn", "paranoid", "greedy")

TARGET_WEIGHTS: dict[str, dict[str, float]] = {
    "aggressive": {
        "territory": 0.35,
        "mobility": 0.25,
        "connectivity": 0.10,
        "safety": 0.25,
        "bottleneck": 0.05,
    },
    "territorial": {
        "territory": 0.45,
        "mobility": 0.10,
        "connectivity": 0.10,
        "safety": 0.10,
        "bottleneck": 0.25,
    },
    "survival": {
        "territory": 0.50,
        "mobility": 0.10,
        "connectivity": 0.20,
        "safety": 0.10,
        "bottleneck": 0.10,
    },
}
"""Poids cibles par profil d'adversaire (``cooperative`` et ``random_like`` : poids de base)."""


def base_weights(config: GameConfig) -> dict[str, float]:
    """Poids de départ : ceux de l'heuristique composite configurée, sinon le seul territoire."""
    spec = config.heuristic
    if spec.name == "composite":
        return normalize_weights({k: v for k, v in spec.params.items() if k in COMPONENTS})
    return {name: 1.0 if name == "territory" else 0.0 for name in COMPONENTS}


def adapted_weights(
    base: dict[str, float],
    probabilities: list[dict[str, float]],
    rate: float,
    min_confidence: float,
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    """Poids w_t (voir le module) ; ``base`` si λ = 0, sans adversaire ou trop peu confiant.

    ``baseline`` : probabilités sans observation p⁰ (uniformes par défaut).
    """
    if rate <= 0.0 or not probabilities:
        return dict(base)
    if max(max(p.values()) for p in probabilities) < min_confidence:
        return dict(base)
    ref = baseline or {k: 1.0 / len(probabilities[0]) for k in probabilities[0]}
    mix = dict.fromkeys(COMPONENTS, 0.0)
    for probs in probabilities:
        excess = {k: max(0.0, p - ref.get(k, 0.0)) for k, p in probs.items()}
        for comp in COMPONENTS:
            shift = sum(
                e * (TARGET_WEIGHTS[k][comp] - base[comp])
                for k, e in excess.items()
                if k in TARGET_WEIGHTS
            )
            mix[comp] += (base[comp] + shift) / len(probabilities)
    return {c: (1.0 - rate) * base[c] + rate * mix[c] for c in COMPONENTS}


class AdaptiveHeuristic(Heuristic):
    """Heuristique de la configuration tant que les poids valent w_base, composite ajustée sinon."""

    def __init__(self, config: GameConfig) -> None:
        super().__init__(config.width, config.height, config.n_players)
        params = config.heuristic.params
        self.base = make_heuristic(config.heuristic, config.width, config.height, config.n_players)
        self.composite = CompositeHeuristic(
            config.width,
            config.height,
            config.n_players,
            tie=str(params.get("tie", "order")),
            separation=bool(params.get("separation", False)),
            **base_weights(config),
        )
        self.base_weights = dict(self.composite.weights)
        self.adapted = False

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.composite.weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        self.composite.set_weights(weights)
        self.adapted = self.composite.weights != self.base_weights

    def shares(self, state: GameState) -> list[float]:
        return self.composite.shares(state) if self.adapted else self.base.shares(state)


@register_agent("adaptive")
class AdaptiveAgent(Agent):
    """Agent adaptatif à modèle d'adversaires (voir le module).

    Paramètres : ``adaptation_rate`` λ ∈ [0, 1], ``observation_window`` (≥ 1),
    ``fallback_agent``, ``fallback_profile`` (profil de l'agent ``social``),
    ``min_confidence`` et ``cooperation_threshold`` dans [0, 1], ``diplomacy``.
    Les autres paramètres (``depth``, ``alpha``, ``engine``…) vont à l'agent de repli.
    """

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        adaptation_rate: float = 0.25,
        observation_window: int = 20,
        fallback_agent: str = "social",
        fallback_profile: str = "team",
        min_confidence: float = 0.35,
        cooperation_threshold: float = 0.4,
        diplomacy: bool = True,
        depth: int = 1,
        rounds: int | None = None,
        tie_break: str = "random",
        seed: int | None = None,
        heuristic: Heuristic | None = None,
        **inner_params: Any,
    ) -> None:
        if heuristic is not None:
            raise ConfigError("l'agent adaptatif construit sa propre heuristique composite")
        for name, value in (
            ("adaptation_rate", adaptation_rate),
            ("min_confidence", min_confidence),
            ("cooperation_threshold", cooperation_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"{name} doit être un nombre : {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} doit être dans [0, 1] : {value!r}")
        if isinstance(observation_window, bool) or not isinstance(observation_window, int):
            raise ConfigError(f"observation_window doit être un entier : {observation_window!r}")
        if observation_window < 1:
            raise ConfigError("observation_window doit être >= 1")
        if fallback_agent not in FALLBACK_AGENTS or fallback_agent not in AGENTS:
            raise ConfigError(
                f"agent de repli inconnu : {fallback_agent} (attendu : {FALLBACK_AGENTS})"
            )
        try:
            heur = AdaptiveHeuristic(config)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        super().__init__(
            player,
            config,
            depth=depth,
            rounds=rounds,
            tie_break=tie_break,
            seed=seed,
            heuristic=heur,
        )
        self.adaptive_heuristic = heur
        self.adaptation_rate = float(adaptation_rate)
        self.min_confidence = float(min_confidence)
        self.cooperation_threshold = float(cooperation_threshold)
        self.diplomatic = bool(diplomacy)
        self.fallback_agent = fallback_agent
        inner = dict(inner_params)
        if fallback_agent == "social":
            inner.setdefault("profile", fallback_profile)
            inner.setdefault("diplomacy", self.diplomatic)
        self.inner = make_agent(
            fallback_agent,
            player,
            config,
            depth=depth,
            rounds=rounds,
            tie_break=tie_break,
            seed=seed,
            heuristic=heur,
            **inner,
        )
        self.base_weights = dict(heur.base_weights)
        self.weights_used = dict(self.base_weights)
        self.model = OpponentModel(config, player, window=observation_window)
        self._snapshot: DiplomacySnapshot | None = None

    # -------------------------------------------------------------- mémoire
    def reset(self) -> None:
        self.inner.reset()
        self.model.reset()
        self.adaptive_heuristic.set_weights(self.base_weights)
        self.weights_used = dict(self.base_weights)
        self._snapshot = None

    def observe_move(
        self,
        player: int,
        direction: int,
        origin: tuple[int, int],
        target: tuple[int, int] | None,
        eliminated: bool,
        state: GameState,
    ) -> None:
        self.model.observe(player, direction, origin, target, eliminated, state, self._snapshot)

    def opponent_probabilities(self, state: GameState) -> dict[int, dict[str, float]]:
        """Probabilités de profils des adversaires vivants."""
        return {q: self.model.probabilities(q) for q in self.model.opponents if state.is_alive(q)}

    # ------------------------------------------------------------- décision
    def _adapt(self, state: GameState) -> None:
        probs = list(self.opponent_probabilities(state).values())
        w = adapted_weights(
            self.base_weights,
            probs,
            self.adaptation_rate,
            self.min_confidence,
            self.model.baseline(),
        )
        self.adaptive_heuristic.set_weights(w)
        self.weights_used = self.adaptive_heuristic.weights

    def _decide_with_context(
        self, state: GameState, moves: list[int], context: DecisionContext | None
    ) -> tuple[int, SearchStats]:
        if context is not None:
            self._snapshot = context.diplomacy
        self._adapt(state)
        self.inner._details = None
        out = self.inner._decide_with_context(state, moves, context)
        self._details = self._explain(state, self.inner._details)
        return out

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        self._adapt(state)
        self.inner._details = None
        out = self.inner._decide(state, moves)
        self._details = self._explain(state, self.inner._details)
        return out

    def _explain(self, state: GameState, inner: DecisionDetails | None) -> DecisionDetails:
        base = inner or DecisionDetails()
        opponents = []
        for q, probs in self.opponent_probabilities(state).items():
            k = max(probs, key=lambda name: probs[name])
            opponents.append((q, k, probs[k]))
        return DecisionDetails(
            scores=base.scores,
            visits=base.visits,
            social_weights=base.social_weights,
            allies=base.allies,
            heuristic_weights=tuple(sorted(self.weights_used.items())),
            opponents=tuple(opponents),
            note=base.note,
        )

    def _diplomacy(
        self, state: GameState, move: int, context: DecisionContext
    ) -> DiplomacyIntent | None:
        snap = context.diplomacy
        if not (self.diplomatic and self.config.diplomacy.enabled and snap is not None):
            return None
        spec = self.config.diplomacy
        team = self.config.team_index
        alive = [state.is_alive(q) for q in range(state.n)]
        base = tactical_intent(
            self.player,
            team,
            alive,
            self.adaptive_heuristic.shares(state),
            snap,
            spec.max_allies_per_player,
        )
        if base is not None and base.kind == "break":
            return base
        if len(snap.allies(self.player)) >= spec.max_allies_per_player:
            return base
        best: int | None = None
        best_p = self.cooperation_threshold
        for q in self.model.opponents:
            if (
                not alive[q]
                or snap.are_allied(self.player, q)
                or snap.has_pending(self.player, q)
                or snap.in_cooldown(self.player, q)
            ):
                continue
            p = self.model.probabilities(q)["cooperative"]
            if p >= best_p:
                best, best_p = q, p
        return DiplomacyIntent("propose", best) if best is not None else base

    def describe(self) -> str:
        return f"adaptive[{self.inner.describe()},λ={self.adaptation_rate:g}]"


__all__ = [
    "FALLBACK_AGENTS",
    "TARGET_WEIGHTS",
    "AdaptiveAgent",
    "AdaptiveHeuristic",
    "adapted_weights",
    "base_weights",
]
