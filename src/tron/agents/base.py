"""Interface commune des agents et registre."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, ClassVar

from tron.agents.context import DecisionContext, DiplomacyIntent
from tron.core.config import ConfigError, GameConfig, HeuristicSpec
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


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class DecisionDetails:
    """Explication typée et sérialisable d'une décision (P07).

    Un agent ne renseigne que ce qu'il a réellement calculé. Dans ``scores``,
    ``None`` signifie « non disponible » (coup élagué ou seulement borné par la
    recherche) : aucune valeur n'est reconstruite après coup.

    * ``scores`` : (direction, valeur pour l'agent) de chaque coup légal à la racine ;
    * ``visits`` : (direction, visites) des coups racine (MCTS) ;
    * ``social_weights`` : coefficients non nuls de la ligne W[p] utilisée ;
    * ``allies`` : alliés tactiques au moment de la décision ;
    * ``heuristic_weights`` : poids de l'heuristique composite utilisés (agent adaptatif) ;
    * ``opponents`` : (adversaire, profil le plus probable, probabilité) ;
    * ``note`` : remarque courte (coup forcé, coup tiré au hasard…).
    """

    scores: tuple[tuple[int, float | None], ...] = ()
    visits: tuple[tuple[int, int], ...] = ()
    social_weights: tuple[tuple[int, float], ...] = ()
    allies: tuple[int, ...] = ()
    heuristic_weights: tuple[tuple[str, float], ...] = ()
    opponents: tuple[tuple[int, str, float], ...] = ()
    note: str = ""

    def score_of(self, move: int) -> float | None:
        return next((v for m, v in self.scores if m == move), None)

    def ranked(self) -> list[tuple[int, float | None]]:
        """Coups racine du meilleur au moins bon ; les valeurs non disponibles en dernier."""
        known = [(m, v) for m, v in self.scores if v is not None]
        known.sort(key=lambda x: -(x[1] or 0.0))
        return known + [(m, v) for m, v in self.scores if v is None]

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON compacte (champs vides omis, valeurs non finies → ``null``)."""
        data: dict[str, Any] = {}
        if self.scores:
            data["scores"] = [[m, _finite(v)] for m, v in self.scores]
        if self.visits:
            data["visits"] = [[m, k] for m, k in self.visits]
        if self.social_weights:
            data["social_weights"] = [[j, w] for j, w in self.social_weights]
        if self.allies:
            data["allies"] = list(self.allies)
        if self.heuristic_weights:
            data["heuristic_weights"] = {k: w for k, w in self.heuristic_weights}
        if self.opponents:
            data["opponents"] = [[q, k, p] for q, k, p in self.opponents]
        if self.note:
            data["note"] = self.note
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionDetails:
        hw = data.get("heuristic_weights") or {}
        return cls(
            scores=tuple(
                (int(m), None if v is None else float(v)) for m, v in data.get("scores", ())
            ),
            visits=tuple((int(m), int(k)) for m, k in data.get("visits", ())),
            social_weights=tuple((int(j), float(w)) for j, w in data.get("social_weights", ())),
            allies=tuple(int(a) for a in data.get("allies", ())),
            heuristic_weights=tuple((str(k), float(w)) for k, w in hw.items()),
            opponents=tuple((int(q), str(k), float(p)) for q, k, p in data.get("opponents", ())),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """Coup choisi (direction 0 à 3, ou PASS), statistiques, intention et explication."""

    move: int
    stats: SearchStats
    diplomacy: DiplomacyIntent | None = None
    details: DecisionDetails | None = None


class Agent(ABC):
    """Agent : ``choose_move(state, player) → Decision``.

    Paramètres communs : ``depth`` (plies), ``rounds`` (tours complets : la
    profondeur vaut alors rounds × joueurs vivants), ``tie_break`` (``random``
    tiré avec la graine de la partie, ou ``fixed`` = ordre N, E, S, O) et
    ``heuristic`` (instance, nom ou ``{name, ...}`` : heuristique propre au
    joueur, sinon celle de la configuration).
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
        heuristic: Heuristic | str | Mapping[str, Any] | None = None,
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
        if heuristic is None or isinstance(heuristic, str | Mapping):
            spec = config.heuristic if heuristic is None else _heuristic_spec(heuristic)
            try:
                heuristic = make_heuristic(spec, config.width, config.height, config.n_players)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        self.heuristic: Heuristic = heuristic
        self.valuation = Valuation(config, self.heuristic)
        self._details: DecisionDetails | None = None
        """Explication de la décision en cours, renseignée par ``_decide`` (P07)."""

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
        self._details = None
        move, stats = self._decide_with_context(state, moves, context)
        details = self._details
        intent = self._diplomacy(state, move, context) if context is not None else None
        elapsed = perf_counter() - t0
        return Decision(move, _with_time(stats, elapsed), intent, details)

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
        """Départage des coups de même valeur."""
        ordered = sorted(ties)
        if self.tie_break == "fixed" or len(ordered) == 1:
            return ordered[0]
        return ordered[int(self.rng.integers(len(ordered)))]

    def reset(self) -> None:
        """Réinitialise la mémoire de l'agent (nouvelle partie)."""

    def snapshot_state(self) -> dict[str, Any]:
        """État interne nécessaire à une reprise exacte (P08), sérialisable en JSON.

        Par défaut : l'état du générateur aléatoire de l'agent. Un agent qui garde
        une mémoire entre ses décisions l'ajoute (voir :meth:`restore_state`).
        """
        return {"rng": self.rng.bit_generator.state}

    def restore_state(self, data: Mapping[str, Any]) -> None:
        """Restaure l'état produit par :meth:`snapshot_state` (après ``reset``)."""
        try:
            self.rng.bit_generator.state = dict(data["rng"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"état aléatoire invalide pour l'agent {self.name}") from exc

    def observe_move(
        self,
        player: int,
        direction: int,
        origin: tuple[int, int],
        target: tuple[int, int] | None,
        eliminated: bool,
        state: GameState,
    ) -> None:
        """Crochet d'observation appelé après chaque demi-coup de la partie.

        ``state`` est l'état réel **après** le coup : il ne doit pas être modifié
        (travailler sur ``state.copy()``). Les agents qui n'en ont pas besoin
        n'implémentent rien.
        """

    def describe(self) -> str:
        return f"{self.name}(d={self.depth})"


def _heuristic_spec(value: str | Mapping[str, Any]) -> HeuristicSpec:
    """Heuristique propre à un joueur (clé ``heuristic`` du joueur dans le YAML)."""
    if isinstance(value, str):
        return HeuristicSpec(name=value)
    params = {k: v for k, v in value.items() if k != "name"}
    return HeuristicSpec(name=str(value.get("name", "voronoi")), params=params)


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


# registre
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
        hint = ""
        if ":" in name or "." in name or "/" in name:
            hint = (
                " ; un chemin de module n'est jamais chargé depuis une configuration : "
                "déclarer un entry point tron.agents"
            )
        raise ConfigError(f"agent inconnu : {name} (connus : {sorted(AGENTS)}){hint}") from None
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
