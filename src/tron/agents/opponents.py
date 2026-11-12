"""Modèle d'adversaire léger, explicable et déterministe (P05).

Le modèle ne devine pas une intention : il estime des probabilités de profils
à partir de choix observables. Pour chaque coup d'un adversaire q, il compare le
coup joué aux autres coups légaux de q (dans l'état d'avant le coup) selon quatre
critères, et enregistre des caractéristiques brutes.

Critères (une valeur par coup possible m, calculée après m) :

* ``aggressive`` : −Σ coups légaux des autres joueurs vivants (réduire leurs options),
  puis, à égalité, se rapprocher de la tête adverse la plus proche
  (critère lexicographique −10·Σ mobilités − distance / (W + H)) ;
* ``territorial`` : part de Voronoi de q ;
* ``survival`` : taille de la plus grande composante libre voisine de la tête de
  q, puis nombre de cases libres voisines (s'éloigner des impasses) ;
* ``cooperative`` : somme des parts de Voronoi des partenaires de q (coéquipiers
  permanents et alliés tactiques) ; non informatif sans partenaire vivant.

Alignement du coup joué m* sur un critère : (c(m*) − min c) / (max c − min c) dans
[0, 1] ; non informatif (``None``) si q n'avait qu'un coup ou si tous les coups se
valent. Caractéristiques brutes (fenêtre glissante de ``window`` décisions) :
variation de mobilité de q, variation de sa part de territoire, distance minimale
de sa nouvelle tête aux autres têtes, variation du nombre de composantes libres
voisines de sa tête, variation de la mobilité totale des autres joueurs.

Probabilités (modèle multinomial déterministe) : pour chaque critère k, score
s_k = (Σ alignements + κ·½) / (n_k + κ) sur la fenêtre (κ = ``prior`` observations
fictives d'alignement ½, ce qui tire vers l'indifférence quand les données
manquent). Logits z_k = β·(s_k − ½) pour les quatre profils comportementaux et
z_random_like = β·``margin`` ; probabilités = softmax(z). Un joueur au hasard a des
alignements de moyenne ½ et ressort ``random_like``. La confiance est la plus
grande probabilité.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from tron.agents.context import DiplomacySnapshot
from tron.core.config import GameConfig
from tron.core.state import FREE, GameState
from tron.eval.composite import CompositeHeuristic

BEHAVIOURS: tuple[str, ...] = ("aggressive", "territorial", "survival", "cooperative")
PROFILES: tuple[str, ...] = (*BEHAVIOURS, "random_like")


@dataclass(frozen=True, slots=True)
class MoveFeatures:
    """Caractéristiques observables d'un coup d'adversaire et alignements par critère."""

    player: int
    ply: int
    delta_mobility: int
    delta_territory: float
    min_head_distance: int | None
    delta_components: int
    others_mobility_delta: int
    alignment: tuple[float | None, ...]
    """Alignement sur chaque critère de :data:`BEHAVIOURS` (``None`` : non informatif)."""


def _mobility(st: GameState, p: int) -> int:
    g = st._g
    h = st._h[p]
    return sum(1 for off in st._off if g[h + off] == FREE)


def partners_of(
    config: GameConfig, player: int, diplomacy: DiplomacySnapshot | None = None
) -> tuple[int, ...]:
    """Coéquipiers permanents et alliés tactiques de ``player`` (lui exclu)."""
    out = set(config.teammates(player))
    if diplomacy is not None:
        out.update(diplomacy.allies(player))
    return tuple(sorted(out))


def criteria(
    state: GameState,
    player: int,
    moves: Sequence[int],
    heuristic: CompositeHeuristic,
    partners: Sequence[int] = (),
) -> dict[str, list[float] | None]:
    """Valeur de chaque critère pour chaque coup de ``moves`` (``player`` doit être au trait).

    ``state`` est modifié puis restauré (``apply``/``undo`` dans ``try/finally``).
    Le critère ``cooperative`` vaut ``None`` sans partenaire vivant.
    """
    if state.current != player:
        raise ValueError(f"le joueur {player} n'est pas au trait")
    out: dict[str, list[float] | None] = {k: [] for k in BEHAVIOURS}
    live_partners = [q for q in partners if q != player and state.is_alive(q)]
    if not live_partners:
        out["cooperative"] = None
    span = state.width + state.height
    for m in moves:
        state.apply(m)
        try:
            rivals = [r for r in range(state.n) if r != player and state.is_alive(r)]
            others = sum(_mobility(state, r) for r in rivals)
            hx, hy = state.head(player)
            near = min(
                (abs(hx - state.head(r)[0]) + abs(hy - state.head(r)[1]) for r in rivals),
                default=0,
            )
            if state.is_alive(player) and state.alive_players():
                f = heuristic.features(state)
                terr = f.territory[player]
                survival = f.safety[player] * max(1, f.free_cells) * 4 + _mobility(state, player)
                coop = sum(f.territory[q] for q in live_partners)
            else:  # coup suicidaire : le pire selon chaque critère individuel
                terr, survival, coop = -1.0, -1.0, 0.0
        finally:
            state.undo()
        agg = out["aggressive"]
        ter = out["territorial"]
        sur = out["survival"]
        assert agg is not None and ter is not None and sur is not None
        agg.append(-10.0 * others - near / span)
        ter.append(float(terr))
        sur.append(float(survival))
        coop_list = out["cooperative"]
        if coop_list is not None:
            coop_list.append(float(coop))
    return out


def alignment(
    values: Sequence[float] | None, chosen_index: int, eps: float = 1e-12
) -> float | None:
    """Position du coup choisi entre le pire (0) et le meilleur (1) ; ``None`` si indifférent."""
    if values is None or len(values) < 2:
        return None
    lo, hi = min(values), max(values)
    if hi - lo <= eps:
        return None
    return (values[chosen_index] - lo) / (hi - lo)


class OpponentModel:
    """Fenêtre glissante d'observations par adversaire et probabilités de profils.

    Paramètres : ``window`` (N décisions, ≥ 1), ``temperature`` β (> 0), ``prior``
    κ (≥ 0), ``margin`` (logit de ``random_like`` divisé par β). Le modèle observe
    les adversaires de ``observer`` (joueurs hors de son équipe permanente).
    """

    def __init__(
        self,
        config: GameConfig,
        observer: int,
        window: int = 20,
        temperature: float = 8.0,
        prior: float = 2.0,
        margin: float = 0.15,
    ) -> None:
        if window < 1:
            raise ValueError("la fenêtre d'observation doit être >= 1")
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError("la température doit être > 0")
        if not (math.isfinite(prior) and prior >= 0):
            raise ValueError("le poids a priori doit être >= 0")
        self.config = config
        self.observer = observer
        self.window = window
        self.temperature = temperature
        self.prior = prior
        self.margin = margin
        n = config.n_players
        team = config.team_index
        self.opponents = tuple(q for q in range(n) if team[q] != team[observer])
        self._heur = CompositeHeuristic(
            config.width, config.height, n, tie=str(config.heuristic.params.get("tie", "order"))
        )
        self._history: dict[int, deque[MoveFeatures]] = {
            q: deque(maxlen=window) for q in self.opponents
        }

    def reset(self) -> None:
        for h in self._history.values():
            h.clear()

    def history(self, player: int) -> tuple[MoveFeatures, ...]:
        self._check(player)
        return tuple(self._history.get(player, ()))

    def _check(self, player: int) -> None:
        if isinstance(player, bool) or not 0 <= player < self.config.n_players:
            raise ValueError(f"joueur inexistant : {player!r}")

    def observe(
        self,
        player: int,
        direction: int,
        origin: tuple[int, int],
        target: tuple[int, int] | None,
        eliminated: bool,
        state: GameState,
        diplomacy: DiplomacySnapshot | None = None,
    ) -> MoveFeatures | None:
        """Enregistre le coup de ``player`` ; ``state`` est l'état **après** le coup.

        ``state`` n'est jamais modifié : l'état d'avant est reconstruit sur une copie
        (historique requis). Renvoie ``None`` si le coup n'est pas analysable
        (joueur de l'équipe de l'observateur, élimination, historique absent).
        """
        self._check(player)
        if player not in self._history or eliminated or target is None or not state._hist:
            return None
        before = state.copy(history=True)
        before.undo()
        if before.current != player or before.head(player) != origin:
            raise ValueError("état incohérent avec le coup observé")
        moves = before.legal_moves()
        if direction not in moves:
            return None
        partners = partners_of(self.config, player, diplomacy)
        values = criteria(before, player, moves, self._heur, partners)
        idx = moves.index(direction)
        aligns = tuple(alignment(values[k], idx) for k in BEHAVIOURS)
        f_before = self._heur.features(before)
        f_after = self._heur.features(state) if state.alive_players() else f_before
        others = [r for r in range(state.n) if r != player]
        heads = [state.head(r) for r in others if state.is_alive(r)]
        tx, ty = target
        dist = min((abs(tx - x) + abs(ty - y) for x, y in heads), default=None)
        mob_before = sum(_mobility(before, r) for r in others if before.is_alive(r))
        mob_after = sum(_mobility(state, r) for r in others if state.is_alive(r))
        feat = MoveFeatures(
            player=player,
            ply=before.ply,
            delta_mobility=_mobility(state, player) - _mobility(before, player),
            delta_territory=f_after.territory[player] - f_before.territory[player],
            min_head_distance=dist,
            delta_components=round(
                2 * (f_after.connectivity[player] - f_before.connectivity[player])
            ),
            others_mobility_delta=mob_after - mob_before,
            alignment=aligns,
        )
        self._history[player].append(feat)
        return feat

    def scores(self, player: int) -> tuple[float, ...]:
        """Score s_k de chaque critère (voir le module), ½ sans observation informative."""
        self._check(player)
        hist = self._history.get(player, ())
        out = []
        for k in range(len(BEHAVIOURS)):
            vals = [f.alignment[k] for f in hist if f.alignment[k] is not None]
            total = sum(v for v in vals if v is not None)
            out.append(
                (total + 0.5 * self.prior) / (len(vals) + self.prior) if vals or self.prior else 0.5
            )
        return tuple(out)

    def probabilities(self, player: int) -> dict[str, float]:
        """Probabilités des profils de :data:`PROFILES` (finies, dans [0, 1], de somme 1)."""
        beta = self.temperature
        logits = [beta * (s - 0.5) for s in self.scores(player)]
        logits.append(beta * self.margin)
        top = max(logits)
        exps = [math.exp(z - top) for z in logits]
        total = sum(exps)
        return {name: e / total for name, e in zip(PROFILES, exps, strict=True)}

    def baseline(self) -> dict[str, float]:
        """Probabilités sans aucune observation (scores à ½) : référence « sans information »."""
        beta = self.temperature
        logits = [0.0] * len(BEHAVIOURS) + [beta * self.margin]
        top = max(logits)
        exps = [math.exp(z - top) for z in logits]
        total = sum(exps)
        return {name: e / total for name, e in zip(PROFILES, exps, strict=True)}

    def confidence(self, player: int) -> float:
        return max(self.probabilities(player).values())

    def most_likely(self, player: int) -> str:
        probs = self.probabilities(player)
        return max(PROFILES, key=lambda k: (probs[k], -PROFILES.index(k)))


__all__ = [
    "BEHAVIOURS",
    "PROFILES",
    "MoveFeatures",
    "OpponentModel",
    "alignment",
    "criteria",
    "partners_of",
]
