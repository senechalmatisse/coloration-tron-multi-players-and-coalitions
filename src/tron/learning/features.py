"""Observations et caractéristiques pour l'apprentissage, NumPy seulement.

Deux encodages, tous deux **relatifs au joueur observateur** (indice relatif 0 :
lui-même, puis les autres joueurs dans l'ordre de jeu), pour qu'une même
politique serve à toutes les places :

* :func:`observe` : observation complète destinée à un réseau ou à un outil
  externe — plans de grille (libre, bord, obstacle, murs et têtes par joueur,
  distance facultative), joueurs vivants, coéquipiers, joueur au trait, masque
  des actions légales, parts de territoire et matrice de préférences facultatives ;
* :class:`FeatureExtractor` : caractéristiques compactes d'un **état après coup**
  (afterstate), entrée du modèle linéaire de référence (:mod:`tron.learning.model`).
  Toutes les valeurs sont dans [−1, 1] et ne dépendent pas de la taille de la
  grille : un modèle entraîné sur une grille s'évalue sur une autre.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tron.core.config import GameConfig
from tron.core.rules import OBSTACLE
from tron.core.state import BORDER, FREE, GameState
from tron.eval.composite import CompositeHeuristic
from tron.eval.voronoi import VoronoiHeuristic

OBS_FORMAT = "tron-obs/1"
FEATURE_SET = "afterstate-v1"
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "own_territory",
    "own_mobility",
    "own_connectivity",
    "own_safety",
    "own_bottleneck",
    "ally_territory",
    "opp_territory",
    "opp_mobility",
    "opp_safety",
    "allies_alive",
    "opponents_alive",
    "opp_distance",
    "zone_lead",
    "in_zone",
    "survive_progress",
    "opp_survive_progress",
    "win",
    "loss",
)
"""Caractéristiques de l'état après coup, dans l'ordre du vecteur (``afterstate-v1``)."""


def relative_order(n: int, player: int) -> list[int]:
    """Joueurs dans l'ordre relatif : ``player`` puis les suivants dans l'ordre de jeu."""
    return [(player + k) % n for k in range(n)]


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Composantes facultatives de l'observation."""

    distances: bool = False
    """Plan des distances depuis la tête de l'observateur : 1 / (1 + d), 0 si inaccessible."""
    territory: bool = False
    """Parts de territoire de Voronoi (ordre relatif)."""
    preferences: bool = False
    """Matrice de préférences sociales de l'observateur (ordre relatif)."""


def channel_names(n_players: int, distances: bool = False) -> list[str]:
    """Noms des plans de :func:`encode_planes` (``walls_k`` / ``head_k`` : joueur relatif k)."""
    names = ["free", "border", "obstacle"]
    names += [f"walls_{k}" for k in range(n_players)]
    names += [f"head_{k}" for k in range(n_players)]
    if distances:
        names.append("distance")
    return names


def distance_plane(state: GameState, player: int) -> np.ndarray:
    """Distances de parcours en largeur depuis la tête de ``player`` sur les cases libres."""
    h2, w2 = state.height + 2, state.width + 2
    out = np.zeros(h2 * w2, dtype=np.float32)
    if not state.is_alive(player):
        return out.reshape(h2, w2)
    g = state._g
    start = int(state.heads[player])
    out[start] = 1.0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        c, d = queue.popleft()
        for off in state._off:
            t = c + off
            if t not in seen and g[t] == FREE:
                seen.add(t)
                out[t] = 1.0 / (2.0 + d)
                queue.append((t, d + 1))
    return out.reshape(h2, w2)


def encode_planes(state: GameState, player: int, distances: bool = False) -> np.ndarray:
    """Plans (C, H+2, W+2) ``float32`` de la grille, bordure comprise.

    Les noms des plans sont donnés par :func:`channel_names`.
    """
    n = state.n
    h2, w2 = state.height + 2, state.width + 2
    g = state.grid.reshape(h2, w2)
    planes = np.zeros((3 + 2 * n + int(distances), h2, w2), dtype=np.float32)
    planes[0] = g == FREE
    planes[1] = g == BORDER
    planes[2] = g == OBSTACLE
    for k, q in enumerate(relative_order(n, player)):
        planes[3 + k] = g == q
        if state.is_alive(q):
            y, x = divmod(int(state.heads[q]), w2)
            planes[3 + n + k, y, x] = 1.0
    if distances:
        planes[-1] = distance_plane(state, player)
    return planes


def action_mask(state: GameState, player: int) -> np.ndarray:
    """Masque des actions 0..3 légales pour ``player`` (vide si ce n'est pas son tour)."""
    mask = np.zeros(4, dtype=np.bool_)
    if state.current == player and state.is_alive(player) and not state.over:
        for m in state.legal_moves(player):
            mask[m] = True
    return mask


def observe(
    state: GameState,
    player: int,
    spec: ObservationSpec | None = None,
    *,
    territory: Sequence[float] | None = None,
    preferences: Sequence[Sequence[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Observation de ``state`` pour ``player`` (dictionnaire de tableaux NumPy).

    ``territory`` (parts absolues) et ``preferences`` (matrice absolue) sont fournis
    par l'appelant quand ``spec`` les demande ; ils sont remis dans l'ordre relatif.
    """
    spec = spec or ObservationSpec()
    n = state.n
    order = relative_order(n, player)
    team = state._team
    obs: dict[str, np.ndarray] = {
        "planes": encode_planes(state, player, spec.distances),
        "alive": np.array([state.is_alive(q) for q in order], dtype=np.float32),
        "same_team": np.array([team[q] == team[player] for q in order], dtype=np.float32),
        "current": np.array(order.index(state.current), dtype=np.int64),
        "legal": action_mask(state, player),
    }
    if spec.territory:
        if territory is None:
            raise ValueError("observation : parts de territoire demandées mais absentes")
        obs["territory"] = np.array([territory[q] for q in order], dtype=np.float32)
    if spec.preferences:
        if preferences is None:
            raise ValueError("observation : matrice de préférences demandée mais absente")
        obs["preferences"] = np.array(
            [[preferences[i][j] for j in order] for i in order], dtype=np.float32
        )
    return obs


class FeatureExtractor:
    """Caractéristiques ``afterstate-v1`` d'un état vu par un joueur (voir ``FEATURE_NAMES``)."""

    def __init__(self, config: GameConfig) -> None:
        self.n = config.n_players
        self.width = config.width
        self.height = config.height
        self.team = config.victory_team_index
        self.composite = CompositeHeuristic(config.width, config.height, config.n_players)
        self.voronoi = VoronoiHeuristic(config.width, config.height, config.n_players)
        rules = config.rules
        self.zones = rules.zones
        self.horizon = rules.target_score or rules.max_rounds or 1
        names = config.victory_names
        self.survive = {
            names.index(name): o.rounds
            for name, o in rules.objectives
            if o.type == "survive" and o.rounds is not None
        }
        self.index = {name: i for i, name in enumerate(FEATURE_NAMES)}

    @property
    def size(self) -> int:
        return len(FEATURE_NAMES)

    def afterstate(self, state: GameState, player: int) -> np.ndarray:
        """Vecteur des caractéristiques de ``state`` (état après le coup de ``player``)."""
        ix = self.index
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        vec[ix["bias"]] = 1.0
        team = self.team
        mine = team[player]
        if state.over:
            winner = state.winning_team()
            if winner == mine:
                vec[ix["win"]] = 1.0
            elif winner >= 0:
                vec[ix["loss"]] = 1.0
            return vec
        if not state.is_alive(player):
            vec[ix["loss"]] = 1.0
            return vec
        f = self.composite.features(state)
        allies = [q for q in range(self.n) if q != player and team[q] == mine]
        opps = [q for q in range(self.n) if team[q] != mine]
        alive_opps = [q for q in opps if state.is_alive(q)]
        vec[ix["own_territory"]] = f.territory[player]
        vec[ix["own_mobility"]] = f.mobility[player]
        vec[ix["own_connectivity"]] = f.connectivity[player]
        vec[ix["own_safety"]] = f.safety[player]
        vec[ix["own_bottleneck"]] = f.bottleneck[player]
        vec[ix["ally_territory"]] = sum(f.territory[q] for q in allies)
        vec[ix["opp_territory"]] = sum(f.territory[q] for q in alive_opps)
        if alive_opps:
            vec[ix["opp_mobility"]] = sum(f.mobility[q] for q in alive_opps) / len(alive_opps)
            vec[ix["opp_safety"]] = max(f.safety[q] for q in alive_opps)
            x, y = state.head(player)
            near = min(abs(x - qx) + abs(y - qy) for qx, qy in (state.head(q) for q in alive_opps))
            vec[ix["opp_distance"]] = near / (self.width + self.height)
        if allies:
            vec[ix["allies_alive"]] = sum(state.is_alive(q) for q in allies) / len(allies)
        if opps:
            vec[ix["opponents_alive"]] = len(alive_opps) / len(opps)
        if self.zones:
            own = state.scores[mine]
            best_other = max((s for t, s in enumerate(state.scores) if t != mine), default=0)
            vec[ix["zone_lead"]] = max(-1.0, min(1.0, (own - best_other) / self.horizon))
            x, y = state.head(player)
            vec[ix["in_zone"]] = float(any(z.contains(x, y) for z in self.zones))
        if self.survive:
            r = state.round
            own_goal = self.survive.get(mine)
            if own_goal is not None:
                vec[ix["survive_progress"]] = min(1.0, r / own_goal)
            others = [min(1.0, r / n) for t, n in self.survive.items() if t != mine]
            if others:
                vec[ix["opp_survive_progress"]] = max(others)
        return vec

    def action_features(
        self, state: GameState, player: int, moves: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Caractéristiques de l'état après chaque coup de ``moves`` (``state`` restauré)."""
        out: dict[int, np.ndarray] = {}
        for m in moves:
            state.apply(m)
            try:
                out[m] = self.afterstate(state, player)
            finally:
                state.undo()
        return out

    def territory(self, state: GameState) -> list[float]:
        """Parts de territoire de Voronoi (absolues) de ``state``."""
        return self.voronoi.shares(state)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SET",
    "OBS_FORMAT",
    "FeatureExtractor",
    "ObservationSpec",
    "action_mask",
    "channel_names",
    "distance_plane",
    "encode_planes",
    "observe",
    "relative_order",
]
