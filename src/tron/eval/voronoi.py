"""Territoire de Voronoi.

Parcours en largeur simultané depuis la tête de chaque joueur vivant. Chaque case
libre reçoit le masque binaire des joueurs qui l'atteignent à distance minimale
(au plus 8 joueurs : un octet). Une case revient au seul joueur de son masque ;
en cas d'égalité, elle revient au premier joueur du masque dans l'ordre de jeu à
partir du joueur au trait (mode ``order``) ou à personne (mode ``neutral``). La
tête compte dans le territoire de son joueur : tout joueur vivant a une part
strictement positive. Les parts sont normalisées sur les cases attribuées, d'où
Σ h = 1 dans les deux modes.

Variante (``separation=True``) : les joueurs dont la région n'est reliée à
aucun autre joueur sont évalués par une borne de remplissage, la parité en damier
(un chemin alterne les couleurs) : au plus 2·min(r, s) + [r > s] coups, avec r
cases de la couleur opposée à la tête et s cases de sa couleur.

Le noyau ``voronoi_kernel`` est compatible Numba ; sans Numba il s'exécute tel
quel en Python sur des listes et des ``memoryview``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tron._jit import JIT_ENABLED, njit
from tron.core.state import GameState
from tron.eval.base import Heuristic, register_heuristic

TIE_MODES = ("order", "neutral")


@njit
def voronoi_kernel(
    grid: Any,
    stride: int,
    offsets: Any,
    heads: Any,
    alive: Any,
    n: int,
    current: int,
    neutral: bool,
    separation: bool,
    dist: Any,
    mask: Any,
    stamp: Any,
    gen: int,
    queue: Any,
    counts: Any,
    link: Any,
    col0: Any,
    col1: Any,
) -> int:
    """Remplit ``counts`` (territoire par joueur) et renvoie le total attribué."""
    for p in range(n):
        counts[p] = 0
        link[p] = 1 << p
        col0[p] = 0
        col1[p] = 0
    # Sources insérées dans l'ordre de jeu à partir du joueur au trait.
    qt = 0
    for k in range(n):
        p = current + k
        if p >= n:
            p -= n
        if alive[p]:
            h = heads[p]
            stamp[h] = gen
            dist[h] = 0
            mask[h] = 1 << p
            queue[qt] = h
            qt += 1
    # Parcours en largeur multi-sources : à la sortie de la file, le masque d'une
    # case est définitif (toutes les cases de distance inférieure sont traitées).
    qh = 0
    while qh < qt:
        u = queue[qh]
        qh += 1
        du1 = dist[u] + 1
        mu = mask[u]
        for k in range(4):
            v = u + offsets[k]
            if grid[v] != -1:
                continue
            if stamp[v] != gen:
                stamp[v] = gen
                dist[v] = du1
                mask[v] = mu
                queue[qt] = v
                qt += 1
            elif dist[v] == du1:
                mask[v] = mask[v] | mu
            elif separation:
                mv = mask[v]
                if mv != mu:
                    m = mv | mu
                    for p in range(n):
                        if (m >> p) & 1:
                            link[p] = link[p] | m
    # Attribution des cases.
    total = 0
    for i in range(qt):
        c = queue[i]
        m = mask[c]
        if (m & (m - 1)) == 0:
            o = 0
            while (m >> o) != 1:
                o += 1
        else:
            if separation:
                for p in range(n):
                    if (m >> p) & 1:
                        link[p] = link[p] | m
            if neutral:
                continue
            o = -1
            for k in range(n):
                p = current + k
                if p >= n:
                    p -= n
                if (m >> p) & 1:
                    o = p
                    break
        counts[o] += 1
        total += 1
        if separation:
            r = c // stride
            if ((r + c - r * stride) & 1) == 0:
                col0[o] += 1
            else:
                col1[o] += 1
    if separation:
        # Fermeture transitive de la relation « régions reliées ».
        for _ in range(n):
            for p in range(n):
                lp = link[p]
                for q in range(n):
                    if (lp >> q) & 1:
                        lp = lp | link[q]
                link[p] = lp
        total = 0
        for p in range(n):
            if alive[p]:
                if link[p] == (1 << p):
                    h = heads[p]
                    r = h // stride
                    if ((r + h - r * stride) & 1) == 0:
                        same = col0[p] - 1
                        other = col1[p]
                    else:
                        same = col1[p] - 1
                        other = col0[p]
                    if other > same:
                        fill = 2 * same + 1
                    else:
                        fill = 2 * other
                    counts[p] = fill + 1
                total += counts[p]
    return total


@register_heuristic("voronoi")
class VoronoiHeuristic(Heuristic):
    """Parts de territoire de Voronoi, avec tableaux de travail préalloués."""

    def __init__(
        self,
        width: int,
        height: int,
        n_players: int,
        tie: str = "order",
        separation: bool = False,
    ) -> None:
        super().__init__(width, height, n_players)
        if tie not in TIE_MODES:
            raise ValueError(f"mode de départage inconnu : {tie} (attendu : {TIE_MODES})")
        self.tie = tie
        self.neutral = tie == "neutral"
        self.separation = bool(separation)
        self.size = (height + 2) * (width + 2)
        n = n_players
        self._dist: Any
        self._mask: Any
        self._stamp: Any
        self._queue: Any
        self._counts: Any
        self._link: Any
        self._col0: Any
        self._col1: Any
        if JIT_ENABLED:
            self._dist = np.zeros(self.size, dtype=np.int32)
            self._mask = np.zeros(self.size, dtype=np.uint8)
            self._stamp = np.zeros(self.size, dtype=np.int64)
            self._queue = np.zeros(self.size, dtype=np.int64)
            self._counts = np.zeros(n, dtype=np.int64)
            self._link = np.zeros(n, dtype=np.int64)
            self._col0 = np.zeros(n, dtype=np.int64)
            self._col1 = np.zeros(n, dtype=np.int64)
        else:
            self._dist = [0] * self.size
            self._mask = [0] * self.size
            self._stamp = [0] * self.size
            self._queue = [0] * self.size
            self._counts = [0] * n
            self._link = [0] * n
            self._col0 = [0] * n
            self._col1 = [0] * n
        self._gen = 0
        self.evaluations = 0

    def counts(self, state: GameState) -> tuple[list[int], int]:
        """Nombre de cases attribuées à chaque joueur et leur total."""
        self._gen += 1
        self.evaluations += 1
        if JIT_ENABLED:
            total = voronoi_kernel(
                state.grid,
                state.stride,
                state.offsets,
                state.heads,
                state.alive,
                state.n,
                state.current,
                self.neutral,
                self.separation,
                self._dist,
                self._mask,
                self._stamp,
                self._gen,
                self._queue,
                self._counts,
                self._link,
                self._col0,
                self._col1,
            )
            return self._counts.tolist(), int(total)
        total = voronoi_kernel(
            state._g,
            state.stride,
            state._off,
            state._h,
            state._a,
            state.n,
            state.current,
            self.neutral,
            self.separation,
            self._dist,
            self._mask,
            self._stamp,
            self._gen,
            self._queue,
            self._counts,
            self._link,
            self._col0,
            self._col1,
        )
        return list(self._counts), total

    def shares(self, state: GameState) -> list[float]:
        counts, total = self.counts(state)
        if total == 0:
            return [0.0] * len(counts)
        return [c / total for c in counts]

    def last_search(self) -> tuple[Any, Any, Any, int]:
        """Tableaux du dernier parcours : distances, masques, marques, génération.

        Une case ``c`` a été atteinte par le dernier appel à :meth:`counts` si
        ``stamp[c] == gen`` ; ``mask[c]`` est alors l'ensemble (bits) des joueurs
        qui l'atteignent à la distance minimale ``dist[c]``. Ces tableaux sont
        réutilisés par d'autres heuristiques (composite) sans refaire le parcours.
        """
        return self._dist, self._mask, self._stamp, self._gen

    def owner_map(self, state: GameState) -> np.ndarray:
        """Propriétaire de chaque case, tableau (H, W) ; −1 pour un mur ou une case neutre."""
        self.counts(state)
        stamp = np.asarray(self._stamp)
        mask = np.asarray(self._mask, dtype=np.int64)
        visited = stamp == self._gen
        order = state.play_order()
        lut = np.full(256, -1, dtype=np.int64)
        for m in range(1, 256):
            if m & (m - 1) == 0:
                lut[m] = m.bit_length() - 1
            elif not self.neutral:
                for p in order:
                    if (m >> p) & 1:
                        lut[m] = p
                        break
        owner = np.full(self.size, -1, dtype=np.int64)
        owner[visited] = lut[mask[visited]]
        return owner.reshape(self.height + 2, self.width + 2)[1:-1, 1:-1]


def warmup() -> None:
    """Déclenche la compilation (ou le chargement du cache) du noyau.

    À appeler avant toute mesure de temps : la compilation JIT est exclue des mesures.
    """
    from tron.core.config import GameConfig  # noqa: PLC0415 - import local pour un usage ponctuel

    cfg = GameConfig.from_dict({"grid": 5, "players": ["a", "b"]})
    state = GameState(cfg)
    for tie in TIE_MODES:
        for sep in (False, True):
            VoronoiHeuristic(5, 5, 2, tie=tie, separation=sep).counts(state)
