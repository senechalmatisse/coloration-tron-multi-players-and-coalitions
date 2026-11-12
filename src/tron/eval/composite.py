"""Heuristique territoriale multicritère (P01).

Pour chaque joueur vivant p, cinq caractéristiques bornées dans [0, 1] :

* ``territory`` : part de Voronoi (tête comprise). Le parcours en largeur est
  celui de :class:`~tron.eval.voronoi.VoronoiHeuristic`, réutilisé tel quel ;
* ``mobility`` : nombre de cases libres voisines de la tête, divisé par 4 ;
* ``connectivity`` : nombre de composantes libres distinctes voisines de la tête
  (sorties vers des régions séparées), ramené à {0, ½, 1} par min(c, 2)/2 ;
* ``safety`` : taille de la plus grande composante libre voisine de la tête,
  divisée par le nombre total de cases libres (marge avant enfermement) ;
* ``bottleneck`` : part des points d'articulation du graphe des cases libres
  (cases dont la suppression déconnecte leur composante) que p atteint le
  premier au sens de Voronoi, avec la même règle d'égalité que ``territory``.

Score brut : raw(p) = Σ_k w_k·x_k(p). Les poids doivent être finis, positifs ou
nuls, de somme strictement positive ; ils sont **normalisés automatiquement** à
une somme de 1. Un poids absent prend sa valeur par défaut (:data:`DEFAULT_WEIGHTS`).

Un plancher raw(p) ≥ 1/(4·W·H) garantit une part strictement positive à tout
joueur vivant : l'ordre victoire > heuristique > défaite de
:class:`~tron.eval.valuation.Valuation` reste vrai même avec un poids de
territoire nul (la plus petite part, au moins 1/(4·W·H·n), dépasse toujours ε).

Parts : h(p) = raw(p) / Σ_vivants raw(q), 0 pour les joueurs éliminés, Σ h = 1.
Si la somme est nulle (cas défensif, sans joueur vivant), les parts sont nulles.

Coût : un parcours en largeur (Voronoi) et un parcours en profondeur itératif
(composantes et points d'articulation, algorithme de Tarjan) par état, partagés
par tous les joueurs. Le noyau est écrit dans le style Numba ; si Numba refuse
de le compiler, il s'exécute en Python pur (:func:`tron._jit.safe_njit`).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tron._jit import JIT_ENABLED, safe_njit
from tron.core.state import GameState
from tron.eval.base import Heuristic, register_heuristic
from tron.eval.voronoi import TIE_MODES, VoronoiHeuristic

COMPONENTS: tuple[str, ...] = ("territory", "mobility", "connectivity", "safety", "bottleneck")
DEFAULT_WEIGHTS: dict[str, float] = {
    "territory": 0.55,
    "mobility": 0.15,
    "connectivity": 0.10,
    "safety": 0.15,
    "bottleneck": 0.05,
}


def normalize_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    """Valide les poids (finis, ≥ 0, somme > 0) et les ramène à une somme de 1.

    Un poids absent prend sa valeur par défaut. Lève ``ValueError`` pour une
    composante inconnue, un poids non numérique, négatif, NaN ou infini, ou une
    somme nulle.
    """
    unknown = set(weights) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"composantes inconnues : {sorted(unknown)} (attendu : {COMPONENTS})")
    out: dict[str, float] = {}
    for name in COMPONENTS:
        value = weights.get(name, DEFAULT_WEIGHTS[name])
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"poids '{name}' non numérique : {value!r}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"poids '{name}' non fini : {value!r}")
        if v < 0.0:
            raise ValueError(f"poids '{name}' négatif : {value!r}")
        out[name] = v
    total = sum(out.values())
    if total <= 0.0:
        raise ValueError("la somme des poids de l'heuristique composite doit être > 0")
    return {k: v / total for k, v in out.items()}


@safe_njit
def composite_kernel(
    grid: Any,
    size: int,
    offsets: Any,
    heads: Any,
    alive: Any,
    n: int,
    order: Any,
    n_order: int,
    neutral: bool,
    vmask: Any,
    vstamp: Any,
    vgen: int,
    gen: int,
    disc_gen: Any,
    disc: Any,
    low: Any,
    parent: Any,
    it: Any,
    stack: Any,
    comp: Any,
    comp_size: Any,
    ap_gen: Any,
    mob: Any,
    conn: Any,
    safe: Any,
    bott: Any,
    totals: Any,
) -> int:
    """Composantes, points d'articulation et caractéristiques ; renvoie le nombre de composantes.

    Sorties : ``mob``, ``conn``, ``safe`` et ``bott`` (comptes bruts par
    joueur), ``totals[0]`` (cases libres) et ``totals[1]`` (points d'articulation).
    """
    # 1. Parcours en profondeur itératif (Tarjan) de chaque composante libre.
    t = 0
    n_comp = 0
    n_free = 0
    n_ap = 0
    for s in range(size):
        if grid[s] != -1 or disc_gen[s] == gen:
            continue
        disc_gen[s] = gen
        disc[s] = t
        low[s] = t
        t += 1
        parent[s] = -1
        it[s] = 0
        comp[s] = n_comp
        csize = 1
        root_children = 0
        sp = 1
        stack[0] = s
        while sp > 0:
            u = stack[sp - 1]
            k = it[u]
            if k < 4:
                it[u] = k + 1
                v = u + offsets[k]
                if grid[v] != -1:
                    continue
                if disc_gen[v] != gen:
                    disc_gen[v] = gen
                    disc[v] = t
                    low[v] = t
                    t += 1
                    parent[v] = u
                    it[v] = 0
                    comp[v] = n_comp
                    csize += 1
                    if u == s:
                        root_children += 1
                    stack[sp] = v
                    sp += 1
                elif v != parent[u] and disc[v] < low[u]:
                    low[u] = disc[v]
            else:
                sp -= 1
                w = parent[u]
                if w >= 0:
                    if low[u] < low[w]:  # noqa: PLR1730 - style Numba explicite
                        low[w] = low[u]
                    if w != s and low[u] >= disc[w] and ap_gen[w] != gen:
                        ap_gen[w] = gen
                        n_ap += 1
        if root_children >= 2:
            ap_gen[s] = gen
            n_ap += 1
        comp_size[n_comp] = csize
        n_free += csize
        n_comp += 1
    # 2. Mobilité, sorties distinctes et marge de chaque joueur vivant.
    for p in range(n):
        mob[p] = 0
        conn[p] = 0
        safe[p] = 0
        bott[p] = 0
    for p in range(n):
        if not alive[p]:
            continue
        h = heads[p]
        m = 0
        best = 0
        nd = 0
        c0 = -1
        c1 = -1
        c2 = -1
        for k in range(4):
            v = h + offsets[k]
            if grid[v] != -1:
                continue
            m += 1
            c = comp[v]
            if c != c0 and c != c1 and c != c2:  # noqa: PLR1714 - style Numba explicite
                if nd == 0:
                    c0 = c
                elif nd == 1:
                    c1 = c
                elif nd == 2:
                    c2 = c
                nd += 1
                if comp_size[c] > best:  # noqa: PLR1730 - style Numba explicite
                    best = comp_size[c]
        mob[p] = m
        conn[p] = nd
        safe[p] = best
    # 3. Points d'articulation contrôlés : propriétaire de Voronoi de la case.
    if n_ap > 0:
        for c in range(size):
            if ap_gen[c] != gen or vstamp[c] != vgen:
                continue
            mk = vmask[c]
            o = -1
            if (mk & (mk - 1)) == 0:
                o = 0
                while (mk >> o) != 1:
                    o += 1
            elif not neutral:
                for i in range(n_order):
                    q = order[i]
                    if (mk >> q) & 1:
                        o = q
                        break
            if o >= 0:
                bott[o] += 1
    totals[0] = n_free
    totals[1] = n_ap
    return n_comp


@dataclass(frozen=True, slots=True)
class CompositeFeatures:
    """Caractéristiques normalisées dans [0, 1] (une valeur par joueur, 0 si éliminé)."""

    territory: tuple[float, ...]
    mobility: tuple[float, ...]
    connectivity: tuple[float, ...]
    safety: tuple[float, ...]
    bottleneck: tuple[float, ...]
    free_cells: int
    articulation_points: int
    components: int

    def component(self, name: str) -> tuple[float, ...]:
        if name not in COMPONENTS:
            raise ValueError(f"composante inconnue : {name}")
        values: tuple[float, ...] = getattr(self, name)
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: list(self.component(name)) for name in COMPONENTS},
            "free_cells": self.free_cells,
            "articulation_points": self.articulation_points,
            "components": self.components,
        }


@register_heuristic("composite")
class CompositeHeuristic(Heuristic):
    """Territoire, mobilité, sorties, marge et contrôle des passages (voir le module)."""

    def __init__(
        self,
        width: int,
        height: int,
        n_players: int,
        *,
        territory: float | None = None,
        mobility: float | None = None,
        connectivity: float | None = None,
        safety: float | None = None,
        bottleneck: float | None = None,
        tie: str = "order",
        separation: bool = False,
    ) -> None:
        super().__init__(width, height, n_players)
        given = {
            name: value
            for name, value in zip(
                COMPONENTS, (territory, mobility, connectivity, safety, bottleneck), strict=True
            )
            if value is not None
        }
        self.weights = normalize_weights(given)
        self._w = tuple(self.weights[name] for name in COMPONENTS)
        self.voronoi = VoronoiHeuristic(width, height, n_players, tie=tie, separation=separation)
        self.tie = tie
        self.separation = bool(separation)
        self.neutral = tie == "neutral"
        self.floor = 1.0 / (4.0 * width * height)
        self.size = (height + 2) * (width + 2)
        size, n = self.size, n_players
        self._work: list[Any]
        self._per_player: list[Any]
        if JIT_ENABLED:
            self._work = [np.zeros(size, dtype=np.int64) for _ in range(9)]
            self._per_player = [np.zeros(n, dtype=np.int64) for _ in range(5)]
            self._totals: Any = np.zeros(2, dtype=np.int64)
        else:
            self._work = [[0] * size for _ in range(9)]
            self._per_player = [[0] * n for _ in range(5)]
            self._totals = [0, 0]
        self._gen = 0
        self.evaluations = 0
        if JIT_ENABLED and not _COMPILED[0]:  # pragma: no cover - avec Numba
            warmup()

    def set_weights(self, weights: Mapping[str, Any]) -> None:
        """Remplace les poids (validés et normalisés, absents = valeurs par défaut)."""
        self.weights = normalize_weights(weights)
        self._w = tuple(self.weights[name] for name in COMPONENTS)

    def _run(self, state: GameState) -> tuple[list[int], int, int]:
        """Exécute Voronoi puis le noyau ; renvoie (comptes Voronoi, total, composantes)."""
        counts, total = self.voronoi.counts(state)
        _dist, vmask, vstamp, vgen = self.voronoi.last_search()
        self._gen += 1
        self.evaluations += 1
        order, mob, conn, safe, bott = self._per_player
        alive_order = state.play_order()
        for i, q in enumerate(alive_order):
            order[i] = q
        disc_gen, disc, low, parent, it, stack, comp, comp_size, ap_gen = self._work
        grid: Any
        offsets: Any
        heads: Any
        alive: Any
        if JIT_ENABLED:  # pragma: no cover - avec Numba
            grid, offsets, heads, alive = state.grid, state.offsets, state.heads, state.alive
        else:
            grid, offsets, heads, alive = state._g, state._off, state._h, state._a
        n_comp = composite_kernel(
            grid,
            state.size,
            offsets,
            heads,
            alive,
            state.n,
            order,
            len(alive_order),
            self.neutral,
            vmask,
            vstamp,
            vgen,
            self._gen,
            disc_gen,
            disc,
            low,
            parent,
            it,
            stack,
            comp,
            comp_size,
            ap_gen,
            mob,
            conn,
            safe,
            bott,
            self._totals,
        )
        return counts, total, int(n_comp)

    def features(self, state: GameState) -> CompositeFeatures:
        """Les cinq caractéristiques normalisées de chaque joueur (0 pour un éliminé)."""
        counts, total, n_comp = self._run(state)
        _order, mob, conn, safe, bott = self._per_player
        n_free, n_ap = int(self._totals[0]), int(self._totals[1])
        n = state.n
        a = state._a
        terr = tuple(counts[p] / total if a[p] and total > 0 else 0.0 for p in range(n))
        mobility = tuple(int(mob[p]) / 4.0 if a[p] else 0.0 for p in range(n))
        connectivity = tuple(min(int(conn[p]), 2) / 2.0 if a[p] else 0.0 for p in range(n))
        safety = tuple(int(safe[p]) / n_free if a[p] and n_free > 0 else 0.0 for p in range(n))
        bottleneck = tuple(int(bott[p]) / n_ap if a[p] and n_ap > 0 else 0.0 for p in range(n))
        return CompositeFeatures(
            territory=terr,
            mobility=mobility,
            connectivity=connectivity,
            safety=safety,
            bottleneck=bottleneck,
            free_cells=n_free,
            articulation_points=n_ap,
            components=n_comp,
        )

    def raw_scores(self, state: GameState) -> list[float]:
        """Scores bruts raw(p) (plancher compris) ; 0 pour les joueurs éliminés."""
        f = self.features(state)
        wt, wm, wc, ws, wb = self._w
        a = state._a
        out = [0.0] * state.n
        for p in range(state.n):
            if a[p]:
                r = (
                    wt * f.territory[p]
                    + wm * f.mobility[p]
                    + wc * f.connectivity[p]
                    + ws * f.safety[p]
                    + wb * f.bottleneck[p]
                )
                out[p] = r if r > self.floor else self.floor
        return out

    def shares(self, state: GameState) -> list[float]:
        counts, total, _n_comp = self._run(state)
        _order, mob, conn, safe, bott = self._per_player
        n_free, n_ap = int(self._totals[0]), int(self._totals[1])
        wt, wm, wc, ws, wb = self._w
        floor = self.floor
        a = state._a
        n = state.n
        raw = [0.0] * n
        s = 0.0
        for p in range(n):
            if not a[p]:
                continue
            # Même ordre d'addition que raw_scores : résultats identiques au bit près.
            terr = counts[p] / total if total > 0 else 0.0
            sf = int(safe[p]) / n_free if n_free > 0 else 0.0
            bt = int(bott[p]) / n_ap if n_ap > 0 else 0.0
            r = (
                wt * terr
                + wm * (int(mob[p]) / 4.0)
                + wc * (min(int(conn[p]), 2) / 2.0)
                + ws * sf
                + wb * bt
            )
            r = max(r, floor)
            raw[p] = r
            s += r
        if s <= 0.0:
            return [0.0] * n
        return [r / s for r in raw]


_COMPILED = [False]


def warmup() -> None:
    """Compile (ou charge du cache) le noyau composite ; sans effet en Python pur."""
    if _COMPILED[0]:
        return
    _COMPILED[0] = True
    from tron.core.config import GameConfig  # noqa: PLC0415 - import local ponctuel

    cfg = GameConfig.from_dict({"grid": 5, "players": ["a", "b"]})
    state = GameState(cfg)
    for tie in TIE_MODES:
        CompositeHeuristic(5, 5, 2, tie=tie).shares(state)


__all__ = [
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "CompositeFeatures",
    "CompositeHeuristic",
    "composite_kernel",
    "normalize_weights",
    "warmup",
]
