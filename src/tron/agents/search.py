"""Moteurs de recherche à profondeur fixée.

Les trois moteurs partagent le même parcours : apply / undo sur un état unique,
joueurs vivants en rotation, feuilles (état terminal ou profondeur épuisée)
évaluées par :class:`~tron.eval.valuation.Valuation`. Un joueur sans coup légal
joue ``PASS`` (il est éliminé) et ce demi-coup compte dans la profondeur.

* ``alphabeta`` : minimax à deux camps sur le scalaire f = Σ c_j·v(j). Le camp du
  joueur racine maximise f, l'autre le minimise ; élagage alpha-bêta complet.
  Cela couvre Paranoid, l'équipe paranoïaque et deux équipes avec α = 1.
* ``maxn`` : MaxN vectoriel. Le joueur p retient le fils qui maximise W[p]·v, et
  le vecteur brut v est remonté (équivalent à la remontée du vecteur transformé
  de SOS). L'élagage superficiel compare U(y), majorant de l'utilité du parent
  sachant que le joueur courant s'assure y, à la meilleure valeur x du parent :
  Korf pour W = I, programme linéaire sur le simplexe pour W quelconque.
* ``negamax`` : negamax alpha-bêta à deux joueurs, référence de validation.

Options (toutes désactivées par défaut) : tri des coups (``static``,
``history``, ``mobility``, ``territory``), table de transposition (meilleur coup
seulement, jamais une valeur : une collision de clé ne peut changer que l'ordre
d'exploration), approfondissement itératif, fenêtres d'aspiration à la racine
(alpha-bêta, avec re-recherche : exactes), et réductions des coups tardifs
(``reduction``, alpha-bêta seulement, **inexactes**, expérimentales).

Départage dans l'arbre : à valeur égale, la direction de plus petit indice
(N < E < S < O) l'emporte. Le résultat ne dépend donc ni de l'ordre
d'exploration ni de l'élagage, qui ne coupe que des fils strictement moins bons.
À la racine, les coups ex aequo (à ``EPS`` près) sont tous renvoyés, et l'agent
applique sa règle de départage (F3.7).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from tron.core.directions import PASS
from tron.core.state import FREE, GameState
from tron.eval.valuation import Valuation

EPS = 1e-12
"""Tolérance d'égalité entre valeurs racine."""
PRUNE_MARGIN = 1e-12
"""Marge de sûreté de l'élagage superficiel (erreurs d'arrondi des bornes)."""
INF = math.inf
_CHECK_EVERY = 256
_TT_MAX = 1 << 20
ORDERINGS = ("static", "history", "mobility", "territory")
"""Tri des coups : N, E, S, O ; historique ; cases libres autour de la case visée ;
part de territoire après le coup (évaluation heuristique, nœuds de profondeur ≥ 2)."""

BoundFn = Callable[[int, int, float], float]
"""U(q, p, y) : majorant de l'utilité de q quand p s'assure au moins y."""


class SearchTimeoutError(Exception):
    """Temps de réflexion épuisé pendant une itération."""


@dataclass(slots=True)
class SearchOptions:
    """Options communes : tri des coups, approfondissement itératif, table, élagage."""

    ordering: str = "static"
    iterative: bool = False
    tt: bool = False
    pruning: bool = True
    time_limit: float | None = None
    aspiration: float | None = None
    """Demi-largeur de la fenêtre d'aspiration racine (alpha-bêta itératif), ``None`` : aucune."""
    reduction: int = 0
    """Réduction des coups tardifs en plies (alpha-bêta, expérimental, inexact), 0 : aucune."""

    def __post_init__(self) -> None:
        if self.ordering not in ORDERINGS:
            raise ValueError(f"tri des coups inconnu : {self.ordering} (attendu : {ORDERINGS})")
        if self.aspiration is not None:
            a = self.aspiration
            if isinstance(a, bool) or not isinstance(a, int | float) or not 0 < a < math.inf:
                raise ValueError(f"aspiration doit être un réel > 0 : {a!r}")
        if isinstance(self.reduction, bool) or self.reduction not in (0, 1, 2):
            raise ValueError(f"reduction doit valoir 0, 1 ou 2 : {self.reduction!r}")


@dataclass(slots=True)
class SearchResult:
    """Résultat d'une recherche à la racine."""

    best: int
    value: float
    ties: list[int]
    root_values: dict[int, float]
    pv: list[int]
    depth: int
    nodes: int = 0
    leaves: int = 0
    prunes: int = 0
    engine: str = ""
    vector: list[float] | None = None
    completed: bool = True
    exact: list[int] = field(default_factory=list)
    """Coups racine dont la valeur est exacte (les autres ne sont que bornés par l'élagage)."""
    pvs: dict[int, list[int]] = field(default_factory=dict)
    """Variante principale de chaque coup racine exact (``pv`` est celle de ``best``)."""
    order_evals: int = 0
    """Évaluations heuristiques faites pour trier les coups (tri ``territory``)."""
    researches: int = 0
    """Re-recherches racine après sortie de la fenêtre d'aspiration."""


@dataclass(slots=True)
class SearchMemory:
    """Mémoire conservée d'une décision à l'autre au cours d'une partie."""

    tt: dict[int, int] | None = None
    history: list[int] | None = None

    @classmethod
    def create(cls, options: SearchOptions, n_players: int, size: int) -> SearchMemory:
        return cls(
            tt={} if options.tt else None,
            history=[0] * (n_players * size) if options.ordering == "history" else None,
        )

    def clear(self) -> None:
        if self.tt is not None:
            self.tt.clear()
        if self.history is not None:
            self.history[:] = [0] * len(self.history)


class _Ctx:
    """Contexte d'une recherche (attributs locaux pour la vitesse)."""

    __slots__ = (
        "bound",
        "camp",
        "coeffs",
        "cut1",
        "deadline",
        "hist",
        "leaves",
        "nodes",
        "order_evals",
        "ordering",
        "prunes",
        "pv",
        "reduction",
        "reorder",
        "researches",
        "root_ply",
        "rows",
        "sign",
        "size",
        "st",
        "tt",
        "val",
    )

    def __init__(
        self, st: GameState, val: Valuation, memory: SearchMemory, deadline: float | None
    ) -> None:
        self.st = st
        self.val = val
        self.root_ply = st.ply
        self.nodes = 0
        self.leaves = 0
        self.prunes = 0
        self.deadline = deadline
        self.pv: list[list[int]] = []
        self.hist = memory.history
        self.tt = memory.tt
        self.size = st.size
        self.cut1 = False
        self.coeffs: Sequence[tuple[int, float]] = ()
        self.sign: Sequence[int] = ()
        self.camp: Sequence[int] | None = None
        self.rows: Sequence[Sequence[tuple[int, float]] | None] = ()
        self.bound: BoundFn | None = None
        self.ordering = "static"
        self.reorder = self.hist is not None or self.tt is not None
        self.reduction = 0
        self.order_evals = 0
        self.researches = 0


# outils
def _check_time(ctx: _Ctx) -> None:
    if ctx.deadline is not None and perf_counter() > ctx.deadline:
        raise SearchTimeoutError


def _ordered(ctx: _Ctx, st: GameState, p: int, moves: list[int], depth: int) -> list[int]:
    """Coup de la table de transposition, puis critère de tri, puis N, E, S, O.

    Le coup de la table n'est qu'une suggestion : il n'est retenu que s'il est
    légal ici (une collision de clé ne change que l'ordre d'exploration).
    """
    best = ctx.tt.get(st.key, -1) if ctx.tt is not None else -1
    mode = ctx.ordering
    if mode == "history" and ctx.hist is not None:
        hist = ctx.hist
        base = p * ctx.size + st._h[p]
        off = st._off
        return sorted(moves, key=lambda d: (d != best, -hist[base + off[d]], d))
    if mode == "mobility":
        g = st._g
        h = st._h[p]
        off = st._off
        free = {d: sum(1 for o in off if g[h + off[d] + o] == FREE) for d in moves}
        return sorted(moves, key=lambda d: (d != best, -free[d], d))
    if mode == "territory" and depth >= 2:
        shares: dict[int, float] = {}
        heur = ctx.val.heuristic
        for d in moves:
            st.apply(d)
            try:
                shares[d] = heur.shares(st)[p]
            finally:
                st.undo()
        ctx.order_evals += len(moves)
        return sorted(moves, key=lambda d: (d != best, -shares[d], d))
    if best < 0 or best not in moves:
        return moves
    return [best, *(m for m in moves if m != best)]


def _reward(ctx: _Ctx, st: GameState, p: int, move: int, depth: int) -> None:
    if ctx.hist is not None:
        ctx.hist[p * ctx.size + st._h[p] + st._off[move]] += depth * depth


def _store(ctx: _Ctx, key: int, move: int) -> None:
    tt = ctx.tt
    if tt is not None:
        if len(tt) >= _TT_MAX:
            tt.clear()
        tt[key] = move


def _any_alive(st: GameState, players: Sequence[int]) -> bool:
    a = st._a
    return any(a[j] for j in players)


def _dot(row: Sequence[tuple[int, float]], v: Sequence[float]) -> float:
    return sum(w * v[j] for j, w in row)


# alpha-bêta
def _ab(ctx: _Ctx, depth: int, ply: int, alpha: float, beta: float) -> float:
    ctx.nodes += 1
    if ctx.nodes % _CHECK_EVERY == 0:
        _check_time(ctx)
    st = ctx.st
    if depth == 0 or st.over or (ctx.camp is not None and not _any_alive(st, ctx.camp)):
        ctx.leaves += 1
        ctx.pv[ply] = []
        return ctx.val.scalar(st, ctx.root_ply, ctx.coeffs)
    p = st.current
    moves = st.legal_moves()
    if not moves:
        st.apply(PASS)
        value = _ab(ctx, depth - 1, ply + 1, alpha, beta)
        st.undo()
        ctx.pv[ply] = [PASS, *ctx.pv[ply + 1]]
        return value
    if len(moves) > 1 and ctx.reorder:
        moves = _ordered(ctx, st, p, moves, depth)
    best_m = moves[0]
    red = ctx.reduction if depth >= 3 else 0
    if ctx.sign[p] > 0:
        best = -INF
        for i, m in enumerate(moves):
            st.apply(m)
            if red and i >= 2:  # coup tardif : recherche réduite, puis complète s'il améliore
                v = _ab(ctx, depth - 1 - red, ply + 1, alpha, beta)
                if v > alpha:
                    v = _ab(ctx, depth - 1, ply + 1, alpha, beta)
            else:
                v = _ab(ctx, depth - 1, ply + 1, alpha, beta)
            st.undo()
            if v > best or (v == best and m < best_m):
                best = v
                best_m = m
                ctx.pv[ply] = [m, *ctx.pv[ply + 1]]
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        ctx.prunes += 1
                        _reward(ctx, st, p, m, depth)
                        break
    else:
        best = INF
        for i, m in enumerate(moves):
            st.apply(m)
            if red and i >= 2:
                v = _ab(ctx, depth - 1 - red, ply + 1, alpha, beta)
                if v < beta:
                    v = _ab(ctx, depth - 1, ply + 1, alpha, beta)
            else:
                v = _ab(ctx, depth - 1, ply + 1, alpha, beta)
            st.undo()
            if v < best or (v == best and m < best_m):
                best = v
                best_m = m
                ctx.pv[ply] = [m, *ctx.pv[ply + 1]]
                if v < beta:
                    beta = v
                    if alpha >= beta:
                        ctx.prunes += 1
                        _reward(ctx, st, p, m, depth)
                        break
    _store(ctx, st.key, best_m)
    return best


def _ab_full(ctx: _Ctx, depth: int, ply: int) -> float:
    """Minimax à deux camps sans élagage (référence des tests)."""
    ctx.nodes += 1
    st = ctx.st
    if depth == 0 or st.over or (ctx.camp is not None and not _any_alive(st, ctx.camp)):
        ctx.leaves += 1
        ctx.pv[ply] = []
        return ctx.val.scalar(st, ctx.root_ply, ctx.coeffs)
    moves = st.legal_moves()
    if not moves:
        st.apply(PASS)
        value = _ab_full(ctx, depth - 1, ply + 1)
        st.undo()
        ctx.pv[ply] = [PASS, *ctx.pv[ply + 1]]
        return value
    maximize = ctx.sign[st.current] > 0
    best = -INF if maximize else INF
    for m in moves:
        st.apply(m)
        v = _ab_full(ctx, depth - 1, ply + 1)
        st.undo()
        if (v > best) if maximize else (v < best):
            best = v
            ctx.pv[ply] = [m, *ctx.pv[ply + 1]]
    return best


def _root_ab(
    ctx: _Ctx,
    depth: int,
    moves: list[int],
    pruning: bool,
    window: tuple[float, float] | None = None,
) -> tuple[dict[int, float], dict[int, list[int]]]:
    """Coups racine ; un coup n'est exact que si sa valeur est strictement dans sa fenêtre.

    Avec ``window = (lo, hi)`` (aspiration) : le premier coup est cherché dans
    (lo, hi) et les suivants dans (meilleur − EPS, hi) ; une sortie par le haut
    (v ≥ hi) ou, pour le premier coup, par le bas (v ≤ lo) est re-cherchée avec
    une fenêtre ouverte de ce côté. Les valeurs exactes, les coups ex aequo et le
    meilleur coup sont donc ceux de la recherche sans fenêtre.
    """
    st = ctx.st
    lo, hi = window if window is not None else (-INF, INF)
    best = -INF
    values: dict[int, float] = {}
    pvs: dict[int, list[int]] = {}
    for m in moves:
        alpha = best - EPS if best > -INF else lo
        st.apply(m)
        if not pruning:
            v = _ab_full(ctx, depth - 1, 1)
            a_used, b_used = -INF, INF
        else:
            a_used, b_used = alpha, hi
            v = _ab(ctx, depth - 1, 1, a_used, b_used)
            if v >= b_used < INF:  # sortie par le haut : borne inférieure seulement
                ctx.researches += 1
                b_used = INF
                v = _ab(ctx, depth - 1, 1, a_used, b_used)
            if best == -INF and v <= a_used and a_used > -INF:  # premier coup sous la fenêtre
                ctx.researches += 1
                a_used = -INF
                v = _ab(ctx, depth - 1, 1, a_used, b_used)
        st.undo()
        values[m] = v
        if a_used < v < b_used or (a_used == -INF and b_used == INF):
            pvs[m] = [m, *ctx.pv[1]]
        best = max(best, v)
    return values, pvs


# MaxN
def _mx(ctx: _Ctx, depth: int, ply: int, q: int, x: float) -> list[float]:
    ctx.nodes += 1
    if ctx.nodes % _CHECK_EVERY == 0:
        _check_time(ctx)
    st = ctx.st
    if depth == 0 or st.over:
        ctx.leaves += 1
        ctx.pv[ply] = []
        return ctx.val.vector(st, ctx.root_ply)
    p = st.current
    moves = st.legal_moves()
    if not moves:
        st.apply(PASS)
        vec = _mx(ctx, depth - 1, ply + 1, q, x)
        st.undo()
        ctx.pv[ply] = [PASS, *ctx.pv[ply + 1]]
        return vec
    if len(moves) > 1 and ctx.reorder:
        moves = _ordered(ctx, st, p, moves, depth)
    row = ctx.rows[p]
    bound = ctx.bound if (q >= 0 and q != p and x > -INF) else None
    best_v: list[float] = []
    best_u = -INF
    best_m = 4
    for m in moves:
        st.apply(m)
        v = _mx(ctx, depth - 1, ply + 1, p, best_u)
        st.undo()
        u = v[p] if row is None else _dot(row, v)
        if u > best_u or (u == best_u and m < best_m):
            best_u = u
            best_v = v
            best_m = m
            ctx.pv[ply] = [m, *ctx.pv[ply + 1]]
        if bound is not None and bound(q, p, best_u) + PRUNE_MARGIN < x:
            ctx.prunes += 1
            if ply == 1:
                ctx.cut1 = True
            _reward(ctx, st, p, best_m, depth)
            break
    _store(ctx, st.key, best_m)
    return best_v


def _root_mx(
    ctx: _Ctx, depth: int, moves: list[int]
) -> tuple[dict[int, float], dict[int, list[int]], dict[int, list[float]]]:
    st = ctx.st
    p = st.current
    row = ctx.rows[p]
    best_u = -INF
    values: dict[int, float] = {}
    pvs: dict[int, list[int]] = {}
    vectors: dict[int, list[float]] = {}
    for m in moves:
        ctx.cut1 = False
        st.apply(m)
        v = _mx(ctx, depth - 1, 1, p, best_u)
        st.undo()
        u = v[p] if row is None else _dot(row, v)
        values[m] = u
        if not ctx.cut1:
            pvs[m] = [m, *ctx.pv[1]]
            vectors[m] = v
        best_u = max(best_u, u)
    return values, pvs, vectors


# negamax
def _nm(ctx: _Ctx, depth: int, ply: int, alpha: float, beta: float) -> float:
    ctx.nodes += 1
    if ctx.nodes % _CHECK_EVERY == 0:
        _check_time(ctx)
    st = ctx.st
    if depth == 0 or st.over:
        ctx.leaves += 1
        ctx.pv[ply] = []
        vec = ctx.val.vector(st, ctx.root_ply)
        p = st.current
        return vec[p] - vec[1 - p]
    moves = st.legal_moves()
    if not moves:
        st.apply(PASS)
        value = -_nm(ctx, depth - 1, ply + 1, -beta, -alpha)
        st.undo()
        ctx.pv[ply] = [PASS, *ctx.pv[ply + 1]]
        return value
    best = -INF
    best_m = 4
    for m in moves:
        st.apply(m)
        v = -_nm(ctx, depth - 1, ply + 1, -beta, -alpha)
        st.undo()
        if v > best or (v == best and m < best_m):
            best = v
            best_m = m
            ctx.pv[ply] = [m, *ctx.pv[ply + 1]]
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    ctx.prunes += 1
                    break
    return best


def _root_nm(
    ctx: _Ctx, depth: int, moves: list[int]
) -> tuple[dict[int, float], dict[int, list[int]]]:
    st = ctx.st
    best = -INF
    values: dict[int, float] = {}
    pvs: dict[int, list[int]] = {}
    for m in moves:
        alpha = best - EPS if best > -INF else -INF
        st.apply(m)
        v = -_nm(ctx, depth - 1, 1, -INF, -alpha)
        st.undo()
        values[m] = v
        if v > alpha:
            pvs[m] = [m, *ctx.pv[1]]
        best = max(best, v)
    return values, pvs


# pilote
def _finish(
    ctx: _Ctx,
    engine: str,
    depth: int,
    values: dict[int, float],
    pvs: dict[int, list[int]],
    vectors: dict[int, list[float]] | None,
) -> SearchResult:
    best_value = max(values[m] for m in pvs)
    ties = sorted(m for m in pvs if values[m] > best_value - EPS)
    best = ties[0]
    pv = pvs[best]
    vector = vectors.get(best) if vectors is not None else None
    if vector is None:
        vector = _pv_vector(ctx, pv)
    return SearchResult(
        best=best,
        value=values[best],
        ties=ties,
        root_values=dict(values),
        pv=pv,
        depth=depth,
        engine=engine,
        vector=vector,
        exact=sorted(pvs),
        pvs={m: list(line) for m, line in pvs.items()},
    )


def _pv_vector(ctx: _Ctx, pv: list[int]) -> list[float]:
    st = ctx.st
    applied = 0
    for m in pv:
        if st.is_terminal():
            break
        st.apply(m)
        applied += 1
    vec = ctx.val.vector(st, ctx.root_ply)
    for _ in range(applied):
        st.undo()
    return vec


def search(
    engine: str,
    state: GameState,
    valuation: Valuation,
    depth: int,
    *,
    options: SearchOptions | None = None,
    memory: SearchMemory | None = None,
    coeffs: Sequence[tuple[int, float]] = (),
    signs: Sequence[int] = (),
    camp: Sequence[int] | None = None,
    rows: Sequence[Sequence[tuple[int, float]] | None] | None = None,
    bound: BoundFn | None = None,
) -> SearchResult:
    """Recherche depuis ``state`` (modifié puis restauré) pour le joueur au trait.

    ``engine`` vaut ``"alphabeta"`` (paramètres ``coeffs``, ``signs``, ``camp``),
    ``"maxn"`` (``rows`` : ligne creuse de W par joueur, ``None`` pour l'identité ;
    ``bound`` : fonction d'élagage) ou ``"negamax"`` (deux joueurs).
    """
    if depth < 1:
        raise ValueError("la profondeur doit être >= 1")
    opts = options or SearchOptions()
    mem = memory or SearchMemory.create(opts, state.n, state.size)
    moves = state.legal_moves()
    if not moves:
        return SearchResult(
            best=PASS,
            value=math.nan,
            ties=[PASS],
            root_values={},
            pv=[PASS],
            depth=0,
            engine=engine,
        )
    if engine == "negamax" and state.n != 2:
        raise ValueError("negamax ne s'applique qu'à deux joueurs")
    if engine == "alphabeta" and (not coeffs or signs[state.current] <= 0):
        raise ValueError("alpha-bêta : le joueur racine doit appartenir au camp maximisant")
    t_start = perf_counter()
    deadline = t_start + opts.time_limit if opts.time_limit else None
    ctx = _Ctx(state, valuation, mem, deadline)
    ctx.coeffs = coeffs
    ctx.sign = signs
    ctx.camp = camp
    ctx.rows = rows if rows is not None else [None] * state.n
    ctx.bound = bound if opts.pruning else None
    ctx.ordering = opts.ordering
    ctx.reorder = ctx.hist is not None or ctx.tt is not None or opts.ordering != "static"
    ctx.reduction = opts.reduction if engine == "alphabeta" and opts.pruning else 0
    root_ply = state.ply
    try:
        return _drive(
            engine,
            state,
            valuation,
            depth,
            opts,
            mem,
            ctx,
            moves,
            root_ply,
            coeffs,
            signs,
            camp,
            rows,
            bound,
        )
    except BaseException:  # toute erreur : l'état reçu est rendu intact
        while state.ply > root_ply:
            state.undo()
        raise


def _drive(
    engine: str,
    state: GameState,
    valuation: Valuation,
    depth: int,
    opts: SearchOptions,
    mem: SearchMemory,
    ctx: _Ctx,
    moves: list[int],
    root_ply: int,
    coeffs: Sequence[tuple[int, float]],
    signs: Sequence[int],
    camp: Sequence[int] | None,
    rows: Sequence[Sequence[tuple[int, float]] | None] | None,
    bound: BoundFn | None,
) -> SearchResult:
    """Itérations de la recherche (une seule sans approfondissement itératif)."""
    order = list(moves)
    result: SearchResult | None = None
    depths = range(1, depth + 1) if opts.iterative else range(depth, depth + 1)
    for d in depths:
        ctx.pv = [[] for _ in range(d + 2)]
        try:
            if engine == "alphabeta":
                window = None
                if opts.aspiration is not None and result is not None and opts.pruning:
                    window = (result.value - opts.aspiration, result.value + opts.aspiration)
                values, pvs = _root_ab(ctx, d, order, opts.pruning, window)
                vectors = None
            elif engine == "maxn":
                values, pvs, vectors = _root_mx(ctx, d, order)
            elif engine == "negamax":
                values, pvs = _root_nm(ctx, d, order)
                vectors = None
            else:
                raise ValueError(f"moteur inconnu : {engine}")
        except SearchTimeoutError:
            while state.ply > root_ply:
                state.undo()
            if result is None:  # garantir au moins une itération complète
                return search(
                    engine,
                    state,
                    valuation,
                    1,
                    options=SearchOptions(
                        opts.ordering, False, opts.tt, opts.pruning, None, None, opts.reduction
                    ),
                    memory=mem,
                    coeffs=coeffs,
                    signs=signs,
                    camp=camp,
                    rows=rows,
                    bound=bound,
                )
            result.completed = False
            break
        result = _finish(ctx, engine, d, values, pvs, vectors)
        if d < depth:
            order = sorted(moves, key=lambda m: (-values[m], m))
    assert result is not None
    result.nodes = ctx.nodes
    result.leaves = ctx.leaves
    result.prunes = ctx.prunes
    result.order_evals = ctx.order_evals
    result.researches = ctx.researches
    if result.ties and result.ties[0] == PASS:  # pragma: no cover - garde-fou
        result.best = PASS
    return result


__all__ = [
    "EPS",
    "ORDERINGS",
    "BoundFn",
    "SearchMemory",
    "SearchOptions",
    "SearchResult",
    "SearchTimeoutError",
    "search",
]
