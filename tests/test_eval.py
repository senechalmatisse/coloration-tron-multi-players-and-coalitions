"""Tests de l'heuristique de Voronoi et de la valeur des feuilles."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import pytest

from conftest import make_config
from tron.core import FREE, PASS, GameState
from tron.eval import (
    UniformHeuristic,
    Valuation,
    VoronoiHeuristic,
    make_heuristic,
    warmup,
)


# référence naïve
def _bfs(st: GameState, start: int) -> dict[int, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for off in st._off:
            v = u + off
            if st._g[v] == FREE and v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def naive_counts(st: GameState, neutral: bool) -> list[int]:
    order = st.play_order()
    rank = {p: i for i, p in enumerate(order)}
    dists = {p: _bfs(st, st._h[p]) for p in order}
    counts = [0] * st.n
    cells = set().union(*(d.keys() for d in dists.values()))
    for c in cells:
        ds = [(dists[p][c], rank[p], p) for p in order if c in dists[p]]
        dmin = min(d for d, _, _ in ds)
        winners = sorted((r, p) for d, r, p in ds if d == dmin)
        if len(winners) > 1 and neutral:
            continue
        counts[winners[0][1]] += 1
    return counts


def naive_separation(st: GameState, counts: list[int]) -> list[int]:
    out = list(counts)
    reach = {p: set(_bfs(st, st._h[p])) for p in st.alive_players()}
    for p in st.alive_players():
        if any(reach[p] & reach[q] for q in reach if q != p):
            continue
        # Joueurs voisins par une arête entre régions ?
        touching = False
        for q, cells in reach.items():
            if q == p:
                continue
            for c in reach[p]:
                if any(c + off in cells for off in st._off):
                    touching = True
        if touching:
            continue
        head = st._h[p]

        def color(c: int) -> int:
            r, col = divmod(c, st.stride)
            return (r + col) & 1

        same = sum(1 for c in reach[p] if c != head and color(c) == color(head))
        other = sum(1 for c in reach[p] if color(c) != color(head))
        fill = 2 * same + 1 if other > same else 2 * other
        out[p] = fill + 1
    return out


def random_state(
    rng: random.Random, w: int, h: int, n: int, plies: int, teams: list[str] | None = None
) -> GameState:
    cfg = make_config(
        w,
        h,
        n,
        teams=teams,
        seed=rng.randrange(1000),
        placement={"mode": "random", "min_distance": 2},
    )
    st = GameState(cfg)
    for _ in range(plies):
        if st.is_terminal():
            break
        moves = st.legal_moves()
        st.apply(rng.choice(moves) if moves else PASS)
    return st


# tests
def test_corridor_tie_modes() -> None:
    cfg = make_config(5, 5, 2, placement={"positions": [[0, 2], [4, 2]]})
    st = GameState(cfg)
    order = VoronoiHeuristic(5, 5, 2, tie="order")
    neutral = VoronoiHeuristic(5, 5, 2, tie="neutral")
    assert order.counts(st) == ([15, 10], 25)
    assert neutral.counts(st) == ([10, 10], 20)
    assert neutral.shares(st) == [0.5, 0.5]
    st.current = 1
    assert order.counts(st) == ([10, 15], 25)
    owners = order.owner_map(st)
    assert owners.shape == (5, 5)
    assert (owners[:, 2] == 1).all() and (owners[:, 0] == 0).all()
    assert (neutral.owner_map(st)[:, 2] == -1).all()


@pytest.mark.parametrize("neutral", [False, True])
def test_kernel_matches_naive_reference(neutral: bool) -> None:
    rng = random.Random(7)
    for trial in range(60):
        n = rng.randint(2, 6)
        st = random_state(rng, rng.randint(5, 12), rng.randint(5, 12), n, rng.randint(0, 60))
        if st.is_terminal():
            continue
        heur = VoronoiHeuristic(st.width, st.height, n, tie="neutral" if neutral else "order")
        counts, total = heur.counts(st)
        assert counts == naive_counts(st, neutral), (trial, st.to_text())
        assert total == sum(counts)
        shares = heur.shares(st)
        assert sum(shares) == pytest.approx(1.0)
        for p in range(n):
            if st.is_alive(p):
                assert shares[p] > 0
            else:
                assert shares[p] == 0


def test_separation_matches_naive_reference() -> None:
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        n = rng.randint(2, 4)
        st = random_state(rng, rng.randint(5, 9), rng.randint(5, 9), n, rng.randint(10, 70))
        if st.is_terminal():
            continue
        base = VoronoiHeuristic(st.width, st.height, n).counts(st)[0]
        sep = VoronoiHeuristic(st.width, st.height, n, separation=True)
        counts, total = sep.counts(st)
        expected = naive_separation(st, base)
        assert counts == expected, st.to_text()
        assert total == sum(counts)
        checked += expected != base
    assert checked >= 2  # des cas séparés ont bien été rencontrés


def test_separated_corridor_parity() -> None:
    cfg = make_config(5, 5, 2)
    text = """
        A....
        aaaaa
        bbbbb
        ...B.
        .....
    """
    st = GameState.from_text(cfg, text)
    heur = VoronoiHeuristic(5, 5, 2, separation=True)
    counts, _ = heur.counts(st)
    # J0 : couloir de 4 cases -> 4 coups ; J1 : bloc 2x5 moins la tête -> borne de parité.
    assert counts[0] == 5
    plain = VoronoiHeuristic(5, 5, 2).counts(st)[0]
    assert plain == [5, 10]
    assert counts[1] <= plain[1]


def test_dead_players_have_no_territory() -> None:
    cfg = make_config(8, 8, 3)
    st = GameState(cfg)
    st.apply(PASS)
    shares = VoronoiHeuristic(8, 8, 3).shares(st)
    assert shares[0] == 0 and sum(shares) == pytest.approx(1.0)
    assert UniformHeuristic(8, 8, 3).shares(st) == [0.0, 0.5, 0.5]


def test_make_heuristic_and_errors() -> None:
    h = make_heuristic("voronoi", 10, 10, 2, tie="neutral")
    assert isinstance(h, VoronoiHeuristic) and h.neutral
    with pytest.raises(ValueError):
        make_heuristic("inconnue", 10, 10, 2)
    with pytest.raises(ValueError):
        VoronoiHeuristic(10, 10, 2, tie="bizarre")
    warmup()


# valuation
def test_valuation_terminal_dominance_and_bounds() -> None:
    rng = random.Random(3)
    cfg = make_config(7, 7, 4, teams=["A", "B", "A", "C"])
    val = Valuation(cfg, VoronoiHeuristic(7, 7, 4))
    assert val.win_value(0) > val.win_value(5) > 1.0
    assert val.dead_value(5) > val.dead_value(0) >= 0.0
    assert val.dead_value(1000) < 1.0 / (7 * 7)
    for _ in range(200):
        st = random_state(rng, 7, 7, 4, rng.randint(0, 50), teams=["A", "B", "A", "C"])
        root = max(0, st.ply - rng.randint(0, 5))
        v = val.vector(st, root)
        assert min(v) >= 0.0
        assert sum(v) <= val.sum_bound + 1e-12
        for p in range(4):
            for q in range(4):
                if p != q:
                    assert v[p] + v[q] <= val.pair_bound(p, q) + 1e-12
        if st.is_terminal():
            winners = [p for p in range(4) if val.team[p] == st.winning_team()]
            assert all(v[p] > 1.0 for p in winners)
        else:
            alive = [v[p] for p in range(4) if st.is_alive(p)]
            dead = [v[p] for p in range(4) if not st.is_alive(p)]
            if dead:
                assert max(dead) < min(alive)
        coeffs = [(0, 1.0), (2, 1.0), (1, -1.0)]
        assert val.scalar(st, root, coeffs) == pytest.approx(v[0] + v[2] - v[1])


def test_scalar_skips_heuristic_when_camp_is_dead() -> None:
    cfg = make_config(8, 8, 3)
    heur = VoronoiHeuristic(8, 8, 3)
    val = Valuation(cfg, heur)
    st = GameState(cfg)
    st.apply(PASS)  # J0 éliminé au demi-coup 0
    before = heur.evaluations
    assert val.scalar(st, 0, [(0, 1.0)]) == 0.0
    assert heur.evaluations == before


def test_owner_map_consistent_with_counts() -> None:
    rng = random.Random(5)
    st = random_state(rng, 10, 8, 3, 25)
    heur = VoronoiHeuristic(10, 8, 3)
    counts, _ = heur.counts(st)
    owners = heur.owner_map(st)
    for p in range(3):
        assert int(np.sum(owners == p)) == counts[p]
