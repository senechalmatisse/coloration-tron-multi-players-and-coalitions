"""Tests de l'heuristique territoriale multicritère (P01)."""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

from conftest import make_config
from tron.agents import MaxNAgent, search
from tron.agents.search import SearchOptions
from tron.core import FREE, PASS, GameConfig, GameState
from tron.eval import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    CompositeHeuristic,
    Valuation,
    VoronoiHeuristic,
    make_heuristic,
    normalize_weights,
)
from tron.sim import Match

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ------------------------------------------------------------- référence naïve
def _components(st: GameState, cells: set[int]) -> tuple[dict[int, int], int]:
    label: dict[int, int] = {}
    k = 0
    for c in sorted(cells):
        if c in label:
            continue
        label[c] = k
        queue = deque([c])
        while queue:
            u = queue.popleft()
            for off in st._off:
                v = u + off
                if v in cells and v not in label:
                    label[v] = k
                    queue.append(v)
        k += 1
    return label, k


def naive_counts(st: GameState, tie: str) -> dict[str, object]:
    """Comptes bruts recalculés par force brute (points d'articulation par suppression)."""
    free = {c for c in range(st.size) if st._g[c] == FREE}
    label, n_comp = _components(st, free)
    sizes = [0] * n_comp
    for k in label.values():
        sizes[k] += 1
    aps = set()
    for c in free:
        rest = {x for x, k in label.items() if k == label[c]} - {c}
        if rest and _components(st, rest)[1] > 1:
            aps.add(c)
    owners = VoronoiHeuristic(st.width, st.height, st.n, tie=tie).owner_map(st)
    bott = [0] * st.n
    for c in aps:
        x, y = st.coords(c)
        if owners[y, x] >= 0:
            bott[int(owners[y, x])] += 1
    mob, conn, safe = [0] * st.n, [0] * st.n, [0] * st.n
    for p in st.alive_players():
        h = st._h[p]
        nbrs = [h + off for off in st._off if st._g[h + off] == FREE]
        comps = {label[v] for v in nbrs}
        mob[p], conn[p] = len(nbrs), len(comps)
        safe[p] = max((sizes[k] for k in comps), default=0)
    return {
        "mob": mob,
        "conn": conn,
        "safe": safe,
        "bott": bott,
        "free": len(free),
        "aps": len(aps),
        "comps": n_comp,
    }


def random_state(
    rng: random.Random,
    w: int,
    h: int,
    n: int,
    plies: int,
    teams: list[str] | None = None,
    heuristic: str = "voronoi",
) -> GameState:
    cfg = make_config(
        w,
        h,
        n,
        teams=teams,
        seed=rng.randrange(1000),
        placement={"mode": "random", "min_distance": 2},
        heuristic=heuristic,
    )
    st = GameState(cfg)
    for _ in range(plies):
        if st.is_terminal():
            break
        moves = st.legal_moves()
        st.apply(rng.choice(moves) if moves else PASS)
    return st


# ------------------------------------------------------------------- noyau
@pytest.mark.parametrize("tie", ["order", "neutral"])
def test_kernel_matches_brute_force(tie: str) -> None:
    rng = random.Random(17 if tie == "order" else 18)
    checked = 0
    for _ in range(120):
        n = rng.randint(2, 6)
        st = random_state(rng, rng.randint(5, 9), rng.randint(5, 9), n, rng.randint(0, 40))
        if st.is_terminal():
            continue
        heur = CompositeHeuristic(st.width, st.height, n, tie=tie)
        f = heur.features(st)
        ref = naive_counts(st, tie)
        assert (f.free_cells, f.articulation_points, f.components) == (
            ref["free"],
            ref["aps"],
            ref["comps"],
        ), st.to_text()
        mob, conn, safe, bott = ref["mob"], ref["conn"], ref["safe"], ref["bott"]
        for p in range(n):
            assert isinstance(mob, list) and isinstance(conn, list)
            assert isinstance(safe, list) and isinstance(bott, list)
            if not st.is_alive(p):
                assert f.mobility[p] == f.safety[p] == f.bottleneck[p] == 0.0
                continue
            assert f.mobility[p] == mob[p] / 4
            assert f.connectivity[p] == min(conn[p], 2) / 2
            assert f.safety[p] == pytest.approx(safe[p] / max(1, f.free_cells))
            if f.articulation_points:
                assert f.bottleneck[p] == pytest.approx(bott[p] / f.articulation_points)
        checked += 1
    assert checked >= 60


def test_shares_contract_on_random_positions() -> None:
    rng = random.Random(5)
    for _ in range(150):
        n = rng.randint(2, 8)
        st = random_state(rng, rng.randint(5, 12), rng.randint(5, 12), n, rng.randint(0, 90))
        if st.is_terminal():
            continue
        heur = CompositeHeuristic(st.width, st.height, n)
        shares = heur.shares(st)
        assert len(shares) == n
        assert all(math.isfinite(s) and s >= 0.0 for s in shares)
        alive = [shares[p] for p in range(n) if st.is_alive(p)]
        assert all(s > 0.0 for s in alive)
        assert all(shares[p] == 0.0 for p in range(n) if not st.is_alive(p))
        assert sum(alive) == pytest.approx(1.0, abs=1e-9)
        raw = heur.raw_scores(st)
        total = sum(raw)  # sommation compensée de Python : égalité à l'arrondi près
        assert [r / total for r in raw] == pytest.approx(shares, abs=1e-15)
        feats = heur.features(st)
        for name in COMPONENTS:
            assert all(0.0 <= x <= 1.0 for x in feats.component(name))
        assert feats.to_dict()["free_cells"] == feats.free_cells


def test_territory_component_is_the_voronoi_share_and_voronoi_is_unchanged() -> None:
    rng = random.Random(8)
    for _ in range(40):
        st = random_state(rng, 9, 9, 3, rng.randint(0, 40))
        if st.is_terminal():
            continue
        vor = VoronoiHeuristic(9, 9, 3)
        before = vor.counts(st)
        comp = CompositeHeuristic(9, 9, 3)
        feats = comp.features(st)
        assert list(feats.territory) == pytest.approx(vor.shares(st))
        assert vor.counts(st) == before  # instances indépendantes : aucun effet de bord
        only_territory = CompositeHeuristic(
            9, 9, 3, territory=1, mobility=0, connectivity=0, safety=0, bottleneck=0
        )
        assert only_territory.shares(st) == pytest.approx(vor.shares(st), abs=1e-12)


# ---------------------------------------------------------------- poids
@pytest.mark.parametrize(
    "weights",
    [
        {"territory": -0.1},
        {"mobility": math.nan},
        {"safety": math.inf},
        {"territory": 0, "mobility": 0, "connectivity": 0, "safety": 0, "bottleneck": 0},
        {"territory": "beaucoup"},
        {"territory": True},
        {"vitesse": 0.3},
    ],
)
def test_invalid_weights_are_rejected(weights: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        normalize_weights(weights)
    with pytest.raises(ValueError):
        make_heuristic("composite", 8, 8, 2, **weights)


def test_weights_are_normalized_and_defaults_apply() -> None:
    assert sum(normalize_weights({}).values()) == pytest.approx(1.0)
    assert normalize_weights({}) == pytest.approx(DEFAULT_WEIGHTS)
    scaled = normalize_weights({k: 10 * v for k, v in DEFAULT_WEIGHTS.items()})
    assert scaled == pytest.approx(DEFAULT_WEIGHTS)
    partial = normalize_weights({"bottleneck": 0})
    assert partial["bottleneck"] == 0.0 and sum(partial.values()) == pytest.approx(1.0)
    rng = random.Random(1)
    st = random_state(rng, 8, 8, 3, 12)
    a = CompositeHeuristic(8, 8, 3).shares(st)
    b = make_heuristic(
        "composite", 8, 8, 3, **{k: 4 * v for k, v in DEFAULT_WEIGHTS.items()}
    ).shares(st)
    assert a == pytest.approx(b, abs=1e-12)
    with pytest.raises(ValueError):
        make_heuristic("composite", 8, 8, 2, tie="bizarre")
    with pytest.raises(ValueError):
        make_heuristic("composite", 8, 8, 2, inconnu=1)


# ------------------------------------------------------------- symétries
def _mirror_text(rows: list[str]) -> str:
    return "\n".join(rows)


def test_mirrored_position_gives_symmetric_shares() -> None:
    rng = random.Random(3)
    for _ in range(25):
        w, h = 9, 7
        left = [["." for _ in range(w)] for _ in range(h)]
        for _ in range(rng.randint(0, 12)):
            x, y = rng.randrange(0, w // 2), rng.randrange(h)
            left[y][x] = "a"
        hx, hy = rng.randrange(0, w // 2), rng.randrange(h)
        left[hy][hx] = "A"
        rows = []
        for y in range(h):
            row = left[y][:]
            for x in range(w // 2):
                ch = left[y][x]
                row[w - 1 - x] = {"a": "b", "A": "B"}.get(ch, ch)
            row[w // 2] = "."
            rows.append("".join(row))
        cfg = make_config(w, h, 2)
        st = GameState.from_text(cfg, _mirror_text(rows), current=0)
        heur = CompositeHeuristic(w, h, 2, tie="neutral")
        s = heur.shares(st)
        assert s[0] == pytest.approx(s[1], abs=1e-12), "\n".join(rows)


def test_rotation_symmetric_four_players() -> None:
    size = 9
    cfg = make_config(size, size, 4)
    grid = [["." for _ in range(size)] for _ in range(size)]

    def rot(x: int, y: int, k: int) -> tuple[int, int]:
        for _ in range(k):
            x, y = size - 1 - y, x
        return x, y

    walls = [(1, 1), (2, 1), (1, 3), (3, 2)]
    head = (2, 2)
    for k in range(4):
        letter = "abcd"[k]
        for wx, wy in walls:
            x, y = rot(wx, wy, k)
            grid[y][x] = letter
        x, y = rot(*head, k)
        grid[y][x] = letter.upper()
    st = GameState.from_text(cfg, "\n".join("".join(r) for r in grid), current=0)
    s = CompositeHeuristic(size, size, 4, tie="neutral").shares(st)
    assert max(s) - min(s) < 1e-12


# ----------------------------------------------------- Valuation et recherche
def test_floor_keeps_alive_above_dead_values() -> None:
    cfg = make_config(5, 5, 3, heuristic={"name": "composite", "territory": 0})
    st = GameState.from_text(
        cfg,
        """
        Aa...
        aa...
        ...B.
        .....
        C....
        """,
    )
    heur = CompositeHeuristic(5, 5, 3, territory=0)
    s = heur.shares(st)
    assert s[0] > 0.0  # J0 n'a plus aucun coup mais vit encore
    val = Valuation(cfg, heur)
    st.apply(PASS)  # J0 éliminé
    v = val.vector(st, 0)
    assert v[0] < min(v[1], v[2])


def test_valuation_bounds_hold_with_composite() -> None:
    rng = random.Random(9)
    teams = ["A", "B", "A", "C"]
    for _ in range(120):
        st = random_state(rng, 7, 7, 4, rng.randint(0, 45), teams=teams, heuristic="composite")
        val = Valuation(st.config, CompositeHeuristic(7, 7, 4))
        root = max(0, st.ply - rng.randint(0, 5))
        v = val.vector(st, root)
        assert min(v) >= 0.0 and sum(v) <= val.sum_bound + 1e-12
        for p in range(4):
            for q in range(4):
                if p != q:
                    assert v[p] + v[q] <= val.pair_bound(p, q) + 1e-12
        if not st.is_terminal():
            alive = [v[p] for p in range(4) if st.is_alive(p)]
            dead = [v[p] for p in range(4) if not st.is_alive(p)]
            if dead:
                assert max(dead) < min(alive)


def test_pruning_stays_exact_with_composite() -> None:
    rng = random.Random(31)
    tested = 0
    for _ in range(25):
        n = rng.choice([2, 3])
        cfg = make_config(
            7, 7, n, seed=rng.randrange(1000), placement={"mode": "random"}, heuristic="composite"
        )
        st = GameState(cfg)
        for _ in range(rng.randint(n, 5 * n)):
            moves = st.legal_moves()
            if st.is_terminal() or not moves:
                break
            st.apply(rng.choice(moves))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        p = st.current
        depth = rng.randint(1, 3)
        val = Valuation(cfg, CompositeHeuristic(7, 7, n))
        kw = {"coeffs": ((p, 1.0),), "signs": tuple(1 if q == p else -1 for q in range(n))}
        full = search("alphabeta", st, val, depth, options=SearchOptions(pruning=False), **kw)
        fast = search("alphabeta", st, val, depth, **kw)
        assert fast.value == pytest.approx(full.value, abs=1e-12) and fast.ties == full.ties
        a = MaxNAgent(p, cfg, depth=depth).run_search(st, depth)
        b = MaxNAgent(p, cfg, depth=depth, pruning=False).run_search(st, depth)
        assert (a.value, a.ties, a.vector) == (b.value, b.ties, b.vector)
        tested += 1
    assert tested >= 12


# ------------------------------------------------------------ intégration
def test_yaml_config_and_match() -> None:
    cfg = GameConfig.from_yaml(CONFIGS / "games" / "composite_20x20.yaml")
    assert cfg.heuristic.name == "composite"
    assert cfg.heuristic.params["territory"] == 0.55
    small = GameConfig.from_dict(
        {
            "grid": 8,
            "seed": 2,
            "heuristic": {"name": "composite", "territory": 0.6, "safety": 0.4},
            "players": [{"agent": "greedy"}, {"agent": "paranoid", "depth": 2}],
        }
    )
    a = Match(small).run()
    b = Match(small).run()
    assert (a.winning_team, a.ranks, a.n_moves) == (b.winning_team, b.ranks, b.n_moves)


def test_no_gui_dependency() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    code = "import sys, tron.eval.composite; print('arcade' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert out.stdout.strip() == "False"
