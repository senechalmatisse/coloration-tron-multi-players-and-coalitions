"""Tests des agents et des moteurs de recherche (jalons J3, J4, J5)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import make_config
from tron.agents import (
    AGENTS,
    GreedyAgent,
    MaxNAgent,
    NegamaxAgent,
    ParanoidAgent,
    RandomAgent,
    SearchMemory,
    SearchOptions,
    SocialAgent,
    build_agents,
    lp_upper,
    make_agent,
    search,
    social_matrix,
    two_camp_reduction,
)
from tron.agents.social import LinearBound, sparse_rows
from tron.core import PASS, ConfigError, GameConfig, GameState
from tron.eval import Valuation, VoronoiHeuristic


def play(config: GameConfig, max_plies: int = 10_000) -> GameState:
    agents = build_agents(config)
    st = GameState(config)
    while not st.is_terminal() and st.ply < max_plies:
        p = st.current
        move = agents[p].choose_move(st.copy(), p).move if st.has_legal_move() else PASS
        st.apply(move)
    return st


def midgame(
    rng: random.Random, w: int, h: int, n: int, plies: int, teams: list[str] | None = None
) -> GameState:
    cfg = make_config(
        w,
        h,
        n,
        teams=teams,
        seed=rng.randrange(10_000),
        placement={"mode": "random", "min_distance": 3},
    )
    st = GameState(cfg)
    for _ in range(plies):
        moves = st.legal_moves()
        if st.is_terminal() or not moves:
            break
        st.apply(rng.choice(moves))
    return st


def snapshot(st: GameState) -> tuple:
    return st.grid.tobytes(), st.heads.tobytes(), st.alive.tobytes(), st.current, st.ply, st.key


# ------------------------------------------------------------- J3 : base
def test_registry_and_errors() -> None:
    assert {"random", "greedy", "negamax", "paranoid", "maxn", "social"} <= set(AGENTS)
    cfg = make_config(8, 8, 3)
    with pytest.raises(ConfigError):
        make_agent("inconnu", 0, cfg)
    with pytest.raises(ConfigError):
        make_agent("paranoid", 0, cfg, profondeur=3)
    with pytest.raises(ConfigError):
        make_agent("negamax", 0, cfg)
    with pytest.raises(ConfigError):
        make_agent("greedy", 0, cfg, tie_break="pile")
    with pytest.raises(ConfigError):
        make_agent("social", 0, cfg, profile="bizarre")
    with pytest.raises(ConfigError):
        make_agent("social", 0, cfg, profile="maxn", engine="alphabeta").choose_move(GameState(cfg))
    with pytest.raises(ConfigError):
        make_agent("maxn", 0, cfg, ordering="magique")


def test_random_agent_plays_safe_moves(rng: random.Random) -> None:
    cfg = make_config(8, 8, 2)
    agent = RandomAgent(0, cfg)
    st = GameState(cfg)
    for _ in range(20):
        if st.is_terminal():
            break
        if st.current == 0:
            d = agent.choose_move(st)
            if st.legal_moves():
                assert d.move in st.legal_moves()
            st.apply(d.move)
        else:
            moves = st.legal_moves()
            st.apply(rng.choice(moves) if moves else PASS)


def test_agents_restore_state_and_report_stats() -> None:
    rng = random.Random(2)
    st = midgame(rng, 9, 9, 3, 9)
    cfg = st.config
    for name in ("random", "greedy", "paranoid", "maxn", "social"):
        agent = make_agent(name, st.current, cfg, depth=3)
        before = snapshot(st)
        dec = agent.choose_move(st)
        assert snapshot(st) == before
        assert dec.move in st.legal_moves()
        assert dec.stats.time >= 0 and dec.stats.nodes >= 1
        assert dec.stats.to_dict()["engine"]
    with pytest.raises(ValueError):
        make_agent("greedy", (st.current + 1) % 3, cfg).choose_move(st, (st.current + 1) % 3)


def test_no_legal_move_returns_pass() -> None:
    cfg = make_config(5, 5, 2)
    st = GameState.from_text(
        cfg,
        """
        Aa...
        aa...
        .....
        .....
        ....B
    """,
    )
    for name in ("random", "greedy", "paranoid", "maxn"):
        assert make_agent(name, 0, cfg).choose_move(st).move == PASS


@pytest.mark.slow
def test_greedy_beats_random_in_duels() -> None:
    """J3 : le glouton gagne au moins 90 % des parties à deux joueurs contre l'aléatoire."""
    wins = 0
    games = 40
    for g in range(games):
        cfg = make_config(
            10,
            10,
            2,
            agents=["greedy", "random"],
            seed=g,
            first_player=g % 2,
            placement={"mode": "regular", "rotate": True},
        )
        out = play(cfg).outcome()
        assert out is not None
        wins += out.winners == (0,)
    assert wins / games >= 0.9


def test_tie_break_modes() -> None:
    cfg = make_config(9, 9, 2, placement={"positions": [[4, 4], [0, 0]]})
    st = GameState(cfg)
    fixed = GreedyAgent(0, cfg, tie_break="fixed")
    dec = fixed.choose_move(st)
    assert dec.stats.ties >= 2 and dec.move == 0  # nord en premier parmi les ex aequo
    rand = [GreedyAgent(0, cfg.with_seed(s)).choose_move(st).move for s in range(30)]
    assert len(set(rand)) >= 2
    assert rand == [GreedyAgent(0, cfg.with_seed(s)).choose_move(st).move for s in range(30)]


# ------------------------------------------------------ J4 : élagage exact
def _two_camp(st: GameState, p: int) -> dict:
    return {
        "coeffs": ((p, 1.0),),
        "signs": tuple(1 if q == p else -1 for q in range(st.n)),
        "camp": (p,),
    }


@pytest.mark.parametrize("n", [2, 3, 4])
def test_alphabeta_equals_full_minimax(n: int) -> None:
    rng = random.Random(100 + n)
    for _ in range(12):
        st = midgame(rng, 8, 8, n, rng.randint(n, 6 * n))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        val = Valuation(st.config, VoronoiHeuristic(8, 8, n))
        p = st.current
        depth = rng.randint(1, 5)
        full = search(
            "alphabeta", st, val, depth, options=SearchOptions(pruning=False), **_two_camp(st, p)
        )
        for opts in (SearchOptions(), SearchOptions("history", True, True)):
            fast = search("alphabeta", st, val, depth, options=opts, **_two_camp(st, p))
            assert fast.value == pytest.approx(full.value, abs=1e-12)
            assert fast.ties == full.ties
            if not opts.iterative:
                assert fast.nodes <= full.nodes
        assert fast.vector is not None
        assert fast.vector[p] == pytest.approx(full.value, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_maxn_shallow_pruning_is_exact(n: int) -> None:
    rng = random.Random(200 + n)
    pruned_total = 0
    tested = 0
    for _ in range(30):
        st = midgame(rng, 9, 9, n, rng.randint(n, 4 * n))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        agent = MaxNAgent(st.current, st.config, depth=rng.randint(2, 5))
        plain = MaxNAgent(st.current, st.config, depth=agent.depth, pruning=False)
        ordered = MaxNAgent(
            st.current,
            st.config,
            depth=agent.depth,
            ordering="history",
            tt=True,
            iterative_deepening=True,
        )
        a = agent.run_search(st, agent.depth)
        b = plain.run_search(st, agent.depth)
        c = ordered.run_search(st, agent.depth)
        assert a.value == b.value == c.value
        assert a.ties == b.ties == c.ties
        assert a.vector == b.vector == c.vector
        assert a.nodes <= b.nodes
        pruned_total += a.prunes
        tested += 1
    assert tested >= 15
    if n == 2:  # à deux joueurs, la règle de Korf (x + y > 1 + μ) coupe souvent
        assert pruned_total > 0


def test_two_players_all_algorithms_agree() -> None:
    """J4 : à deux joueurs, Negamax, Paranoid, MaxN et SOS donnent la même valeur."""
    rng = random.Random(9)
    checked = 0
    for _ in range(25):
        st = midgame(rng, 8, 8, 2, rng.randint(2, 30))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        p, q = st.current, 1 - st.current
        cfg = st.config
        depth = rng.randint(1, 6)
        kw = {"depth": depth}
        par = ParanoidAgent(p, cfg, **kw).run_search(st, depth)
        mx = MaxNAgent(p, cfg, **kw).run_search(st, depth)
        nm = NegamaxAgent(p, cfg, **kw).run_search(st, depth)
        so_par = SocialAgent(p, cfg, profile="paranoid", engine="maxn", **kw).run_search(st, depth)
        so_mx = SocialAgent(p, cfg, profile="team", alpha=0.0, **kw).run_search(st, depth)
        assert par.vector is not None
        assert mx.value == pytest.approx(par.value, abs=1e-12)
        assert so_par.value == pytest.approx(par.value, abs=1e-12)
        assert nm.value == pytest.approx(par.vector[p] - par.vector[q], abs=1e-12)
        # Équipes d'un joueur : W = I se ramène à deux camps (f = v(p) − v(q), comme negamax).
        assert so_mx.engine == "alphabeta"
        assert so_mx.value == pytest.approx(nm.value, abs=1e-12)
        assert par.ties == mx.ties == nm.ties == so_par.ties == so_mx.ties
        if depth == 1:
            gr = GreedyAgent(p, cfg, tie_break="fixed").choose_move(st)
            assert gr.stats.value == pytest.approx(par.root_values[gr.move], abs=1e-12)
        checked += 1
    assert checked >= 15


# ---------------------------------------------------------- J5 : équipes
def test_social_profiles_reproduce_reference_agents() -> None:
    rng = random.Random(21)
    for _ in range(10):
        n = rng.choice([3, 4])
        st = midgame(rng, 8, 8, n, rng.randint(n, 5 * n))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        p, cfg, depth = st.current, st.config, rng.randint(2, 4)
        par = ParanoidAgent(p, cfg, depth=depth).run_search(st, depth)
        auto = SocialAgent(p, cfg, profile="paranoid", depth=depth)
        assert auto.engine_name(st) == "alphabeta"
        vec = SocialAgent(p, cfg, profile="paranoid", engine="maxn", depth=depth)
        vec_plain = SocialAgent(
            p, cfg, profile="paranoid", engine="maxn", depth=depth, pruning=False
        )
        r_auto, r_vec, r_plain = (a.run_search(st, depth) for a in (auto, vec, vec_plain))
        assert r_auto.value == pytest.approx(par.value, abs=1e-12)
        assert r_vec.value == pytest.approx(par.value, abs=1e-12)
        assert r_plain.value == pytest.approx(par.value, abs=1e-12)
        assert r_auto.ties == r_vec.ties == r_plain.ties == par.ties
        assert r_vec.nodes <= r_plain.nodes
        mx = MaxNAgent(p, cfg, depth=depth).run_search(st, depth)
        so = SocialAgent(p, cfg, profile="team", alpha=0.0, depth=depth)
        assert so.engine_name(st) == "maxn"
        r_so = so.run_search(st, depth)
        assert (r_so.value, r_so.ties, r_so.vector) == (mx.value, mx.ties, mx.vector)


def test_generalized_pruning_is_exact_for_arbitrary_matrices() -> None:
    rng = random.Random(33)
    total_prunes = 0
    tested = 0
    for trial in range(40):
        n = rng.choice([3, 4])
        teams = None
        if trial % 4 == 0:  # parties en équipes : borne de somme plus lâche, exactitude seule
            teams = ["A", "B"] + [rng.choice("AB") for _ in range(n - 2)]
        st = midgame(rng, 9, 9, n, rng.randint(n, 4 * n), teams=teams)
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        matrix = [
            [1.0 if i == j else round(rng.uniform(-1, 1), 2) for j in range(n)] for i in range(n)
        ]
        if teams is not None:
            matrix = social_matrix("team", st.config.team_index, st.current, alpha=0.5)
        kw = {"profile": "custom", "matrix": matrix, "engine": "maxn", "depth": 4}
        a = SocialAgent(st.current, st.config, **kw)
        b = SocialAgent(st.current, st.config, pruning=False, **kw)
        ra, rb = a.run_search(st, 4), b.run_search(st, 4)
        assert ra.value == rb.value and ra.ties == rb.ties and ra.vector == rb.vector
        assert ra.nodes <= rb.nodes
        total_prunes += ra.prunes
        tested += 1
    assert tested >= 20
    assert total_prunes > 0


def test_two_team_reduction_matches_vector_search_without_eliminations() -> None:
    rng = random.Random(44)
    checked = 0
    for _ in range(12):
        st = midgame(rng, 10, 10, 4, rng.randint(0, 6), teams=["A", "B", "A", "B"])
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        p, cfg = st.current, st.config
        ab = SocialAgent(p, cfg, profile="team", alpha=1.0, depth=3)
        assert ab.engine_name(st) == "alphabeta"
        assert ab.reduction(st).kind == "two_teams"  # type: ignore[union-attr]
        vec = SocialAgent(p, cfg, profile="team", alpha=1.0, engine="maxn", depth=3)
        ra, rv = ab.run_search(st, 3), vec.run_search(st, 3)
        assert rv.vector is not None
        if any(not st.is_alive(i) for i in range(4)) or min(rv.vector) < 1e-6:
            continue  # réduction exacte seulement sans joueur éliminé
        assert ra.value == pytest.approx(2 * rv.value - 1, abs=1e-12)
        assert ra.ties == rv.ties
        checked += 1
    assert checked >= 5


# ----------------------------------------------------- matrices et bornes
def test_social_matrices_and_reductions() -> None:
    team = [0, 1, 0, 1]
    alive = [True] * 4
    assert social_matrix("maxn", team, 0) == [[float(i == j) for j in range(4)] for i in range(4)]
    par = social_matrix("paranoid", team, 2)
    assert par[2][2] == 1 and par[0][2] == -1 and par[0][0] == 0
    red = two_camp_reduction(par, team, 2, alive)
    assert red is not None and red.kind == "paranoid" and red.signs == (-1, -1, 1, -1)
    tp = social_matrix("team_paranoid", team, 1)
    red = two_camp_reduction(tp, team, 1, alive)
    assert red is not None and red.kind == "rank1" and red.signs == (-1, 1, -1, 1)
    assert dict(red.coeffs) == {1: 1.0, 3: 1.0}
    t1 = social_matrix("team", team, 0, alpha=1.0)
    red = two_camp_reduction(t1, team, 0, alive)
    assert red is not None and red.kind == "two_teams"
    assert two_camp_reduction(social_matrix("team", team, 0, alpha=0.5), team, 0, alive) is None
    three = [0, 1, 2, 2]
    assert two_camp_reduction(social_matrix("team", three, 0), three, 0, alive) is None
    assert sparse_rows(social_matrix("maxn", team, 0)) == [None] * 4
    with pytest.raises(ValueError):
        social_matrix("team", team, 0, alpha=2.0)
    with pytest.raises(ValueError):
        social_matrix("custom", team, 0, matrix=[[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        social_matrix("custom", team, 0)


def test_lp_bound_matches_scipy_and_is_valid() -> None:
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = np.random.default_rng(5)
    for _ in range(80):
        n = int(rng.integers(2, 6))
        a = rng.uniform(-1, 1, n).round(2)
        b = rng.uniform(-1, 1, n).round(2)
        s = float(rng.uniform(0.8, 2.0))
        y = float(rng.uniform(-0.5, 1.0))
        res = linprog(-b, A_ub=np.vstack([-a, np.ones(n)]), b_ub=[-y, s], bounds=[(0, None)] * n)
        got = lp_upper(list(a), list(b), s, y)
        if res.status == 2:
            assert got == -np.inf
        else:
            assert got == pytest.approx(-res.fun, abs=1e-9)
            bound = LinearBound([list(b), list(a)], s)
            assert bound(0, 1, y) == pytest.approx(got, abs=1e-9)
    ident = LinearBound([[1.0, 0.0], [0.0, 1.0]], 1.0)
    assert ident(0, 1, 0.3) == pytest.approx(0.7)


def test_iterative_deepening_with_time_limit_returns_a_move() -> None:
    rng = random.Random(4)
    st = midgame(rng, 10, 10, 3, 6)
    agent = ParanoidAgent(st.current, st.config, depth=40, time_limit=0.05)
    before = snapshot(st)
    dec = agent.choose_move(st)
    assert snapshot(st) == before
    assert dec.move in st.legal_moves()
    assert 1 <= dec.stats.depth < 40
    mem = SearchMemory.create(SearchOptions(), st.n, st.size)
    assert mem.tt is None and mem.history is None


def test_rounds_parameter_scales_depth() -> None:
    cfg = make_config(8, 8, 3)
    st = GameState(cfg)
    agent = ParanoidAgent(0, cfg, rounds=2)
    assert agent.effective_depth(st) == 6
    st.apply(PASS)
    assert agent.effective_depth(st) == 4
    assert "r=2" in agent.describe()
