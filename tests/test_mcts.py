"""Tests de l'agent Monte-Carlo Tree Search (P04)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from conftest import make_config
from tron.agents import AGENTS, MCTSAgent, make_agent
from tron.core import PASS, ConfigError, GameConfig, GameState
from tron.experiments import Plan
from tron.sim import Match

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def snapshot(st: GameState) -> tuple[Any, ...]:
    return (
        st.grid.tobytes(),
        st.heads.tobytes(),
        st.alive.tobytes(),
        st.death_ply.tobytes(),
        st.current,
        st.ply,
        st.key,
        list(st.team_alive),
        st.teams_alive,
        list(st.elim_order),
        len(st._hist),
        st.n_moves,
    )


def midgame(n: int, size: int = 9, plies: int = 6, seed: int = 0) -> GameState:
    cfg = make_config(size, size, n, seed=seed, placement={"mode": "regular", "rotate": True})
    st = GameState(cfg)
    rng = random.Random(seed)
    for _ in range(plies):
        moves = st.legal_moves()
        if st.is_terminal() or not moves:
            break
        st.apply(rng.choice(moves))
    return st


def test_registry_and_parameter_validation() -> None:
    assert "mcts" in AGENTS
    cfg = make_config(8, 8, 2)
    agent = make_agent("mcts", 0, cfg, simulations=10, depth=3)
    assert isinstance(agent, MCTSAgent) and "mcts(n=10" in agent.describe()
    for bad in (
        {"simulations": 0},
        {"simulations": 2.5},
        {"simulations": True},
        {"exploration": -0.1},
        {"exploration": float("nan")},
        {"exploration": float("inf")},
        {"rollout_policy": "astucieuse"},
        {"rollout_depth": 0},
        {"time_limit": 0},
        {"profile": "inconnu"},
        {"inconnu": 1},
    ):
        with pytest.raises(ConfigError):
            make_agent("mcts", 0, cfg, **bad)


@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_state_restored_legal_move_and_exact_simulation_count(policy: str) -> None:
    st = midgame(3, plies=5, seed=2)
    assert len(st.legal_moves()) >= 2
    agent = MCTSAgent(st.current, st.config, simulations=60, rollout_policy=policy)
    before = snapshot(st)
    dec = agent.choose_move(st)
    assert snapshot(st) == before
    assert dec.move in st.legal_moves()
    s = dec.stats
    assert s.leaves == 60 and s.engine == "mcts" and s.prunes == 0
    assert 1 < s.nodes <= 61 and s.depth >= 1 and s.ties >= 1
    assert s.pv and s.pv[0] == dec.move and s.time >= 0
    assert sum(v for v, _ in agent.last_root.values()) == 60
    assert dec.move in agent.last_root


def test_deterministic_for_fixed_seed_and_varies_with_seed() -> None:
    st = midgame(2, plies=4, seed=5)
    cfg = st.config

    def decide(seed: int) -> tuple[int, int, tuple[int, ...]]:
        a = MCTSAgent(st.current, cfg.with_seed(seed), simulations=40)
        d = a.choose_move(st)
        return d.move, d.stats.nodes, d.stats.pv

    assert decide(1) == decide(1)
    outcomes = {decide(s) for s in range(8)}
    assert len(outcomes) > 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_works_from_two_to_eight_players(n: int) -> None:
    st = midgame(n, size=12, plies=n, seed=n)
    if len(st.legal_moves()) < 2:
        pytest.skip("position sans choix")
    agent = MCTSAgent(st.current, st.config, simulations=25, rollout_depth=20)
    before = snapshot(st)
    dec = agent.choose_move(st)
    assert snapshot(st) == before and dec.move in st.legal_moves()
    assert dec.stats.leaves == 25


def test_pass_forced_move_terminal_and_small_board() -> None:
    cfg = make_config(5, 5, 2)
    trapped = GameState.from_text(
        cfg,
        """
        Aa...
        aa...
        .....
        .....
        ....B
        """,
    )
    assert MCTSAgent(0, cfg, simulations=5).choose_move(trapped).move == PASS
    forced = GameState.from_text(
        cfg,
        """
        A....
        a....
        .....
        .....
        ....B
        """,
    )
    forced.key = forced.compute_key()
    dec = MCTSAgent(0, cfg, simulations=5).choose_move(forced)
    assert dec.move == 1 and dec.stats.leaves == 0  # coup unique : pas de simulation
    full = MCTSAgent(0, cfg, simulations=5, forced_shortcut=False).choose_move(forced)
    assert full.move == 1 and full.stats.leaves == 5
    # Une simulation qui élimine un joueur (PASS) et termine la partie reste cohérente.
    tight = GameState.from_text(
        cfg,
        """
        A.bbb
        aabBb
        bbbbb
        .....
        .....
        """,
    )
    agent = MCTSAgent(0, cfg, simulations=30, forced_shortcut=False)
    before = snapshot(tight)
    dec = agent.choose_move(tight)
    assert snapshot(tight) == before and dec.move == 1
    assert dec.stats.value > 1.0  # J1 est enfermé : la victoire est trouvée


def test_time_limit_stops_early_but_plays() -> None:
    st = midgame(4, size=16, plies=4, seed=1)
    agent = MCTSAgent(st.current, st.config, simulations=10**6, time_limit=0.05)
    dec = agent.choose_move(st)
    assert dec.move in st.legal_moves() and 1 <= dec.stats.leaves < 10**6


def test_social_profile_changes_utilities() -> None:
    cfg = make_config(9, 9, 4, teams=["A", "B", "A", "B"], seed=3)
    st = midgame(4, plies=4, seed=3)
    st = GameState(cfg, positions=[st.head(p) for p in range(4)])
    team = MCTSAgent(0, cfg, simulations=30, profile="team", alpha=1.0)
    assert team._rows[0] == [(0, 1.0), (2, 1.0)]
    assert team.choose_move(st).move in st.legal_moves()


def test_duels_against_reference_agents_run_quickly() -> None:
    for opponent in ("random", "greedy"):
        cfg = make_config(
            8,
            8,
            2,
            agents=["mcts", opponent],
            params=[{"simulations": 40}, {}],
            seed=4,
            placement={"mode": "regular", "rotate": True},
        )
        a = Match(cfg).run()
        b = Match(cfg).run()
        assert a.winning_team in ("J0", "J1")
        assert (a.ranks, a.n_moves) == (b.ranks, b.n_moves)
        assert a.totals[0].leaves > 0


@pytest.mark.slow
def test_mcts_beats_random() -> None:
    wins = 0
    for g in range(16):
        cfg = make_config(
            10,
            10,
            2,
            agents=["mcts", "random"],
            params=[{"simulations": 100}, {}],
            seed=g,
            first_player=g % 2,
            placement={"mode": "regular", "rotate": True},
        )
        wins += Match(cfg, metrics=False).run().winners == (0,)
    assert wins >= 13


def test_example_configs_and_plans_load() -> None:
    cfg = GameConfig.from_yaml(CONFIGS / "games" / "mcts_duel_12x12.yaml")
    assert cfg.players[0].agent == "mcts" and cfg.players[0].params["simulations"] == 400
    comp = GameConfig.from_yaml(CONFIGS / "games" / "composite_20x20.yaml")
    assert comp.players[1].agent == "mcts"
    plan = Plan.from_yaml(CONFIGS / "plans" / "mcts_comparison.yaml")
    conds = plan.conditions()
    assert len(conds) == 40
    mcts_solo = next(c for c in conds if c.factors["solo_agent"] == "mcts")
    assert mcts_solo.config["players"][0]["simulations"] == 300
    other = next(c for c in conds if c.factors["solo_agent"] == "paranoid")
    assert "simulations" not in other.config["players"][0]
    sims = Plan.from_yaml(CONFIGS / "plans" / "mcts_simulations.yaml")
    assert {c.factors["simulations"] for c in sims.conditions()} == {50, 100, 250, 500, 1000}
