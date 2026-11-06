"""Tests du moteur de jeu (jalon J1)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import make_config, random_playout
from tron.core import PASS, ConfigError, GameConfig, GameState, derive_seed, make_rng
from tron.core.directions import direction_name, parse_direction
from tron.core.placement import initial_positions, random_positions, regular_positions


# ------------------------------------------------------------------ config
def test_config_formats_and_defaults() -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 12,
            "defaults": {"agent": "paranoid", "depth": 4, "tie_break": "fixed"},
            "players": [
                {"name": "a", "team": "X"},
                {"name": "b", "team": "Y", "depth": 2},
                "random",
            ],
        }
    )
    assert (cfg.width, cfg.height) == (12, 12)
    assert cfg.players[0].agent == "paranoid"
    assert cfg.players[0].params == {"tie_break": "fixed"}
    assert cfg.players[1].depth == 2
    assert cfg.players[2].agent == "random" and cfg.players[2].team == "J2"
    assert cfg.team_names == ("X", "Y", "J2")
    assert cfg.team_index == (0, 1, 2)
    cfg2 = GameConfig.from_dict({"grid": {"width": 7, "height": 9}, "players": ["a", "b"]})
    assert cfg2.height == 9
    assert GameConfig.from_dict({"width": 7, "height": 8, "players": ["a", "b"]}).width == 7


def test_config_roundtrip(tmp_path) -> None:
    cfg = make_config(15, 11, 4, teams=["A", "A", "B", "B"], params={"alpha": 0.5})
    assert GameConfig.from_dict(cfg.to_dict()) == cfg
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert GameConfig.from_yaml(path) == cfg
    assert cfg.teammates(0) == (1,)
    assert cfg.team_sizes == (2, 2)


@pytest.mark.parametrize(
    "data",
    [
        {"grid": 4, "players": ["a", "b"]},
        {"grid": 101, "players": ["a", "b"]},
        {"grid": 10, "players": ["a"]},
        {"grid": 10, "players": [{"name": "a", "team": "T"}, {"name": "b", "team": "T"}]},
        {"grid": 10, "players": [{"name": "a"}, {"name": "a"}]},
        {"grid": 10, "players": ["a", "b"], "first_player": 2},
        {"grid": 10, "players": ["a", "b"], "placement": {"mode": "bizarre"}},
        {"grid": 10, "players": ["a", "b"], "placement": {"positions": [[0, 0], [0, 0]]}},
        {"grid": 10, "players": ["a", "b"], "placement": {"positions": [[0, 0], [10, 0]]}},
        {"grid": 10, "players": ["a", "b"], "inconnu": 1},
        {"grid": 10, "players": [{"name": "a", "depth": 0}, "b"]},
    ],
)
def test_config_validation(data: dict) -> None:
    with pytest.raises(ConfigError):
        GameConfig.from_dict(data)


# --------------------------------------------------------------- placement
def test_regular_placement_is_symmetric() -> None:
    pos = regular_positions(20, 20, 2, 0.35)
    (x0, y0), (x1, y1) = pos
    assert y0 == y1 and x0 + x1 == 19
    pos4 = regular_positions(20, 20, 4, 0.35)
    assert len(set(pos4)) == 4
    assert sorted(x for x, _ in pos4) == sorted(y for _, y in pos4)


def test_placement_depends_only_on_seed() -> None:
    cfg = make_config(20, 20, 3, placement={"mode": "regular", "rotate": True}, seed=5)
    assert initial_positions(cfg) == initial_positions(cfg.with_seed(5))
    assert initial_positions(cfg) != initial_positions(cfg.with_seed(6))
    cfg_r = make_config(20, 20, 6, placement={"mode": "random", "min_distance": 5}, seed=3)
    pos = initial_positions(cfg_r)
    assert pos == initial_positions(cfg_r)
    for i, (x, y) in enumerate(pos):
        for u, v in pos[i + 1 :]:
            assert abs(x - u) + abs(y - v) >= 5
    with pytest.raises(ConfigError):
        random_positions(5, 5, 8, 10, make_rng(0))


def test_dense_regular_placement_has_distinct_cells() -> None:
    pos = regular_positions(5, 5, 8, 0.1)
    assert len(set(pos)) == 8


def test_seeds() -> None:
    assert derive_seed(1, 2) == derive_seed(1, 2)
    assert derive_seed(1, 2) != derive_seed(1, 3)
    assert 0 <= derive_seed(7) < 2**63
    assert make_rng(3, 1).integers(1000) == make_rng(3, 1).integers(1000)


def test_directions() -> None:
    assert parse_direction("O") == parse_direction("W") == 3
    assert parse_direction(None) == PASS == parse_direction("-")
    assert direction_name(0) == "N" and direction_name(PASS) == "-"
    with pytest.raises(ValueError):
        parse_direction("Z")
    with pytest.raises(ValueError):
        parse_direction(7)


# ------------------------------------------------------------------ règles
def test_initial_state() -> None:
    cfg = make_config(10, 10, 2, first_player=1)
    st = GameState(cfg)
    assert st.current == 1
    for p in range(2):
        x, y = st.head(p)
        assert st.cell(x, y) == p
    assert int((st.grid >= 0).sum()) == 2
    assert int((st.grid == -2).sum()) == 2 * 12 + 2 * 10
    assert st.key == st.compute_key()


def test_legal_moves_and_borders() -> None:
    cfg = make_config(5, 5, 2, placement={"positions": [[0, 0], [4, 4]]})
    st = GameState(cfg)
    assert st.legal_moves(0) == [1, 2]  # E, S
    assert st.legal_moves(1) == [0, 3]  # N, O
    assert st.apply(0) is False  # nord = hors grille : élimination
    assert not st.is_alive(0) and st.is_terminal()
    out = st.outcome()
    assert out is not None and out.winners == (1,) and out.ranks == (2, 1)


def test_collision_with_wall_eliminates() -> None:
    cfg = make_config(5, 5, 2, placement={"positions": [[1, 2], [3, 2]]})
    st = GameState(cfg)
    assert st.apply(1)  # J0 va en (2, 2)
    assert st.current == 1
    assert st.apply(3) is False  # J1 percute le mur de J0
    assert st.outcome().winners == (0,)  # type: ignore[union-attr]


def test_apply_undo_restores_everything(rng: random.Random) -> None:
    cfg = make_config(9, 7, 4, teams=["A", "B", "A", "C"])
    st = GameState(cfg)
    snapshots = []
    while not st.is_terminal():
        snapshots.append(
            (
                st.grid.copy(),
                st.heads.copy(),
                st.alive.copy(),
                st.current,
                st.key,
                st.ply,
                list(st.team_alive),
                st.teams_alive,
                list(st.elim_order),
                st.n_moves,
            )
        )
        moves = st.legal_moves()
        if moves and rng.random() < 0.9:
            st.apply(rng.choice(moves))
        else:
            st.apply(PASS)
        assert st.key == st.compute_key()
    for snap in reversed(snapshots):
        st.undo()
        assert np.array_equal(st.grid, snap[0])
        assert np.array_equal(st.heads, snap[1])
        assert np.array_equal(st.alive, snap[2])
        assert (st.current, st.key, st.ply) == snap[3:6]
        assert st.team_alive == snap[6] and st.teams_alive == snap[7]
        assert st.elim_order == snap[8] and st.n_moves == snap[9]


def test_rotation_skips_eliminated_players() -> None:
    cfg = make_config(10, 10, 3)
    st = GameState(cfg)
    st.apply(PASS)  # J0 éliminé
    assert st.current == 1
    st.apply(st.legal_moves()[0])
    assert st.current == 2
    st.apply(st.legal_moves()[0])
    assert st.current == 1
    assert st.play_order() == [1, 2]


def test_team_game_ends_when_one_team_left() -> None:
    cfg = make_config(10, 10, 4, teams=["A", "B", "A", "B"])
    st = GameState(cfg)
    st.apply(PASS)  # J0 (A)
    assert not st.is_terminal()
    st.apply(PASS)  # J1 (B)
    assert not st.is_terminal()
    st.apply(PASS)  # J2 (A) : il ne reste que J3 (B)
    assert st.is_terminal()
    out = st.outcome()
    assert out is not None
    assert out.winning_team == 1 and out.winners == (3,)
    assert out.ranks == (4, 3, 2, 1)
    assert out.elimination_order == (0, 1, 2)


def test_team_win_with_several_survivors() -> None:
    cfg = make_config(10, 10, 3, teams=["A", "B", "B"])
    st = GameState(cfg)
    st.apply(PASS)
    out = st.outcome()
    assert out is not None and out.winners == (1, 2) and out.ranks == (3, 1, 1)


def test_random_games_terminate_and_copy_is_independent(rng: random.Random) -> None:
    for seed in range(5):
        cfg = make_config(8, 8, 3, seed=seed, placement={"mode": "random"})
        st = GameState(cfg)
        cp = st.copy()
        plies = random_playout(st, rng)
        assert st.is_terminal()
        assert plies <= 64 + 3
        assert cp.ply == 0 and not cp.is_terminal()
        light = st.copy(history=False)
        assert np.array_equal(light.grid, st.grid) and light._hist == []
        out = st.outcome()
        assert out is not None and sorted(out.ranks)[0] == 1


def test_text_roundtrip() -> None:
    cfg = make_config(5, 5, 2)
    text = """
        A....
        a.bbB
        .....
        .....
        .....
    """
    st = GameState.from_text(cfg, text, current=1)
    assert st.to_text().splitlines()[1] == "a.bbB"
    assert st.legal_moves(0) == [1]
    assert st.key == st.compute_key()
    assert "ply=0" in repr(st)
