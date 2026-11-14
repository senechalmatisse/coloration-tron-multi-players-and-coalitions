"""Tests de la sauvegarde et de la reprise exacte d'une partie."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import make_config
from tron.cli import main
from tron.core import PASS, GameConfig, GameState
from tron.sim import CHECKPOINT_FORMAT, CheckpointError, GameLog, Match, load_checkpoint

CONFIG = {
    "grid": 11,
    "seed": 5,
    "placement": {"mode": "regular", "rotate": True},
    "diplomacy": {"enabled": True, "betrayal_penalty": 0.2},
    "players": [
        {"name": "par", "team": "A", "agent": "paranoid", "depth": 3},
        {"name": "dyn", "team": "B", "agent": "social",   "depth": 2, "profile": "dynamic_team"},
        {"name": "ada", "team": "C", "agent": "adaptive", "depth": 2, "adaptation_rate": 0.5},
        {"name": "mc",  "team": "D", "agent": "mcts",     "simulations": 30},
        {"name": "rnd", "team": "B", "agent": "random"},
    ],
}


def _signature(e: Any) -> tuple[Any, ...]:
    return (
        e.ply,
        e.player,
        e.direction,
        e.eliminated,
        e.stats.nodes if e.stats else None,
        e.stats.value if e.stats else None,
        e.stats.pv if e.stats else None,
        e.diplomacy,
        e.details.to_dict() if e.details else None,
    )


# état
def test_state_snapshot_roundtrip_restores_every_field() -> None:
    rng = random.Random(4)
    cfg = make_config(9, 7, 4, teams=["A", "B", "A", "C"], seed=2)
    for _ in range(5):
        st = GameState(cfg)
        for _ in range(rng.randint(0, 40)):
            if st.is_terminal():
                break
            moves = st.legal_moves()
            st.apply(rng.choice(moves) if moves and rng.random() < 0.9 else PASS)
        data = json.loads(json.dumps(st.to_snapshot()))
        back = GameState.from_snapshot(cfg, data)
        assert np.array_equal(back.grid, st.grid) and np.array_equal(back.heads, st.heads)
        assert np.array_equal(back.alive, st.alive)
        assert np.array_equal(back.death_ply, st.death_ply)
        assert (back.current, back.ply, back.n_moves, back.key) == (
            st.current,
            st.ply,
            st.n_moves,
            st.key,
        )
        assert back.elim_order == st.elim_order and back.team_alive == st.team_alive
        assert back.teams_alive == st.teams_alive and back.positions == st.positions
        assert back._hist == st._hist
        while st.ply:  # l'historique restauré permet de tout défaire
            st.undo()
            back.undo()
            assert back.key == st.key and np.array_equal(back.grid, st.grid)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d.update(key=d["key"] ^ 1),
        lambda d: d.update(width=d["width"] + 1),
        lambda d: d.update(grid=d["grid"][:-1]),
        lambda d: d.update(grid="9" + d["grid"][1:]),
        lambda d: d.update(team_alive=[9] * len(d["team_alive"])),
        lambda d: d.update(heads=[d["heads"][1], d["heads"][0], *d["heads"][2:]]),
        lambda d: d.pop("current"),
    ],
)
def test_inconsistent_snapshots_are_rejected(corrupt: Any) -> None:
    cfg = make_config(8, 8, 3, seed=1)
    st = GameState(cfg)
    st.apply(st.legal_moves()[0])
    data = st.to_snapshot()
    corrupt(data)
    with pytest.raises(ValueError):
        GameState.from_snapshot(cfg, data)


# partie
@pytest.mark.parametrize("cut", [0, 9, 31])
def test_resumed_game_is_identical_to_uninterrupted_game(tmp_path: Path, cut: int) -> None:
    cfg = GameConfig.from_dict(CONFIG)
    ref = Match(cfg)
    expected = ref.run()
    m = Match(cfg)
    for _ in range(cut):
        m.step()
    path = m.save_checkpoint(tmp_path / "c.json")
    assert not list(tmp_path.glob("*.tmp"))  # écriture atomique
    resumed = Match.from_checkpoint(path)
    assert resumed.state.ply == cut
    result = resumed.run()
    assert [_signature(e) for e in resumed.events] == [_signature(e) for e in ref.events[cut:]]
    assert (result.winning_team, result.ranks, result.n_plies) == (
        expected.winning_team,
        expected.ranks,
        expected.n_plies,
    )
    assert result.metrics == expected.metrics
    assert [(t.decisions, t.nodes, t.leaves) for t in result.totals] == [
        (t.decisions, t.nodes, t.leaves) for t in expected.totals
    ]


def test_checkpoint_content_and_finished_game(tmp_path: Path) -> None:
    cfg = GameConfig.from_dict(CONFIG)
    m = Match(cfg)
    m.run()
    path = m.save_checkpoint(tmp_path / "fin.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == CHECKPOINT_FORMAT and data["result"] is not None
    assert {"rng", "tt", "history"} <= set(data["agents"][0]["state"])
    assert {"inner", "model", "weights"} <= set(data["agents"][2]["state"])
    assert data["diplomacy"]["events"] and len(data["moves"]) == m.state.ply
    again = load_checkpoint(path)
    assert again.state.is_terminal()
    assert again.run().ranks == m.result.ranks  # type: ignore[union-attr]


def test_incompatible_partial_and_corrupted_checkpoints(tmp_path: Path) -> None:
    cfg = GameConfig.from_dict(CONFIG)
    m = Match(cfg)
    for _ in range(5):
        m.step()
    path = m.save_checkpoint(tmp_path / "c.json")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    def write(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    with pytest.raises(CheckpointError, match="incompatible"):
        load_checkpoint(write("v9.json", json.dumps({**data, "format": "tron-checkpoint/9"})))
    with pytest.raises(CheckpointError, match="corrompue"):
        load_checkpoint(write("partial.json", text[: len(text) // 2]))
    without_agents = {k: v for k, v in data.items() if k != "agents"}
    with pytest.raises(CheckpointError, match="incomplète"):
        load_checkpoint(write("missing.json", json.dumps(without_agents)))
    bad_key = json.loads(text)
    bad_key["state"]["key"] ^= 12345
    with pytest.raises(CheckpointError, match="Zobrist"):
        load_checkpoint(write("key.json", json.dumps(bad_key)))
    bad_agent = json.loads(text)
    bad_agent["agents"][0]["agent"] = "maxn"
    with pytest.raises(CheckpointError):
        load_checkpoint(write("agent.json", json.dumps(bad_agent)))
    bad_moves = json.loads(text)
    bad_moves["moves"] = bad_moves["moves"][:-1]
    with pytest.raises(CheckpointError):
        load_checkpoint(write("moves.json", json.dumps(bad_moves)))
    with pytest.raises(CheckpointError):
        load_checkpoint(write("list.json", "[1, 2]"))


def test_cli_checkpoint_chain_and_complete_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = tmp_path / "partie.yaml"
    GameConfig.from_dict(CONFIG).to_yaml(cfg_path)
    first = tmp_path / "c1.json"
    second = tmp_path / "c2.json"
    log = tmp_path / "complet.jsonl"
    args = ["play", str(cfg_path), "--no-gui", "--checkpoint", str(first), "--stop-after", "12"]
    assert main(args) == 0
    assert (
        main(["resume-checkpoint", str(first), "--stop-after", "30", "--checkpoint", str(second)])
        == 0
    )
    assert main(["resume-checkpoint", str(second), "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "arrêtée au demi-coup 12" in out and "reprise au demi-coup 30" in out
    ref = Match(GameConfig.from_dict(CONFIG)).run()
    replay = GameLog.load(log)
    assert len(replay.moves) == ref.n_plies
    final = replay.final_state()
    assert final.outcome() is not None and final.outcome().ranks == ref.ranks  # type: ignore[union-attr]
    assert main(["play", str(cfg_path), "--checkpoint", str(first)]) == 2  # exige --no-gui
