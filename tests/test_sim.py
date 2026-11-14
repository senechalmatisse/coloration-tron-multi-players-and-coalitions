"""Tests de la simulation, du journal et de l'arène."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from conftest import make_config
from tron.core import PASS
from tron.sim import GameLog, LogError, Match, Recorder, game_config, run_arena


def _config(seed: int = 3):
    return make_config(
        10,
        10,
        4,
        teams=["A", "B", "A", "B"],
        agents=["paranoid", "maxn", "social", "greedy"],
        depth=[3, 2, 2, 1],
        params=[{}, {}, {"profile": "team", "alpha": 1.0}, {}],
        seed=seed,
        placement={"mode": "regular", "rotate": True},
    )


def test_match_is_reproducible() -> None:
    a = Match(_config()).run()
    m = Match(_config())
    b = m.run()
    assert (a.winning_team, a.ranks, a.n_moves, a.n_plies) == (
        b.winning_team,
        b.ranks,
        b.n_moves,
        b.n_plies,
    )
    first = [(e.player, e.direction) for e in Match(_config()).events]
    assert first == []
    seq1 = [(e.player, e.direction) for e in m.events]
    m2 = Match(_config())
    m2.run()
    assert seq1 == [(e.player, e.direction) for e in m2.events]
    assert m.step() is None and m.finished
    assert sum(t.decisions for t in b.totals) <= b.n_plies
    assert b.winning_team in ("A", "B")


def test_different_seeds_give_different_games() -> None:
    games = {tuple((e.player, e.direction) for e in _run(s).events) for s in range(4)}
    assert len(games) > 1


def _run(seed: int) -> Match:
    m = Match(_config(seed))
    m.run()
    return m


def test_recorder_roundtrip(tmp_path) -> None:
    path = tmp_path / "partie.jsonl"
    match = Match(_config(), observers=[Recorder(path)])
    result = match.run()
    log = GameLog.load(path)
    assert log.config == match.config
    assert len(log.moves) == result.n_plies
    final = log.final_state()
    assert np.array_equal(final.grid, match.state.grid)
    assert final.outcome() == match.state.outcome()
    assert log.result is not None and log.result["winning_team"] == result.winning_team
    states = list(log.states())
    assert len(states) == result.n_plies + 1
    assert log.agents and log.agents[0].startswith("paranoid")
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    assert rec["type"] == "move" and "nodes" in rec


def test_corrupted_log_is_detected(tmp_path) -> None:
    path = tmp_path / "p.jsonl"
    Match(_config(), observers=[Recorder(path)]).run()
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["elim"] = not rec["elim"]
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LogError):
        GameLog.load(path).final_state()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{pas du json\n", encoding="utf-8")
    with pytest.raises(LogError):
        GameLog.load(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"type": "move"}\n', encoding="utf-8")
    with pytest.raises(LogError):
        GameLog.load(empty)


def test_pass_events_when_trapped() -> None:
    cfg = make_config(
        5, 5, 2, agents=["greedy", "greedy"], placement={"positions": [[0, 0], [4, 4]]}
    )
    m = Match(cfg)
    m.state.apply(1)
    m.state.apply(3)
    result = m.run()
    assert result.n_plies >= 3
    elim = [e for e in m.events if e.eliminated]
    assert len(elim) == 1 and elim[0].target is None
    assert elim[0].direction == PASS or elim[0].stats is not None


def test_arena_rotates_first_player_and_writes_outputs(tmp_path) -> None:
    cfg = make_config(8, 8, 3, agents="greedy", seed=11)
    seen: list[int] = []
    results = run_arena(cfg, games=3, out_dir=tmp_path, on_result=lambda g, r: seen.append(g))
    assert seen == [0, 1, 2]
    assert [r.first_player for r in results] == [0, 1, 2]
    assert len({r.seed for r in results}) == 3
    with (tmp_path / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and rows[0]["game"] == "0"
    assert {"rank_p0", "nodes_p2", "git_hash", "focus_win"} <= set(rows[0])
    assert rows[0]["team_p7"] == ""
    assert len(list(tmp_path.glob("game_*.jsonl"))) == 3
    assert game_config(cfg, 0, 1) == cfg
    single = run_arena(cfg, games=1)
    assert single[0].seed == 11
