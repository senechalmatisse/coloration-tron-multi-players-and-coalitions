"""API Python publique."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tron
from tron import api
from tron.api import (
    ConfigError,
    GameConfig,
    LogError,
    Match,
    MatchResult,
    create_match,
    load_game_config,
    load_replay,
    resume_game,
    run_game,
    run_games,
)

ROOT = Path(__file__).resolve().parents[1]
DUEL = ROOT / "configs" / "games" / "learning_duel_10x10.yaml"
_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
_TIMING = ("duration", "time_p")


def _small() -> dict[str, Any]:
    return {
        "name": "api_small",
        "grid": 9,
        "seed": 4,
        "placement": {"mode": "regular", "rotate": True},
        "players": [
            {"name": "a", "team": "A", "agent": "greedy"},
            {"name": "b", "team": "B", "agent": "paranoid", "depth": 2},
            {"name": "c", "team": "A", "agent": "random"},
        ],
    }


def _row(result: MatchResult) -> dict[str, Any]:
    return {k: v for k, v in result.to_row().items() if not k.startswith(_TIMING)}


def test_public_names_and_lazy_package_exports() -> None:
    for name in api.__all__:
        assert getattr(api, name) is getattr(tron, name)
    assert api.API_VERSION == "1.0" and isinstance(tron.__version__, str)
    from tron.sim import GameLog as InternalLog
    from tron.sim import Match as InternalMatch

    assert api.Match is InternalMatch and api.GameLog is InternalLog  # mêmes classes
    for exc in (api.ConfigError, api.LogError, api.CheckpointError, api.ModelError):
        assert issubclass(exc, ValueError)
    with pytest.raises(AttributeError):
        _ = tron.not_a_name  # type: ignore[attr-defined]
    code = "import sys, tron; print('tron.api' in sys.modules, 'tron.sim' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=_ENV
    )
    assert out.stdout.strip() == "False False"  # import tron reste léger


def test_load_game_config_sources_and_roundtrip(tmp_path: Path) -> None:
    cfg = load_game_config(DUEL)
    assert load_game_config(cfg) is cfg
    assert load_game_config(cfg.to_dict()) == cfg
    as_json = tmp_path / "duel.json"
    as_json.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert load_game_config(as_json) == cfg
    assert load_game_config(str(DUEL)) == cfg
    with pytest.raises(ConfigError, match="introuvable"):
        load_game_config(tmp_path / "absent.yaml")
    with pytest.raises(ConfigError):
        load_game_config(42)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        load_game_config({"grid": 9, "players": []})
    with pytest.raises(AttributeError):
        cfg.seed = 3  # type: ignore[misc]  # configuration immuable


def test_run_game_without_gui_matches_internal_classes(tmp_path: Path) -> None:
    log = tmp_path / "g.jsonl"
    result = run_game(_small(), seed=9, log=log)
    internal = Match(GameConfig.from_dict(_small()).with_seed(9)).run()
    assert _row(result) == _row(internal)
    replay = load_replay(log)
    final = replay.final_state()
    assert final.is_terminal() and final.n_moves == result.n_moves
    assert sum(1 for _ in replay.states()) == len(replay.moves) + 1
    with pytest.raises(LogError, match="introuvable"):
        load_replay(tmp_path / "absent.jsonl")


def test_create_match_is_single_use_and_validates_agents() -> None:
    match = create_match(_small(), seed=2, first_player=1, metrics=False)
    assert match.config.seed == 2 and match.config.first_player == 1
    first = match.run()
    assert match.run() is first  # run() consomme la partie
    with pytest.raises(ConfigError, match="agents"):
        create_match(_small(), agents=[])
    agents = api.build_agents(GameConfig.from_dict(_small()))
    custom = create_match(_small(), agents=agents)
    assert custom.agents[0] is agents[0]


def test_run_games_parallel_equals_sequential(tmp_path: Path) -> None:
    seq = run_games(_small(), 4, workers=1, out_dir=tmp_path / "seq", logs=True)
    par = run_games(_small(), 4, workers=2, out_dir=tmp_path / "par", logs=True)
    assert [_row(r) for r in seq] == [_row(r) for r in par]
    for name in ("seq", "par"):
        files = sorted(p.name for p in (tmp_path / name).iterdir())
        assert files == [f"game_{g:04d}.jsonl" for g in range(4)] + ["results.csv"]
    rows = {}
    for name in ("seq", "par"):
        with (tmp_path / name / "results.csv").open(encoding="utf-8") as fh:
            rows[name] = [
                {k: v for k, v in r.items() if not k.startswith(_TIMING)}
                for r in csv.DictReader(fh)
            ]
    assert rows["seq"] == rows["par"]
    seen: list[int] = []
    run_games(_small(), 2, workers=2, on_result=lambda g, _r: seen.append(g))
    assert sorted(seen) == [0, 1]
    for bad in ({"games": 0}, {"games": 2, "workers": 0}, {"games": True}):
        with pytest.raises(ConfigError):
            run_games(_small(), **bad)  # type: ignore[arg-type]


def test_resume_game_from_checkpoint(tmp_path: Path) -> None:
    match = create_match(_small(), seed=5)
    for _ in range(7):
        match.step()
    saved = match.save_checkpoint(tmp_path / "c.json")
    resumed = resume_game(saved).run()
    assert _row(resumed) == _row(run_game(_small(), seed=5))


@pytest.mark.parametrize(
    ("script", "args", "expected"),
    [
        ("agent_personnalise.py", [], "vainqueur"),
        ("campagne.py", [str(DUEL), "2", "2"], "2 parties"),
        ("partie_unique.py", [str(DUEL)], "learning_duel : vainqueur"),
    ],
)
def test_examples_run(script: str, args: list[str], expected: str) -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=_ENV,
    )
    assert expected in out.stdout


def test_replay_example(tmp_path: Path) -> None:
    run_game(DUEL, log=tmp_path / "d.jsonl")
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "rejeu.py"), str(tmp_path / "d.jsonl")],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=_ENV,
    )
    assert "équipe gagnante" in out.stdout
