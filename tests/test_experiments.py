"""Tests des plans d'expériences, des scénarios et de l'exécution (F5.3 à F5.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tron.core import ConfigError, GameConfig
from tron.experiments import (
    Plan,
    ResultStore,
    build_config,
    estimate,
    parse_composition,
    run_plan,
)
from tron.experiments.scenarios import set_path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

TINY = {
    "name": "mini",
    "scenario": "solo_vs_coalition",
    "games": 4,
    "seed": 3,
    "base": {"coalition_mode": "team", "placement": {"mode": "regular", "rotate": True}},
    "factors": {"k": [1, 2], "solo_depth": [1, 2], "coalition_depth": [1], "grid": [7]},
}


def test_bundled_plans_and_configs_are_valid() -> None:
    a = Plan.from_yaml(CONFIGS / "plans" / "etude_a.yaml")
    conds = a.conditions()
    assert len(conds) == 6 * 4 * 5 * 3
    assert sum(c.games for c in conds) == 30 * 200 + 330 * 100
    b = Plan.from_yaml(CONFIGS / "plans" / "etude_b.yaml")
    assert len(b.conditions()) == 5 * 2 * 4 * 4 * 3
    assert len(Plan.from_yaml(CONFIGS / "plans" / "etude_a_cooperation.yaml").conditions()) == 96
    assert len(Plan.from_yaml(CONFIGS / "plans" / "pilote.yaml").tasks()) == 48 * 24
    for path in (CONFIGS / "games").glob("*.yaml"):
        GameConfig.from_yaml(path)


def test_paired_seeds_and_seat_rotation() -> None:
    plan = Plan.from_dict(TINY)
    tasks = plan.tasks()
    assert len(tasks) == 4 * 4
    by_cond: dict[str, list] = {}
    for t in tasks:
        by_cond.setdefault(t.condition, []).append(t)
    two_players = [ts for ts in by_cond.values() if ts[0].factors["k"] == 1]
    assert [t.seed for t in two_players[0]] == [t.seed for t in two_players[1]]
    ts = two_players[0]
    assert [t.first_player for t in ts] == [0, 1, 0, 1]
    assert ts[0].seed == ts[1].seed != ts[2].seed
    per_game = Plan.from_dict({**TINY, "seats": "per_game"}).tasks()
    assert len({t.seed for t in per_game if t.cond_index == 0}) == 4
    cfg = ts[1].game_config()
    assert cfg.first_player == 1 and cfg.seed == ts[1].seed


def test_designs_and_filters() -> None:
    sub = Plan.from_dict({**TINY, "design": {"type": "subset", "n": 3, "seed": 1}})
    assert len(sub.conditions()) == 3
    lst = Plan.from_dict(
        {
            **TINY,
            "design": {
                "type": "list",
                "conditions": [{"k": 3, "solo_depth": 2, "coalition_depth": 1, "grid": 8}],
            },
        }
    )
    assert [c.n_players for c in lst.conditions()] == [4]
    exc = Plan.from_dict({**TINY, "exclude": [{"k": 2}]})
    assert {c.factors["k"] for c in exc.conditions()} == {1}
    key = Plan.from_dict({**TINY, "key_conditions": {"games": 6, "where": [{"k": 2}]}})
    assert sorted(c.games for c in key.conditions()) == [4, 4, 6, 6]
    with pytest.raises(ConfigError):
        Plan.from_dict({**TINY, "design": "magique"}).conditions()
    with pytest.raises(ConfigError):
        Plan.from_dict({**TINY, "inconnu": 1})
    with pytest.raises(ConfigError):
        Plan.from_dict({"name": "x"})
    with pytest.raises(ConfigError):
        Plan.from_dict({**TINY, "factors": {"k": 3}})
    with pytest.raises(ConfigError):
        Plan.from_dict({**TINY, "seats": "bizarre"})


def test_scenarios() -> None:
    cfg = GameConfig.from_dict(
        build_config(
            "solo_vs_coalition",
            {
                "k": 3,
                "solo_depth": 6,
                "coalition_depth": 2,
                "grid": 12,
                "coalition_mode": "maxn",
                "depth_unit": "rounds",
            },
        )
    )
    assert cfg.n_players == 4 and cfg.team_names == ("solo", "coalition")
    assert cfg.players[1].agent == "maxn" and cfg.players[1].params["rounds"] == 2
    teams = GameConfig.from_dict(
        build_config(
            "teams", {"composition": "1v2v3", "depth_t0": 5, "mode_t1": "maxn", "grid": [12, 10]}
        )
    )
    assert [p.team for p in teams.players] == ["T0", "T1", "T2", "T1", "T2", "T2"]
    assert teams.players[0].depth == 5 and teams.players[1].agent == "maxn"
    assert (teams.width, teams.height) == (12, 10)
    assert parse_composition("2c4") == [2, 4]
    for bad in ("5", "0v2", "5v5", "axb"):
        with pytest.raises(ConfigError):
            parse_composition(bad)
    base = {"grid": 10, "players": [{"name": "a"}, {"name": "b"}]}
    custom = build_config("custom", {"config": base, "players.1.depth": 3, "seed": 4})
    assert custom["players"][1]["depth"] == 3 and base["players"][1] == {"name": "b"}
    data: dict = {"a": [{"b": 1}]}
    set_path(data, "a.0.b", 2)
    assert data == {"a": [{"b": 2}]}
    with pytest.raises(ConfigError):
        build_config("inconnu", {})
    with pytest.raises(ConfigError):
        build_config("custom", {})
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"k": 9})
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"coalition_mode": "anarchie"})
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"depth_unit": "secondes"})


def test_run_plan_resume_and_determinism(tmp_path) -> None:
    plan = Plan.from_dict(TINY)
    out_seq = tmp_path / "seq"
    first = run_plan(plan, out_seq, workers=1, limit=5, progress=False)
    assert (first.played, first.skipped) == (5, 0)
    second = run_plan(plan, out_seq, workers=1, progress=False)
    assert (second.played, second.skipped) == (11, 5)
    rows = ResultStore(out_seq / "results.csv").rows()
    assert len(rows) == 16
    assert len({(r["condition"], r["game"]) for r in rows}) == 16
    assert {"f_k", "f_solo_depth", "focus_win", "git_hash"} <= set(rows[0])
    assert (out_seq / "manifest.json").exists()
    out_par = tmp_path / "par"
    run_plan(plan, out_par, workers=2, progress=False, logs=True)
    par = ResultStore(out_par / "results.csv").rows()
    keys = ("condition", "game", "seed", "winning_team", "n_moves", "rank_p0", "nodes_p1")

    def norm(rs: list[dict[str, str]]) -> list[tuple[str, ...]]:
        return sorted(tuple(r[k] for k in keys) for r in rs)

    assert norm(par) == norm(rows)
    assert len(list((out_par / "logs").glob("*.jsonl"))) == 16
    third = run_plan(plan, out_par, workers=2, progress=False)
    assert third.played == 0
    fresh = run_plan(plan, out_par, workers=1, resume=False, limit=2, progress=False)
    assert fresh.played == 2 and len(ResultStore(out_par / "results.csv").rows()) == 2


def test_store_repairs_partial_line_and_checks_columns(tmp_path) -> None:
    path = tmp_path / "r.csv"
    path.write_text("condition,game,x\nc,0,1\nc,1,", encoding="utf-8")
    store = ResultStore(path)
    assert store.completed() == {("c", 0)}
    store.append({"condition": "c", "game": 1, "x": 2})
    store.close()
    assert path.read_text(encoding="utf-8").splitlines() == ["condition,game,x", "c,0,1", "c,1,2"]
    with pytest.raises(ValueError):
        ResultStore(path).append({"autre": 1})
    assert ResultStore(tmp_path / "absent.csv").rows() == []


def test_estimate() -> None:
    est = estimate(Plan.from_dict(TINY), samples=2, workers=2)
    assert est["tasks"] == 16 and est["sampled"] == 2
    assert est["wall_hours"] == pytest.approx(est["cpu_hours"] / 2)
