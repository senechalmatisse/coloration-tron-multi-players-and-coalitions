"""Tests de l'analyse statistique et de la commande ``tron``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tron.analysis import (
    analyze,
    default_formula,
    enrich,
    fit_logit,
    holm,
    mcnemar_exact,
    paired_comparisons,
    summarize,
    wilson_interval,
)
from tron.cli import main

ROOT = Path(__file__).resolve().parents[1]
TINY_PLAN = {
    "name": "mini",
    "scenario": "solo_vs_coalition",
    "games": 4,
    "seed": 3,
    "base": {"coalition_mode": "team"},
    "factors": {"k": [1, 2], "solo_depth": [1, 2], "coalition_depth": [1, 2], "grid": [7]},
}


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_holm_and_mcnemar() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def _synthetic(n_seeds: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(n_seeds):
        for d in (1, 2, 4, 8):
            for k in (1, 2):
                logit = -0.5 + 0.8 * np.log2(d) - 1.2 * (k - 1)
                win = int(rng.random() < 1 / (1 + np.exp(-logit)))
                rows.append(
                    {
                        "f_solo_depth": d,
                        "f_k": k,
                        "f_grid": 10,
                        "seed": seed,
                        "first_player": 0,
                        "focus_win": win,
                        "focus_team": "solo",
                        "team_p0": "solo",
                        "team_p1": "coalition",
                        "team_p2": "coalition",
                        "n_players": k + 1,
                        "rank_p0": 1 if win else 2,
                        "n_moves": 40,
                        "duration": 0.1,
                        "nodes_p0": 100 * d,
                        "decisions_p0": 10,
                        "time_p0": 0.01 * d,
                    }
                )
    return enrich(pd.DataFrame(rows))


def test_summary_paired_and_regression() -> None:
    df = _synthetic()
    assert set(df["chance"].round(3)) == {0.5, 0.333}
    summary = summarize(df)
    assert len(summary) == 8 and summary["n"].eq(300).all()
    assert (summary["ci_low"] <= summary["win_rate"]).all()
    assert summary["nodes_per_decision"].max() == pytest.approx(80)
    paired = paired_comparisons(df, "f_solo_depth")
    assert list(paired["high"]) == [2, 4, 8]
    assert (paired["pairs"] == 600).all() and (paired["p_holm"] >= paired["p_value"]).all()
    formula = default_formula(df)
    assert "log2(solo_depth):k" in formula and "grid" not in formula
    res = fit_logit(df, formula, force_numpy=True)
    coef = dict(zip(res.table["term"], res.table["coef"], strict=True))
    assert coef["log2(solo_depth)"] == pytest.approx(0.8, abs=0.25)
    assert coef["k"] == pytest.approx(-1.2, abs=0.35)
    assert res.converged and res.clusters == 300 and "numpy" in res.to_text()
    with pytest.raises(ValueError):
        fit_logit(df, "focus_win ~ C(k)", force_numpy=True)
    with pytest.raises(ValueError):
        fit_logit(df, "focus_win ~ a * b * c", force_numpy=True)


def test_statsmodels_matches_numpy_fallback() -> None:
    pytest.importorskip("statsmodels")
    df = _synthetic()
    formula = default_formula(df)
    res = fit_logit(df, formula, force_numpy=True)
    res_sm = fit_logit(df, formula)
    assert res_sm.backend == "statsmodels"
    assert np.allclose(res_sm.table["coef"], res.table["coef"], atol=1e-4)
    assert np.allclose(res_sm.table["std_err"], res.table["std_err"], rtol=0.05)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_path = tmp_path / "mini.yaml"
    plan_path.write_text(yaml.safe_dump(TINY_PLAN), encoding="utf-8")
    out = tmp_path / "res"
    assert main(["run", str(plan_path), "--dry-run"]) == 0
    assert "32 parties" in capsys.readouterr().out
    assert main(["run", str(plan_path), "--estimate", "2", "--workers", "1"]) == 0
    assert main(["run", str(plan_path), "--out", str(out), "--workers", "1"]) == 0
    assert main(["analyze", str(out / "results.csv")]) == 0
    text = capsys.readouterr().out
    assert "tableau" in text and "figure" in text
    assert (out / "analysis" / "summary.csv").exists()
    assert list((out / "analysis").glob("heatmap_*.png"))

    game = ROOT / "configs" / "games" / "ffa4_20x20.yaml"
    cfg = yaml.safe_load(game.read_text(encoding="utf-8"))
    cfg["grid"] = 8
    for p in cfg["players"]:
        p["depth"] = 2
    small = tmp_path / "ffa.yaml"
    small.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    arena = tmp_path / "arena"
    assert (
        main(["play", str(small), "--no-gui", "--games", "2", "--out", str(arena), "--seed", "4"])
        == 0
    )
    assert "victoires par équipe" in capsys.readouterr().out
    log = arena / "game_0000.jsonl"
    assert main(["replay", str(log), "--no-gui"]) == 0
    assert (
        main(["replay", str(log), "--gif", str(tmp_path / "r.gif"), "--every", "4", "--cell", "4"])
        == 0
    )
    assert (tmp_path / "r.gif").exists()
    assert (
        main(["play", str(small), "--no-gui", "--gif", str(tmp_path / "p.gif"), "--cell", "4"]) == 0
    )
    assert (
        main(
            [
                "bench",
                "--grid",
                "8",
                "--players",
                "2",
                "--depth",
                "2",
                "--evals",
                "50",
                "--decisions",
                "3",
            ]
        )
        == 0
    )
    assert "évaluations/s" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("grid: 3\nplayers: [a, b]\n", encoding="utf-8")
    assert main(["play", str(bad), "--no-gui"]) == 2
    assert main(["analyze", str(tmp_path / "absent.csv")]) == 2


def test_analyze_teams_results(tmp_path: Path) -> None:
    plan = {
        "name": "equipes",
        "scenario": "teams",
        "games": 2,
        "seed": 1,
        "base": {"depth": 1},
        "factors": {"composition": ["1v1", "1v2"], "depth_t0": [1, 2], "grid": [7]},
    }
    from tron.experiments import Plan, run_plan

    run_plan(Plan.from_dict(plan), tmp_path, workers=1, progress=False)
    res = analyze(tmp_path / "results.csv", tmp_path / "an")
    assert res.paired is not None and res.figures
    assert res.regression is None
