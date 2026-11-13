"""Tests des campagnes comparatives et des analyses enrichies (P06)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tron.analysis import analyze, mcnemar_exact, paired_comparisons
from tron.analysis.regression import default_formula, fit_logit, prepare
from tron.analysis.stats import wilcoxon_p
from tron.cli import main
from tron.core import ConfigError, GameConfig
from tron.experiments import Plan, ResultStore, build_config, run_plan

PLANS = Path(__file__).resolve().parents[1] / "configs" / "plans"
NEW_PLANS = (
    "heuristic_comparison",
    "mcts_comparison",
    "mcts_simulations",
    "dynamic_alliances",
    "adaptive_agents",
    "ablation",
)


@pytest.mark.parametrize("name", NEW_PLANS)
def test_new_plans_load_pair_seeds_and_rotate_seats(name: str) -> None:
    plan = Plan.from_yaml(PLANS / f"{name}.yaml")
    conds = plan.conditions()
    assert conds and plan.pilot
    for cond in conds:
        cfg = GameConfig.from_dict({**cond.config, "seed": 1, "first_player": 0})
        assert cfg.n_players == cond.n_players
    pilot = plan.pilot_plan()
    assert pilot.name.endswith("_pilot") and pilot.games <= 4
    tasks = pilot.tasks()
    by_cond: dict[str, list] = {}
    for t in tasks:
        by_cond.setdefault(t.condition, []).append(t)
    groups: dict[int, list[list]] = {}
    for ts in by_cond.values():
        n = GameConfig.from_dict({**ts[0].config, "seed": 0, "first_player": 0}).n_players
        groups.setdefault(n, []).append(ts)
        assert [t.first_player for t in ts] == [g % n for g in range(len(ts))]
        for g, t in enumerate(ts):
            assert t.seed == ts[g - g % n].seed  # une graine par tour complet de sièges
    for same_n in groups.values():  # graines appariées entre conditions comparables
        seeds = [[t.seed for t in ts] for ts in same_n]
        assert all(s == seeds[0] for s in seeds)


def test_plan_specific_factors_reach_the_right_players() -> None:
    ablation = Plan.from_yaml(PLANS / "ablation.yaml").conditions()
    safety = next(c for c in ablation if c.factors["ablate"] == "safety")
    solo = safety.config["players"][0]
    assert solo["heuristic"] == {"name": "composite", "safety": 0.0}
    assert "heuristic" not in safety.config["players"][1]
    dipl = Plan.from_yaml(PLANS / "dynamic_alliances.yaml").conditions()
    assert len(dipl) == 20
    off = [c for c in dipl if not c.factors["diplomacy_enabled"]]
    assert len(off) == 2 and all(c.config["diplomacy"]["enabled"] is False for c in off)
    assert all(p["profile"] == "dynamic_team" for p in dipl[0].config["players"])
    ada = Plan.from_yaml(PLANS / "adaptive_agents.yaml").conditions()
    adaptive = next(c for c in ada if c.factors["solo_agent"] == "adaptive")
    assert adaptive.config["players"][0]["adaptation_rate"] in (0.25, 0.5)
    social = next(c for c in ada if c.factors["solo_agent"] == "social")
    assert "adaptation_rate" not in social.config["players"][0]
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"ablate": "vitesse", "solo_heuristic": "composite"})
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"ablate": "safety"})  # pas d'heuristique composite
    with pytest.raises(ConfigError):
        build_config("solo_vs_coalition", {"agent_solo": "mcts", "solo_agent": "maxn"})
    with pytest.raises(ConfigError):
        Plan.from_dict({"name": "x", "scenario": "teams", "factors": {"k": [1]}, "pilot": {"n": 2}})
    with pytest.raises(ConfigError):
        Plan.from_yaml(PLANS / "ablation.yaml").pilot_plan(0)


TINY = {
    "name": "mini_p06",
    "scenario": "solo_vs_coalition",
    "games": 4,
    "seed": 11,
    "base": {
        "solo_agent": "paranoid",
        "solo_depth": 2,
        "solo_heuristic": "composite",
        "coalition_mode": "greedy",
        "placement": {"mode": "regular", "rotate": True},
    },
    "factors": {"ablate": ["none", "safety", "mobility"], "k": [1], "grid": [7]},
}


def test_run_resume_manifest_and_analysis(tmp_path: Path) -> None:
    plan = Plan.from_dict(TINY)
    out = tmp_path / "res"
    first = run_plan(plan, out, workers=1, progress=False, limit=5)
    second = run_plan(plan, out, workers=1, progress=False)
    assert (first.played, second.played, second.skipped) == (5, 7, 5)
    rows = ResultStore(out / "results.csv").rows()
    assert len(rows) == 12 and len({(r["condition"], r["game"]) for r in rows}) == 12
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["factors"] == {"ablate": ["none", "safety", "mobility"], "k": [1], "grid": [7]}
    assert manifest["plugins"] == []  # extensions chargées (P12) : aucune ici
    assert {"code", "log_format", "metrics"} <= set(manifest["versions"])
    assert manifest["conditions"] == 3 and manifest["limitations"] == []
    res = analyze(out / "results.csv", out / "an")
    assert res.paired is not None
    paired = pd.read_csv(res.paired)
    abl = paired[paired["factor"] == "f_ablate"]
    assert set(abl["low"]) == {"none"} and set(abl["high"]) == {"safety", "mobility"}
    assert (abl["pairs"] == 4).all() and {"time_diff", "nodes_diff", "p_holm"} <= set(abl)
    assert any(p.name == "winrate_ablate.png" for p in res.figures)
    assert res.regression is not None
    assert "ablate__safety" in res.regression.read_text(encoding="utf-8")


def test_diplomacy_plans_record_their_limitations(tmp_path: Path) -> None:
    plan = Plan.from_dict(
        {
            "name": "dipl_mini",
            "scenario": "teams",
            "games": 1,
            "seed": 3,
            "base": {"mode": "dynamic_team", "depth": 1},
            "factors": {"diplomacy_enabled": [False, True], "composition": ["1v1v1"], "grid": [7]},
        }
    )
    run_plan(plan, tmp_path, workers=1, progress=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert any("constante pendant une recherche" in n for n in manifest["limitations"])
    res = analyze(tmp_path / "results.csv", tmp_path / "an")
    assert res.diplomacy is not None and res.diplomacy.exists()


def test_paired_statistics_on_a_known_dataset() -> None:
    rows = []
    # 10 graines : « b » gagne là où « a » perd 6 fois, l'inverse 1 fois, 3 égalités.
    outcomes = [(0, 1)] * 6 + [(1, 0)] + [(1, 1)] * 3
    for seed, (wa, wb) in enumerate(outcomes):
        for level, win, cost in (("a", wa, 10.0), ("b", wb, 10.0 + seed + 1)):
            rows.append(
                {
                    "f_variant": level,
                    "seed": seed,
                    "first_player": 0,
                    "focus_win": win,
                    "focus_time_per_decision": 0.01,
                    "focus_nodes_per_decision": cost,
                }
            )
    df = pd.DataFrame(rows)
    res = paired_comparisons(df, "f_variant", reference="a")
    row = res.iloc[0]
    assert (row["low"], row["high"], row["pairs"]) == ("a", "b", 10)
    assert (row["high_only"], row["low_only"]) == (6, 1)
    assert row["p_value"] == pytest.approx(mcnemar_exact(6, 1))
    assert row["p_value"] == pytest.approx(2 * (1 + 7) / 2**7)
    assert row["nodes_diff"] == pytest.approx(5.5)
    assert row["time_diff"] == 0.0 and row["time_p_value"] == 1.0
    from scipy.stats import wilcoxon

    assert row["nodes_p_value"] == pytest.approx(wilcoxon(list(range(1, 11))).pvalue)
    assert wilcoxon_p([0.0, 0.0]) == 1.0 and wilcoxon_p([math.nan]) == 1.0
    assert paired_comparisons(df, "f_variant", reference="absent").empty
    consecutive = paired_comparisons(df, "f_variant")
    assert list(consecutive["high"]) == ["b"]


def test_regression_codes_new_factors() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(400):
        for heur in ("voronoi", "composite"):
            for dipl in (False, True):
                logit = -0.2 + 0.9 * (heur == "composite") - 0.6 * dipl
                rows.append(
                    {
                        "f_solo_heuristic": heur,
                        "f_diplomacy_enabled": dipl,
                        "seed": seed,
                        "focus_win": int(rng.random() < 1 / (1 + math.exp(-logit))),
                    }
                )
    df = pd.DataFrame(rows)
    data = prepare(df)
    assert set(data["diplomacy_enabled"]) == {0, 1}
    assert "solo_heuristic__composite" in data and "solo_heuristic__voronoi" not in data
    formula = default_formula(df)
    assert formula == "focus_win ~ diplomacy_enabled + solo_heuristic__composite"
    table = fit_logit(df, formula, force_numpy=True).table
    coef = dict(zip(table["term"], table["coef"], strict=True))
    assert coef["solo_heuristic__composite"] == pytest.approx(0.9, abs=0.3)
    assert coef["diplomacy_enabled"] == pytest.approx(-0.6, abs=0.3)


def test_cli_pilot_and_games_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({**TINY, "pilot": {"games": 2}}), encoding="utf-8")
    assert main(["run", str(path), "--dry-run", "--pilot"]) == 0
    assert "mini_p06_pilot : 3 conditions, 6 parties" in capsys.readouterr().out
    assert main(["run", str(path), "--dry-run", "--games", "1"]) == 0
    assert "3 conditions, 3 parties" in capsys.readouterr().out
