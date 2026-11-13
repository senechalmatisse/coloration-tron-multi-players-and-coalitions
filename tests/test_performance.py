"""Mesures de performance reproductibles (P13).

Les tests par défaut vérifient la mécanique des mesures (positions déterministes,
sommes de contrôle, comparaison, profil, CLI) sans aucun seuil de temps. Les
tests de charge, marqués ``load``, sont exclus par défaut : ``pytest -m load``.
"""

from __future__ import annotations

import copy
import json
import pstats
from pathlib import Path

import pytest
import yaml

from tron._jit import backend_name
from tron.agents import make_agent
from tron.benchmarks import (
    BENCH_FORMAT,
    PHASES,
    SuiteParams,
    compare,
    fingerprint,
    is_partitioned,
    position_set,
    profile,
    run_suite,
)
from tron.benchmarks.positions import benchmark_config
from tron.cli.main import main
from tron.core import GameConfig
from tron.sim import Match

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "reference" / "suite-python-20x20-4p.json"


def test_position_sets_are_deterministic_and_match_their_phase() -> None:
    a = position_set(12, 3, per_phase=2, seed=0)
    b = position_set(12, 3, per_phase=2, seed=0)
    assert fingerprint(a) == fingerprint(b) != fingerprint(position_set(12, 3, 2, seed=1))
    assert [p.phase for p in a] == [ph for ph in PHASES for _ in range(2)]
    for pos in a:
        st = pos.state()
        fill = st.n_moves / 144
        assert not st.is_terminal() and st.legal_moves() and st.teams_alive >= 2
        if pos.phase == "opening":
            assert 0.02 <= fill <= 0.09 and all(st.is_alive(p) for p in range(3))
        elif pos.phase == "midgame":
            assert not is_partitioned(st)
        elif pos.phase == "partitioned":
            assert is_partitioned(st)
    with pytest.raises(ValueError):
        position_set(10, 2, per_phase=0)


def test_quick_suite_is_reproducible_in_work_and_results() -> None:
    first = run_suite(SuiteParams.quick_params())
    second = run_suite(SuiteParams.quick_params())
    assert first["format"] == BENCH_FORMAT and first["environment"]["backend"] == backend_name()
    assert first["positions"] == second["positions"] and first["positions"]["count"] == 8
    names = {(m["name"], m["phase"]) for m in first["measures"]}
    assert ("results_write", "all") in names and ("search", "partitioned") in names
    for m1, m2 in zip(first["measures"], second["measures"], strict=True):
        assert (m1["name"], m1["phase"], m1["ops"], m1["checksum"]) == (
            m2["name"],
            m2["phase"],
            m2["ops"],
            m2["checksum"],
        )
        assert m1["seconds"] > 0 and m1["per_op_us"] > 0
    rows = compare(second, first)
    assert all(r["same_result"] and r["comparable"] for r in rows)


def test_compare_flags_result_and_parameter_changes() -> None:
    ref = run_suite(SuiteParams.quick_params(repeats=2, rows=10))
    changed = copy.deepcopy(ref)
    changed["measures"][2]["checksum"] = "0" * 16
    rows = compare(changed, ref)
    assert [r["same_result"] for r in rows].count(False) == 1
    changed["params"]["depth"] = 9
    assert not any(r["comparable"] for r in compare(changed, ref))
    with pytest.raises(ValueError, match="format"):
        compare({"format": "autre"}, ref)


def test_profile_reports_functions() -> None:
    text, rows, stats = profile(lambda: sorted(range(10_000), key=lambda v: -v), limit=5)
    assert "function calls" in text and 0 < len(rows) <= 5
    assert {"function", "file", "line", "ncalls", "tottime", "cumtime"} <= set(rows[0])
    assert isinstance(stats, pstats.Stats)


def test_reference_results_match_the_current_position_set() -> None:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert ref["format"] == BENCH_FORMAT and ref["environment"]["backend"] == "python"
    params = ref["params"]
    positions = position_set(params["grid"], params["players"], params["per_phase"], params["seed"])
    assert fingerprint(positions) == ref["positions"]["fingerprint"]  # positions inchangées
    assert all(r["same_result"] for r in compare(ref, ref))


def test_cli_bench_suite_reference_profile_and_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "suite.json"
    assert main(["bench", "--suite", "--quick", "--format", "json", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["suite"]["format"] == BENCH_FORMAT and out.exists()
    assert main(["bench", "--suite", "--quick", "--reference", str(out)]) == 0
    text = capsys.readouterr().out
    assert "rapport" in text and "identique" in text and "DIFFÉRENT" not in text
    prof = tmp_path / "suite.prof"
    code = main(
        [
            "bench",
            "--suite",
            "--quick",
            "--profile",
            "--top",
            "5",
            "--format",
            "json",
            "--profile-out",
            str(prof),
        ]
    )
    assert code == 0 and len(json.loads(capsys.readouterr().out)["profile"]) == 5
    assert pstats.Stats(str(prof)).total_calls > 0
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "name": "mini",
                "scenario": "solo_vs_coalition",
                "games": 2,
                "seed": 1,
                "base": {"solo_agent": "greedy", "coalition_mode": "team"},
                "factors": {"solo_depth": [1], "coalition_depth": [1], "k": [1], "grid": [7]},
            }
        ),
        encoding="utf-8",
    )
    camp = tmp_path / "camp.json"
    code = main(
        ["bench", "--campaign", str(plan), "--workers", "1", "--limit", "2", "--out", str(camp)]
    )
    assert code == 0 and "RSS max" in capsys.readouterr().out
    row = json.loads(camp.read_text(encoding="utf-8"))["rows"][0]
    assert row["workers"] == 1 and row["games"] == 2 and row["max_rss_mb"] > 0
    legacy = ["bench", "--grid", "8", "--players", "2", "--evals", "20", "--decisions", "2"]
    assert main([*legacy, "--depth", "2"]) == 0  # mode historique inchangé
    assert "évaluations/s" in capsys.readouterr().out


# ----------------------------------------------------------------- charge
@pytest.mark.load
@pytest.mark.parametrize(("grid", "players", "depth"), [(40, 8, 3), (60, 4, 4)])
def test_load_search_on_large_grids(grid: int, players: int, depth: int) -> None:
    for pos in position_set(grid, players, per_phase=1, seed=3):
        st = pos.state()
        key, snap = st.key, st.to_snapshot()
        agent = make_agent("paranoid", st.current, pos.config, depth=depth)
        decision = agent.choose_move(st)
        assert decision.move in st.legal_moves()
        assert st.key == key and st.to_snapshot() == snap


@pytest.mark.load
def test_load_mcts_many_simulations() -> None:
    for pos in position_set(30, 4, per_phase=1, seed=5):
        st = pos.state()
        key = st.key
        agent = make_agent("mcts", st.current, pos.config, simulations=3000)
        assert agent.choose_move(st).move in st.legal_moves() and st.key == key


@pytest.mark.load
def test_load_full_melee_and_parallel_campaign(tmp_path: Path) -> None:
    melee = GameConfig.from_yaml(ROOT / "configs" / "games" / "melee_8_30x30.yaml")
    result = Match(melee).run()
    assert result.n_moves > 0 and sum(r == 1 for r in result.ranks) >= 1
    big = benchmark_config(50, 8, seed=2)
    assert Match(big).run().n_plies > 0
    from tron.benchmarks.runner import bench_campaign

    rows = bench_campaign(ROOT / "configs" / "plans" / "pilote.yaml", (1, 2), limit=24)
    assert [r["games"] for r in rows] == [24, 24]


def test_quick_suite_results_equal_the_stored_reference() -> None:
    """Équivalence fonctionnelle : mêmes positions, mêmes résultats de calcul qu'à la référence."""
    ref = json.loads((REFERENCE.parent / "suite-python-quick.json").read_text(encoding="utf-8"))
    if ref["environment"]["backend"] != backend_name():
        pytest.skip("référence enregistrée avec un autre noyau")
    rows = compare(run_suite(SuiteParams.quick_params()), ref)
    assert rows and all(r["comparable"] for r in rows)
    assert [r["name"] for r in rows if not r["same_result"]] == []
