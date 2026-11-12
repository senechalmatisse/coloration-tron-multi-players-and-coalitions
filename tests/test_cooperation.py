"""Tests des métriques de coopération, de cohésion et de stabilité (P02)."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from conftest import make_config
from tron.analysis import analyze, condition_summary, cooperation_summary, has_metrics, team_table
from tron.core import PASS, GameState
from tron.experiments import Plan, ResultStore, run_plan
from tron.schema import METRICS_VERSION, TEAM_METRIC_COLUMNS
from tron.sim import GameLog, Match, MetricsCollector, Recorder, metrics_from_moves

SRC = Path(__file__).resolve().parents[1] / "src"


def _team_config():
    return make_config(
        10,
        10,
        4,
        teams=["A", "B", "A", "B"],
        agents=["paranoid", "maxn", "social", "greedy"],
        depth=[2, 2, 2, 1],
        params=[{}, {}, {"profile": "team", "alpha": 1.0}, {}],
        seed=3,
        placement={"mode": "regular", "rotate": True},
    )


# ------------------------------------------------------------ cas construits
def test_two_solo_players_have_neutral_team_metrics() -> None:
    cfg = make_config(5, 5, 2, placement={"positions": [[0, 0], [4, 4]]})
    # J0 : E, J1 : N, J0 : PASS (abandon alors qu'il avait un coup).
    m = metrics_from_moves(cfg, [(0, 0), (4, 4)], [(0, 1), (1, 0), (0, PASS)])
    assert m.version == METRICS_VERSION
    assert m.survival_plies == (2, 3) and m.team_survival_plies == (2, 3)
    assert m.team_rank == (2, 1)
    assert m.intra_team_distance_mean == (None, None)
    assert m.cooperation_proxy == (None, None)
    assert m.collective_vs_individual_gap == (None, None)
    assert m.ally_collisions == (0, 0)  # abandon avec un coup légal : cause indéterminée
    assert m.team_moves == (1, 1)


def _trap_state(blocker_row: str) -> GameState:
    cfg = make_config(5, 5, 3, teams=["A", "A", "B"])
    return GameState.from_text(
        cfg,
        f"""
        BaaA.
        {blocker_row}
        .....
        .....
        ....C
        """,
        current=1,
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("aaa..", 1),  # enfermé uniquement par des murs du coéquipier J0
        ("caa..", 0),  # un mur adverse participe : cause mixte, non comptée
        ("baa..", 1),  # son propre mur ne compte pas comme cause
    ],
)
def test_ally_trap_detection(row: str, expected: int) -> None:
    st = _trap_state(row)
    col = MetricsCollector(st.config, territory_every=0)
    col.start(st)
    origin = st.head(1)
    assert st.legal_moves() == []
    st.apply(PASS)
    col.record(st, 1, origin, PASS, None, True, 0)
    m = col.finish(st)
    assert m.ally_collisions == (expected, 0)


def test_direct_collision_with_teammate_wall() -> None:
    st = _trap_state("aaa..")
    col = MetricsCollector(st.config, territory_every=0)
    origin = st.head(1)
    st.apply(1)  # vers l'est : mur de J0
    col.record(st, 1, origin, 1, None, True, 0)
    assert col.finish(st).ally_collisions == (1, 0)


def test_blocking_moves_distances_and_gap() -> None:
    cfg = make_config(
        7, 5, 3, teams=["A", "A", "B"], placement={"positions": [[0, 2], [3, 2], [6, 0]]}
    )
    positions = [(0, 2), (3, 2), (6, 0)]
    # J0 : E (1,2) ; J1 : O (2,2), voisine de la tête de J0 -> blocage ; J2 : S ;
    # J0 : N (1,1) ; J1 : N (2,1), voisine de la tête de J0 -> blocage ; J2 : O.
    moves = [(0, 1), (1, 3), (2, 2), (0, 0), (1, 0), (2, 3)]
    m = metrics_from_moves(cfg, positions, moves, territory_every=0)
    assert m.team_moves == (4, 2)
    assert m.blocking_moves == (2, 0)
    st = GameState(cfg, positions=positions)
    dists = [3]
    for _p, d in moves:
        st.apply(d)
        dists.append(abs(st.head(0)[0] - st.head(1)[0]) + abs(st.head(0)[1] - st.head(1)[1]))
    assert m.intra_team_distance_mean[0] == pytest.approx(sum(dists) / len(dists))
    assert m.intra_team_distance_mean[1] is None
    proxy = m.cooperation_proxy[0]
    d = m.intra_team_distance_mean[0]
    assert d is not None
    assert proxy == pytest.approx(0.5 * (1 - d / (7 + 5 - 2)) + 0.5 * (1 - 2 / 4))
    assert m.territory_observations == 0 and m.territory_share_mean == (None, None)
    assert m.collective_vs_individual_gap[0] == 0.0  # personne n'est éliminé


def test_collector_validation() -> None:
    with pytest.raises(ValueError):
        MetricsCollector(make_config(5, 5, 2), territory_every=-1)


# ---------------------------------------------------------------- parties
def test_match_metrics_are_deterministic_and_exported() -> None:
    a = Match(_team_config()).run()
    b = Match(_team_config()).run()
    assert a.metrics is not None and a.metrics == b.metrics
    m = a.metrics
    assert m.teams == ("A", "B") and m.team_sizes == (2, 2)
    assert m.territory_observations > 0
    assert sum(t for t in m.territory_share_mean if t is not None) == pytest.approx(1.0, abs=0.02)
    winner = m.teams.index(a.winning_team)
    assert m.team_rank[winner] == 1 and m.team_survival_plies[winner] == a.n_plies
    row = a.to_row()
    assert row["metrics_version"] == METRICS_VERSION
    assert row["team_t0"] == "A" and row["team_t2"] == ""
    assert {f"{prefix}_t0" for prefix, _ in TEAM_METRIC_COLUMNS} <= set(row)
    off = Match(_team_config(), metrics=False).run()
    assert off.metrics is None
    off_row = off.to_row()
    assert list(off_row) == list(row)  # mêmes colonnes, valeurs vides
    assert off_row["metrics_version"] == "" and off_row["tcoop_t0"] == ""
    assert (off.winning_team, off.ranks, off.n_moves) == (a.winning_team, a.ranks, a.n_moves)


def test_metrics_recomputed_from_log(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    result = Match(_team_config(), observers=[Recorder(path)]).run()
    log = GameLog.load(path)
    assert log.metrics() == result.metrics
    assert log.result is not None and log.result["metrics"]["teams"] == ["A", "B"]


# ---------------------------------------------------------------- analyse
TINY = {
    "name": "coop",
    "scenario": "solo_vs_coalition",
    "games": 4,
    "seed": 5,
    "base": {"coalition_mode": "team", "solo_depth": 1, "coalition_depth": 1},
    "factors": {"k": [1, 2, 3], "grid": [7]},
}


def test_campaign_analysis_produces_cooperation_outputs(tmp_path: Path) -> None:
    run_plan(Plan.from_dict(TINY), tmp_path, workers=1, progress=False)
    df = pd.read_csv(tmp_path / "results.csv")
    assert has_metrics(df)
    table = team_table(df)
    assert len(table) == 2 * len(df)  # deux équipes par partie
    assert set(table["team"]) == {"solo", "coalition"}
    solo = table[table["team"] == "solo"]
    assert solo["cooperation_proxy"].isna().all() and (solo["team_size"] == 1).all()
    coal = table[(table["team"] == "coalition") & (table["team_size"] >= 2)]
    assert coal["cooperation_proxy"].notna().all()
    summary = cooperation_summary(df)
    assert len(summary) == 3 * 2 and "cooperation_proxy_mean" in summary
    assert (summary["n"] == 4).all()
    cond = condition_summary(df)
    assert set(cond["f_k"]) == {2, 3}  # k = 1 : pas d'équipe de deux joueurs
    res = analyze(tmp_path / "results.csv", tmp_path / "an")
    assert res.cooperation is not None and res.cooperation.exists()
    assert res.cooperation_conditions is not None
    assert any("cooperation_proxy_k" in p.name for p in res.figures)
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert f'"metrics_version": {METRICS_VERSION}' in manifest


def test_old_csv_without_metrics_is_still_analysed(tmp_path: Path) -> None:
    run_plan(Plan.from_dict(TINY), tmp_path / "run", workers=1, progress=False)
    df = pd.read_csv(tmp_path / "run" / "results.csv")
    first_metric = list(df.columns).index("metrics_version")
    old = df.iloc[:, :first_metric]
    old_path = tmp_path / "old.csv"
    old.to_csv(old_path, index=False)
    assert not has_metrics(old) and team_table(old).empty and cooperation_summary(old).empty
    res = analyze(old_path, tmp_path / "an_old")
    assert res.summary.exists() and res.cooperation is None


def test_resume_of_an_older_campaign_keeps_its_columns(tmp_path: Path) -> None:
    plan = Plan.from_dict(TINY)
    out = tmp_path / "res"
    run_plan(plan, out, workers=1, progress=False, limit=3)
    path = out / "results.csv"
    df = pd.read_csv(path)
    first_metric = list(df.columns).index("metrics_version")
    df.iloc[:, :first_metric].to_csv(path, index=False)  # simule un CSV d'avant P02
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = run_plan(plan, out, workers=1, progress=False)
    assert summary.played == 12 - 3
    assert any("version antérieure" in str(w.message) for w in caught)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and "metrics_version" not in rows[0]
    store = ResultStore(path)
    with pytest.raises(ValueError):
        store.append({"autre": 1})


def test_analysis_module_does_not_import_engine_numba_or_arcade() -> None:
    code = (
        "import sys, tron.analysis.cooperation; "
        "print(any(m in sys.modules for m in ('numba', 'arcade', 'tron.sim', 'tron.agents')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert out.stdout.strip() == "False"
