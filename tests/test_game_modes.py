"""Tests des modes de jeu (P09)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from conftest import make_config
from tron.core import PASS, ConfigError, GameConfig, GameState
from tron.eval import Valuation, VoronoiHeuristic
from tron.sim import Match

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(
    n: int = 3, teams: list[str] | None = None, rules: object = None, **kw: object
) -> GameConfig:
    data = make_config(9, 9, n, teams=teams, **kw).to_dict()  # type: ignore[arg-type]
    if rules is not None:
        data["rules"] = rules
    return GameConfig.from_dict(data)


# ----------------------------------------------------------------- survie
@pytest.mark.parametrize(
    "rules",
    [
        {"mode": "inconnu"},
        {"mode": "survival", "victory_scope": "monde"},
        {"mode": "classic", "victory_scope": "individual"},
        {"mode": "survival", "bizarre": 1},
        42,
    ],
)
def test_rules_validation(rules: object) -> None:
    with pytest.raises(ConfigError):
        _cfg(rules=rules)


def test_rules_roundtrip_and_defaults() -> None:
    classic = _cfg()
    assert "rules" not in classic.to_dict() and classic.rules.mode == "classic"
    assert classic.victory_team_index == classic.team_index
    surv = _cfg(teams=["A", "A", "B"], rules={"mode": "survival", "victory_scope": "individual"})
    assert GameConfig.from_dict(surv.to_dict()) == surv
    assert surv.victory_team_index == (0, 1, 2) and surv.team_index == (0, 0, 1)
    assert surv.victory_names == ("J0", "J1", "J2")
    assert GameConfig.from_dict({**classic.to_dict(), "rules": "survival"}).rules.mode == "survival"


def test_individual_survival_continues_among_teammates() -> None:
    teams = ["A", "A", "B"]
    team_cfg = _cfg(teams=teams, rules={"mode": "survival", "victory_scope": "team"})
    ind_cfg = _cfg(teams=teams, rules={"mode": "survival", "victory_scope": "individual"})
    for cfg, expected_terminal in ((team_cfg, True), (ind_cfg, False)):
        st = GameState(cfg)
        st.apply(st.legal_moves()[0])  # J0
        st.apply(st.legal_moves()[0])  # J1
        st.apply(PASS)  # J2 (équipe B) éliminé
        assert st.is_terminal() is expected_terminal
    st = GameState(ind_cfg)
    st.apply(PASS)  # J0 éliminé : J1 (A) et J2 (B) restent
    assert not st.is_terminal()
    st.apply(st.legal_moves()[0])
    st.apply(PASS)  # J2 éliminé : J1 seul survivant
    out = st.outcome()
    assert out is not None and out.winning_team == 1 and out.winners == (1,)
    st.undo()
    assert not st.is_terminal() and st.outcome() is None
    st.apply(PASS)
    assert st.is_terminal()


def test_individual_valuation_rewards_only_the_survivor() -> None:
    cfg = _cfg(teams=["A", "A", "B"], rules={"mode": "survival", "victory_scope": "individual"})
    val = Valuation(cfg, VoronoiHeuristic(9, 9, 3))
    st = GameState(cfg)
    st.apply(PASS)
    st.apply(st.legal_moves()[0])
    st.apply(PASS)
    v = val.vector(st, 0)
    assert v[1] > 1.0 and v[0] < val.eps and v[2] < val.eps
    assert sum(v) <= val.sum_bound + 1e-12
    assert val.pair_bound(0, 1) == val.pair_bound_diff  # coéquipiers, mais victoire individuelle


def test_survival_results_and_classic_compatibility() -> None:
    cfg = GameConfig.from_yaml(CONFIGS / "games" / "survival_individual_12x12.yaml")
    result = Match(cfg).run()
    assert result.mode == "survival" and result.individual_winner is not None
    winner = result.individual_winner
    assert result.ranks[winner] == 1 and result.ranks.count(1) == 1
    assert result.winning_team == cfg.players[winner].team
    assert result.survival_order[0] == winner and sorted(result.survival_order) == [0, 1, 2, 3]
    row = result.to_row()
    assert row["rules_mode"] == "survival" and row["individual_winner"] == winner
    classic = make_config(9, 9, 3, seed=4, placement={"mode": "regular", "rotate": True})
    r = Match(classic).run()
    assert r.mode == "classic" and r.to_row()["rules_mode"] == "classic"
    surv_team = GameConfig.from_dict({**classic.to_dict(), "rules": {"mode": "survival"}})
    r2 = Match(surv_team).run()  # même règle de fin : même partie
    assert (r2.ranks, r2.n_plies, r2.winning_team) == (r.ranks, r.n_plies, r.winning_team)


def test_apply_undo_keeps_terminal_flag_consistent() -> None:
    rng = random.Random(2)
    cfg = _cfg(
        4, teams=["A", "B", "A", "B"], rules={"mode": "survival", "victory_scope": "individual"}
    )
    st = GameState(cfg)
    flags = []
    while not st.is_terminal():
        flags.append((st.over, st.teams_alive, st.key))
        moves = st.legal_moves()
        st.apply(rng.choice(moves) if moves and rng.random() < 0.85 else PASS)
        assert st.over == (st.teams_alive <= 1)
    for snap in reversed(flags):
        st.undo()
        assert (st.over, st.teams_alive, st.key) == snap
