"""Tests de la diplomatie tactique."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import make_config
from tron.agents import (
    AGENTS,
    DecisionContext,
    DiplomacyIntent,
    DiplomacySnapshot,
    SocialAgent,
    dynamic_matrix,
    make_agent,
    social_matrix,
    tactical_intent,
)
from tron.core import ConfigError, DiplomacySpec, GameConfig, GameState
from tron.sim import (
    AllianceEvent,
    DiplomacyError,
    DiplomacyState,
    GameLog,
    LogError,
    Match,
    Recorder,
    summarize_diplomacy,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _state(n: int = 4, **spec: Any) -> DiplomacyState:
    return DiplomacyState(DiplomacySpec(enabled=True, **spec), n)


# gestionnaire
def test_unilateral_proposal_stays_pending() -> None:
    d = _state()
    events = d.propose(0, 1, ply=0)
    assert [e.kind for e in events] == ["proposed"]
    snap = d.snapshot(1)
    assert snap.pending == ((0, 1, 0),) and snap.active_pairs == ()
    assert snap.has_pending(0, 1) and snap.proposals_to(1) == (0,)
    assert not d.are_allied(0, 1)


def test_reciprocal_proposal_forms_exactly_one_alliance() -> None:
    d = _state()
    d.propose(0, 1, ply=0)
    events = d.propose(1, 0, ply=1)
    assert [e.kind for e in events] == ["proposed", "formed"]
    assert d.are_allied(0, 1) and d.are_allied(1, 0)
    snap = d.snapshot(2)
    assert snap.active_pairs == ((0, 1),) and snap.pending == ()
    assert snap.allies(0) == (1,) and snap.allies(1) == (0,)
    with pytest.raises(DiplomacyError):
        d.propose(0, 1, ply=2)  # déjà alliés


def test_expiration_at_ttl() -> None:
    d = _state(proposal_ttl_plies=3)
    d.propose(0, 1, ply=2)
    alive = [True] * 4
    assert d.advance(4, alive) == []
    events = d.advance(5, alive)
    assert events == [AllianceEvent(5, 0, 1, "expired")]
    assert d.snapshot(5).pending == ()
    d.propose(0, 1, ply=5)  # une nouvelle proposition est possible
    d.propose(1, 0, ply=7)
    assert d.are_allied(0, 1)


def test_unilateral_break_cooldown_and_betrayal_rule() -> None:
    d = _state(break_cooldown_plies=4)
    d.propose(0, 1, 0)
    d.propose(1, 0, 1)
    assert [e.kind for e in d.break_alliance(1, 0, 3)] == ["broken"]
    assert not d.are_allied(0, 1)
    snap = d.snapshot(4)
    assert snap.betrayals[1][0] == 1 and snap.trust[0][1] == 0.5 and snap.trust[1][0] == 1.0
    assert snap.in_cooldown(0, 1)
    with pytest.raises(DiplomacyError):
        d.propose(0, 1, 5)  # délai après rupture jusqu'au demi-coup 7
    d.propose(0, 1, 7)
    d.propose(1, 0, 8)
    d.break_alliance(0, 1, 9)  # 1 avait rompu le premier : pas une trahison de 0
    assert d.snapshot(10).betrayals[0][1] == 0
    with pytest.raises(DiplomacyError):
        d.break_alliance(0, 1, 10)  # plus alliés
    summary = summarize_diplomacy(d.events, final_ply=12)
    assert (summary.alliances_formed, summary.alliance_breaks, summary.betrayals) == (2, 2, 1)
    assert summary.alliance_duration_mean == pytest.approx((2 + 1) / 2)


def test_alliance_dissolved_when_a_player_is_eliminated() -> None:
    d = _state()
    d.propose(0, 1, 0)
    d.propose(1, 0, 1)
    d.propose(2, 1, 2)
    events = d.advance(3, [True, False, True, True])
    assert [(e.kind, e.player, e.target) for e in events] == [
        ("dissolved", 1, 0),
        ("expired", 2, 1),
    ]
    assert d.allies(0) == ()
    with pytest.raises(DiplomacyError):
        d.propose(0, 1, 4)  # cible éliminée
    with pytest.raises(DiplomacyError):
        d.propose(1, 2, 4)  # proposant éliminé
    assert summarize_diplomacy(d.events, 10).alliance_duration_mean == 2


@pytest.mark.parametrize(("player", "target"), [(0, 0), (0, 9), (-1, 1), (0, -2)])
def test_invalid_players_are_rejected(player: int, target: int) -> None:
    d = _state()
    with pytest.raises(DiplomacyError):
        d.propose(player, target, 0)


def test_duplicates_caps_and_rejected_intents() -> None:
    d = _state(max_allies_per_player=1)
    d.propose(0, 1, 0)
    with pytest.raises(DiplomacyError):
        d.propose(0, 1, 1)  # doublon
    d.propose(1, 0, 1)
    with pytest.raises(DiplomacyError):
        d.propose(0, 2, 2)  # plafond d'un allié
    d.propose(2, 0, 2)
    with pytest.raises(DiplomacyError):
        d.propose(0, 2, 3)
    events = d.apply_intent(0, DiplomacyIntent("propose", 3), 4)
    assert [e.kind for e in events] == ["rejected"]
    events = d.apply_intent(3, DiplomacyIntent("break", 2), 4)
    assert events == [AllianceEvent(4, 3, 2, "rejected")]
    with pytest.raises(ValueError):
        DiplomacyIntent("annexer", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DiplomacyIntent("propose", None)
    assert summarize_diplomacy(d.events, 5).rejected_intents == 2


# configuration
def test_config_validation_and_roundtrip(tmp_path: Path) -> None:
    base = {"grid": 8, "players": ["a", "b", "c"]}
    cfg = GameConfig.from_dict({**base, "diplomacy": {"enabled": True, "betrayal_penalty": 0.2}})
    assert cfg.diplomacy.enabled and cfg.diplomacy.betrayal_penalty == 0.2
    assert GameConfig.from_dict(cfg.to_dict()) == cfg
    path = tmp_path / "d.yaml"
    cfg.to_yaml(path)
    assert GameConfig.from_yaml(path) == cfg
    plain = GameConfig.from_dict(base)
    assert "diplomacy" not in plain.to_dict() and not plain.diplomacy.enabled
    assert GameConfig.from_dict({**base, "diplomacy": True}).diplomacy.enabled
    for bad in (
        {"proposal_ttl_plies": 0},
        {"max_allies_per_player": 0},
        {"alliance_weight": 1.5},
        {"alliance_weight": float("nan")},
        {"betrayal_penalty": -0.1},
        {"break_cooldown_plies": -1},
        {"free_action": False},
        {"enabled": "oui"},
        {"inconnu": 1},
    ):
        with pytest.raises(ConfigError):
            GameConfig.from_dict({**base, "diplomacy": {"enabled": True, **bad}})
    with pytest.raises(ConfigError):
        GameConfig.from_dict({**base, "diplomacy": [1, 2]})
    example = GameConfig.from_yaml(CONFIGS / "games" / "diplomacy_4p_20x20.yaml")
    assert example.diplomacy.enabled and example.team_names == ("solo", "coalition", "outsider")


# ---------------------------------------------------------- matrice et politique
def _snapshot(n: int, pairs: tuple[tuple[int, int], ...] = (), betrayals: Any = None) -> Any:
    bet = betrayals or tuple(tuple(0 for _ in range(n)) for _ in range(n))
    trust = tuple(tuple(1.0 / (1 + bet[q][p]) for q in range(n)) for p in range(n))
    return DiplomacySnapshot(0, pairs, (), trust, bet)


def test_dynamic_matrix_order_symmetry_and_bounds() -> None:
    team = [0, 1, 1, 2]
    base = social_matrix("team", team, 1, alpha=0.5)
    w = dynamic_matrix(base, _snapshot(4, ((0, 3),)), alliance_weight=0.65)
    assert w[0][3] == w[3][0] == 0.65  # statut d'alliance symétrique
    assert w[1][2] == 0.5 and w[0][1] == 0.0
    assert dynamic_matrix(base, _snapshot(4), 0.65) == base
    bet = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0))  # 3 a trahi 0 deux fois
    w2 = dynamic_matrix(base, _snapshot(4, ((0, 3),), bet), 0.65, betrayal_penalty=0.5)
    assert w2[0][3] == pytest.approx(0.65 - 1.0)  # préférence de 0 envers 3 abaissée
    assert w2[3][0] == 0.65  # asymétrique : 0 n'a pas trahi 3
    w3 = dynamic_matrix(base, _snapshot(4, (), ((0, 0, 0, 0),) * 3 + ((5, 0, 0, 0),)), 0.0, 1.0)
    assert w3[0][3] == -1.0  # bornée
    with pytest.raises(ValueError):
        dynamic_matrix(base, _snapshot(4), 1.5)


def test_tactical_policy_against_the_leader() -> None:
    team = [0, 1, 1, 2]
    alive = [True] * 4
    shares = [0.2, 0.3, 0.3, 0.2]  # équipe 1 meneuse (0,6)
    snap = _snapshot(4)
    assert tactical_intent(0, team, alive, shares, snap, 2) == DiplomacyIntent("propose", 3)
    assert (
        tactical_intent(1, team, alive, shares, snap, 2) is None
    )  # l'équipe meneuse ne propose pas
    pending = DiplomacySnapshot(0, (), ((3, 0, 0),), snap.trust, snap.betrayals)
    assert tactical_intent(0, team, alive, shares, pending, 2) == DiplomacyIntent("propose", 3)
    allied = _snapshot(4, ((0, 1),))
    assert tactical_intent(0, team, alive, shares, allied, 2) == DiplomacyIntent("break", 1)
    assert tactical_intent(0, team, alive, shares, _snapshot(4, ((0, 3),)), 1) is None


# parties
def _dipl_config(seed: int = 1, enabled: bool = True) -> GameConfig:
    return GameConfig.from_dict(
        {
            "name": "dipl",
            "grid": 10,
            "seed": seed,
            "placement": {"mode": "regular", "rotate": True},
            "diplomacy": {"enabled": enabled},
            "players": [
                {"name": "solo", "team": "solo", "agent": "paranoid", "depth": 3},
                {
                    "name": "c1",
                    "team": "co",
                    "agent": "social",
                    "depth": 2,
                    "profile": "dynamic_team",
                },
                {
                    "name": "c2",
                    "team": "co",
                    "agent": "social",
                    "depth": 2,
                    "profile": "dynamic_team",
                },
                {
                    "name": "out",
                    "team": "out",
                    "agent": "social",
                    "depth": 2,
                    "profile": "dynamic_team",
                },
            ],
        }
    )


def _trace(m: Match) -> list[tuple[Any, ...]]:
    return [(e.player, e.direction, e.diplomacy) for e in m.events]


def test_same_seed_same_diplomatic_history_and_moves() -> None:
    games = []
    for _ in range(2):
        m = Match(_dipl_config(seed=3))
        m.run()
        games.append(_trace(m))
    assert games[0] == games[1]
    seen: set[str] = set()
    for seed in range(5):
        m = Match(_dipl_config(seed))
        m.run()
        seen.update(e.kind for ev in m.events for e in ev.diplomacy)
    assert {"proposed", "expired"} <= seen


def test_permanent_teams_and_winner_are_unchanged() -> None:
    for seed in range(4):
        m = Match(_dipl_config(seed))
        result = m.run()
        out = m.state.outcome()
        assert out is not None
        assert result.winning_team == m.config.team_names[out.winning_team]
        assert result.winning_team in ("solo", "co", "out")
        winners = {m.config.players[p].team for p in out.winners}
        assert winners == {result.winning_team}  # un seul camp permanent gagne
        assert m.diplomacy is not None
        assert all(not m.diplomacy.allies(p) for p in range(4) if not m.state.is_alive(p))


def test_disabled_diplomacy_is_identical_to_historical_behaviour() -> None:
    no_key = _dipl_config(seed=2, enabled=False)
    data = no_key.to_dict()
    assert "diplomacy" not in data
    a = Match(GameConfig.from_dict(data))
    a.run()
    explicit = GameConfig.from_dict({**data, "diplomacy": {"enabled": False}})
    b = Match(explicit)
    b.run()
    assert _trace(a) == _trace(b)
    assert all(e.diplomacy == () for e in a.events)
    assert a.result is not None and a.result.metrics is not None
    assert not a.result.metrics.diplomacy_enabled and a.result.metrics.proposals == 0
    assert a.diplomacy is None


def test_historical_agents_accept_missing_or_present_context() -> None:
    cfg = _dipl_config()
    st = GameState(cfg)
    ctx = DecisionContext(ply=0, diplomacy=_snapshot(4))
    for name in ("random", "greedy", "paranoid", "maxn", "negamax", "social"):
        if name not in AGENTS or name == "negamax":
            continue
        a = make_agent(name, 0, cfg, depth=2).choose_move(st.copy())
        b = make_agent(name, 0, cfg, depth=2).choose_move(st.copy(), 0, ctx)
        assert a.move == b.move and b.diplomacy is None


def test_social_agent_without_diplomacy_keeps_static_matrix() -> None:
    cfg = _dipl_config()
    agent = SocialAgent(1, cfg, profile="team", depth=2)
    assert not agent.diplomatic
    st = GameState(cfg)
    st.apply(st.legal_moves()[0])
    ctx = DecisionContext(1, _snapshot(4, ((1, 3),)))
    dec = agent.choose_move(st, 1, ctx)
    assert dec.diplomacy is None and agent.matrix == agent.base_matrix
    dyn = SocialAgent(1, cfg, profile="dynamic_team", depth=2)
    dyn.choose_move(st, 1, ctx)
    assert dyn.matrix[1][3] == pytest.approx(cfg.diplomacy.alliance_weight)
    dyn.reset()
    assert dyn.matrix == dyn.base_matrix
    with pytest.raises(ConfigError):
        SocialAgent(1, cfg, profile="dynamic_team", break_margin=2.0)


# journal et métriques
def test_log_records_and_replays_diplomacy(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    m = Match(_dipl_config(seed=0), observers=[Recorder(path)])
    result = m.run()
    log = GameLog.load(path)
    assert log.format == "tron-log/2"
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["diplomacy"]["enabled"] is True and header["config"]["diplomacy"]["enabled"]
    live = [e for ev in m.events for e in ev.diplomacy]
    assert log.diplomacy_events() == live
    assert log.metrics() == result.metrics
    assert result.metrics is not None and result.metrics.diplomacy_enabled
    assert result.metrics.proposals == sum(1 for e in live if e.kind == "proposed")
    row = result.to_row()
    assert row["dipl_enabled"] == 1 and row["dipl_proposals"] == result.metrics.proposals


def test_old_v1_logs_are_still_readable_and_unknown_formats_rejected(tmp_path: Path) -> None:
    path = tmp_path / "v2.jsonl"
    Match(make_config(8, 8, 2, seed=4), observers=[Recorder(path)]).run()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["format"] = "tron-log/1"
    header.pop("diplomacy")
    result = json.loads(lines[-1])
    result.pop("metrics")
    v1 = tmp_path / "v1.jsonl"
    v1.write_text(
        "\n".join([json.dumps(header), *lines[1:-1], json.dumps(result)]) + "\n", encoding="utf-8"
    )
    log = GameLog.load(v1)
    assert log.format == "tron-log/1" and log.diplomacy_events() == []
    assert log.final_state().is_terminal()
    assert log.metrics().teams == ("J0", "J1")
    header["format"] = "tron-log/9"
    bad = tmp_path / "v9.jsonl"
    bad.write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(LogError):
        GameLog.load(bad)
