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


def test_classic_zobrist_keys_unchanged() -> None:
    """Cles d'une partie classique identiques a celles du code d'origine (commit 5db062c)."""
    cfg = GameConfig.from_dict(
        {
            "grid": 11,
            "seed": 5,
            "placement": {"mode": "regular"},
            "players": [
                {"name": "a", "team": "A"},
                {"name": "b", "team": "B"},
                {"name": "c", "team": "A"},
            ],
        }
    )
    st = GameState(cfg)
    rng = random.Random(7)
    keys = [st.key]
    for _ in range(40):
        if st.is_terminal():
            break
        moves = st.legal_moves()
        st.apply(rng.choice(moves) if moves else PASS)
        keys.append(st.key)
    assert (len(keys), keys[0], keys[-1]) == (41, 2498487232173843075, 3013909396529148895)


# --------------------------------------------------------- contrôle de zone
def _zone_cfg(**rules: object) -> GameConfig:
    base = {
        "grid": 7,
        "placement": {"positions": [[1, 1], [5, 5], [5, 1]]},
        "players": [
            {"name": "a", "team": "A"},
            {"name": "b", "team": "B"},
            {"name": "c", "team": "A"},
        ],
        "rules": {
            "mode": "zone_control",
            "zones": [{"x": 0, "y": 0, "width": 3, "height": 3}, [4, 4, 3, 3]],
            "target_score": 3,
            **rules,
        },
    }
    return GameConfig.from_dict(base)


@pytest.mark.parametrize(
    "rules",
    [
        {"mode": "zone_control", "target_score": 3},  # pas de zone
        {"mode": "zone_control", "zones": [[0, 0, 2, 2]]},  # ni score cible ni durée
        {"mode": "zone_control", "zones": [[8, 8, 2, 2]], "target_score": 2},  # hors grille
        {"mode": "zone_control", "zones": [[0, 0, 0, 2]], "target_score": 2},  # zone vide
        {"mode": "zone_control", "zones": [[0, 0]], "target_score": 2, "zone_rule": "voronoi"},
        {"mode": "zone_control", "zones": [[0, 0]], "target_score": 999},
        {"mode": "classic", "zones": [[0, 0]]},
        {"mode": "zone_control", "zones": [{"x": 0, "y": 0, "z": 1}], "target_score": 2},
    ],
)
def test_zone_rules_validation(rules: object) -> None:
    with pytest.raises(ConfigError):
        _cfg(rules=rules)


def test_zone_points_contested_zones_and_undo() -> None:
    cfg = _zone_cfg()
    assert GameConfig.from_dict(cfg.to_dict()) == cfg
    st = GameState(cfg)
    assert st.round == 0 and st.scores == [0, 0] and st.key == st.compute_key()
    keys = [st.key]
    # Tour 1 : a (A) reste dans la zone 0, b (B) dans la zone 1, c (A) hors zone.
    for d in (1, 3, 2):  # a -> (2, 1) ; b -> (4, 5) ; c -> (5, 2) : fin du tour
        assert st.round == 0
        st.apply(d)
        keys.append(st.key)
    assert st.round == 1 and st.scores == [1, 1] and st.key == st.compute_key()
    # Tour 2 : idem (b descend en (4, 6), toujours dans la zone 1).
    for d in (2, 2, 2):
        st.apply(d)
        keys.append(st.key)
    assert st.round == 2 and st.scores == [2, 2]
    # Tour 3 : c entre dans la zone 1 occupee par b : zone disputee, seule A marque.
    for d in (3, 1, 2):  # a -> (1, 2) ; b -> (5, 6) ; c -> (5, 4)
        st.apply(d)
        keys.append(st.key)
    assert st.head(1) == (5, 6) and st.head(2) == (5, 4)
    assert st.round == 3 and st.scores == [3, 2]
    assert st.is_terminal() and st.key == st.compute_key()  # score cible 3 atteint par A
    out = st.outcome()
    assert out is not None and out.winning_team == 0 and out.winners == (0, 2)
    assert out.ranks == (1, 1, 1)  # rangs de survie : tout le monde est vivant
    keys.pop()
    while st.ply:
        st.undo()
        assert st.key == keys.pop() == st.compute_key()
    assert st.round == 0 and st.scores == [0, 0] and not st.is_terminal()


def test_zone_draw_on_equal_scores() -> None:
    timed = GameState(_zone_cfg(target_score=None, max_rounds=1))
    for d in (1, 3, 2):
        timed.apply(d)
    out = timed.outcome()
    assert timed.is_terminal() and timed.round == 1 and timed.scores == [1, 1]
    assert out is not None and out.winning_team == -1 and out.winners == ()
    capped = GameState(_zone_cfg(target_score=None, max_rounds=2))
    for d in (1, 3, 3, 2, 3, 3):  # b quitte la zone 1 au tour 2
        capped.apply(d)
    out = capped.outcome()
    assert capped.scores == [2, 1] and out is not None and out.winning_team == 0


def test_zone_last_team_alive_wins_before_target() -> None:
    st = GameState(_zone_cfg(target_score=30))
    st.apply(1)
    st.apply(PASS)  # b, seul joueur de B, est elimine
    out = st.outcome()
    assert st.is_terminal() and st.scores == [0, 0]
    assert out is not None and out.winning_team == 0 and out.winners == (0, 2)


def test_zone_target_score_decides_the_winner() -> None:
    cfg = _zone_cfg(target_score=2, max_rounds=None)
    st = GameState(cfg)
    for d in (1, 3, 3, 2, 3, 3):  # a reste en zone 0 ; b quitte la zone 1 ; c erre
        st.apply(d)
    assert st.is_terminal() and st.scores[0] >= 2
    out = st.outcome()
    assert out is not None and out.winning_team == 0 and out.winners == (0, 2)
    assert st.scores == [2, 1]  # b a quitte la zone 1 au tour 2


def test_zone_random_walk_keeps_incremental_key_and_snapshot() -> None:
    rng = random.Random(3)
    cfg = _zone_cfg(target_score=30, max_rounds=60)
    st = GameState(cfg)
    while not st.is_terminal():
        moves = st.legal_moves()
        st.apply(rng.choice(moves) if moves and rng.random() < 0.95 else PASS)
        assert st.key == st.compute_key()
        cp = st.copy()
        if cp.ply:
            cp.undo()
            assert cp.key == cp.compute_key() and st.key == st.compute_key()
        back = GameState.from_snapshot(cfg, st.to_snapshot())
        assert (back.round, back.scores, back.key, back.over) == (
            st.round,
            st.scores,
            st.key,
            st.over,
        )
    plies = st.ply
    while st.ply:
        st.undo()
    assert st.round == 0 and st.scores == [0, 0] and plies > 0


def test_zone_match_log_replay_and_checkpoint(tmp_path: Path) -> None:
    import json

    from tron.sim import CheckpointError, GameLog, Recorder

    cfg = GameConfig.from_yaml(CONFIGS / "games" / "zone_control_16x16.yaml")
    path = tmp_path / "z.jsonl"
    result = Match(cfg, observers=[Recorder(path)]).run()
    assert result.mode == "zone_control" and result.rounds > 0 and len(result.scores) == 2
    assert result.to_row()["team_scores"] == ";".join(str(s) for s in result.scores)
    log = GameLog.load(path)
    final = log.final_state()
    assert (final.round, final.scores) == (result.rounds, list(result.scores))
    assert any("zone" in m for m in log.moves)
    # Reprise au milieu d'un tour, apres plusieurs changements de controle.
    m = Match(cfg)
    while m.state.round < 5 or m.state.ply % 4 != 2:
        m.step()
    saved = m.save_checkpoint(tmp_path / "z.json")
    resumed_log = tmp_path / "r.jsonl"
    r2 = Match.from_checkpoint(saved, observers=[Recorder(resumed_log)]).run()
    assert (r2.scores, r2.rounds, r2.winning_team) == (
        result.scores,
        result.rounds,
        result.winning_team,
    )

    def timeless(moves: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{k: v for k, v in mv.items() if k != "time"} for mv in moves]

    # Memes coups, memes details et memes evenements de zone (seules les durees varient).
    assert timeless(GameLog.load(resumed_log).moves) == timeless(log.moves)
    data = json.loads(saved.read_text(encoding="utf-8"))
    for bad in (None, [0], [0, 0, 0, 5]):
        data["zone_controllers"] = bad
        saved.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CheckpointError):
            Match.from_checkpoint(saved)


def test_zone_heuristic_contract() -> None:
    from tron.eval import ZoneHeuristic, make_heuristic

    cfg = _zone_cfg(target_score=50)
    st = GameState(cfg)
    heur = ZoneHeuristic(7, 7, 3)
    s = heur.shares(st)
    assert sum(s) == pytest.approx(1.0) and all(x > 0 for x in s)
    st.apply(PASS)
    s = heur.shares(st)
    assert s[0] == 0.0 and sum(s) == pytest.approx(1.0)
    plain = GameState(make_config(7, 7, 2))
    assert heur.shares(plain)[:2] == VoronoiHeuristic(7, 7, 3).shares(plain)[:2]
    for bad in (
        {"territory": -1},
        {"score": float("nan")},
        {"territory": 0, "proximity": 0, "score": 0},
    ):
        with pytest.raises(ValueError):
            make_heuristic("zone", 7, 7, 3, **bad)


def test_zone_rendering_and_panel_status() -> None:
    from tron.viz import LiveSource, PlaybackController, make_palette, render_frame

    cfg = _zone_cfg(target_score=50)
    st = GameState(cfg)
    pal = make_palette(cfg.team_index)
    img = render_frame(st, pal, cell=8, labels=False)
    assert tuple(img[0, 4]) == pal.ink_secondary  # bord haut de la zone 0
    assert tuple(img[4 * 8 + 3, 4 * 8]) == pal.ink_secondary  # bord gauche de la zone 1
    plain = GameState(make_config(7, 7, 3))
    assert not (render_frame(plain, pal, cell=8, labels=False) == pal.ink_secondary).all(-1).any()
    ctl = PlaybackController(LiveSource(cfg), paused=True)
    assert ctl.panel().status.endswith("· tour 0 · A 0 – B 0")
    for _ in range(3):
        ctl.step_forward()
    assert "· tour 1 · " in ctl.panel().status
