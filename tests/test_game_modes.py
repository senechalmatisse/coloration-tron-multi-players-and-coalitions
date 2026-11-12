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
        {"mode": "zone_control", "zones": [[0, 0]], "target_score": 0},
        {"mode": "zone_control", "zones": [[0, 0]], "target_score": 2.5},
        {"mode": "zone_control", "zones": [[0, 0]], "target_score": True},
        {"mode": "zone_control", "zones": [[0, 0]], "max_rounds": "12"},
        {"mode": "zone_control", "zones": [[0, 0.5]], "target_score": 2},
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


# ------------------------------------------------------ obstacles dynamiques
def _obst_cfg(seed: int = 4, grid: int = 9, **obstacles: object) -> GameConfig:
    spec = {"every": 2, "count": 3, "max": 12, **obstacles}
    return GameConfig.from_dict(
        {
            "grid": grid,
            "seed": seed,
            "placement": {"mode": "regular"},
            "players": [
                {"name": "a", "team": "A"},
                {"name": "b", "team": "B"},
                {"name": "c", "team": "A"},
            ],
            "rules": {"mode": "classic", "obstacles": spec},
        }
    )


def _walk(st: GameState, rng: random.Random, limit: int = 400) -> list[int]:
    moves = []
    while not st.is_terminal() and len(moves) < limit:
        legal = st.legal_moves()
        d = rng.choice(legal) if legal else PASS
        st.apply(d)
        moves.append(d)
    return moves


@pytest.mark.parametrize(
    "obstacles",
    [
        {"every": 0},
        {"every": 2, "count": 0},
        {"every": 2, "start": 0},
        {"every": 2, "max": 0},
        {"every": 2, "margin": -1},
        {"every": 2, "rate": 3},
        {"count": 2},
        {"every": True},
        {"every": 1.5},
        {"every": "3"},
        "souvent",
    ],
)
def test_obstacle_rules_validation(obstacles: object) -> None:
    with pytest.raises(ConfigError):
        _cfg(rules={"mode": "classic", "obstacles": obstacles})


def test_obstacle_rules_roundtrip_and_combinations() -> None:
    cfg = _obst_cfg(start=3, margin=0)
    assert GameConfig.from_dict(cfg.to_dict()) == cfg
    obs = cfg.rules.obstacles
    assert obs is not None and obs.to_dict() == {
        "every": 2,
        "count": 3,
        "margin": 0,
        "start": 3,
        "max": 12,
    }
    assert [r for r in range(1, 10) if obs.is_event(r)] == [3, 5, 7, 9]
    assert [r for r in range(1, 8) if _obst_cfg().rules.obstacles.is_event(r)] == [2, 4, 6]  # type: ignore[union-attr]
    for mode in ({"mode": "survival", "victory_scope": "individual"}, {"mode": "survival"}):
        combo = _cfg(rules={**mode, "obstacles": {"every": 1}})
        assert GameConfig.from_dict(combo.to_dict()) == combo
    zone = _zone_cfg(target_score=40)
    combo = GameConfig.from_dict(
        {**zone.to_dict(), "rules": {**zone.to_dict()["rules"], "obstacles": {"every": 1}}}
    )
    st = GameState(combo)
    keys = [st.key]
    _walk(st, random.Random(2))
    assert st.n_obstacles > 0 and st.key == st.compute_key()
    while st.ply:
        st.undo()
    assert st.key == keys[0] and st.n_obstacles == 0 and st.scores == [0, 0]


def test_obstacles_are_deterministic_scheduled_and_safe() -> None:
    from tron.core import FREE, OBSTACLE

    cfg = _obst_cfg(margin=1)
    obs = cfg.rules.obstacles
    assert obs is not None
    rng = random.Random(11)
    st = GameState(cfg)
    events: list[tuple[int, list[tuple[int, int]]]] = []
    while not st.is_terminal():
        before = st.grid.copy()
        legal = st.legal_moves()
        st.apply(rng.choice(legal) if legal else PASS)
        placed = st.last_obstacles()
        if placed:
            events.append((st.round, placed))
            assert obs.is_event(st.round) and len(placed) <= obs.count
            heads = [st.head(p) for p in range(st.n) if st.is_alive(p)]
            for x, y in placed:
                assert before[st.index(x, y)] == FREE  # jamais sur une case occupee
                assert all(abs(x - hx) + abs(y - hy) > obs.margin for hx, hy in heads)
        assert st.n_obstacles == int((st.grid == OBSTACLE).sum()) <= 12
    assert events and sum(len(c) for _, c in events) == st.n_obstacles
    # Memes graine et coups : memes obstacles ; autre graine, memes coups : autres obstacles.
    played = _played_directions(st)
    replay = GameState(cfg)
    other = GameState(_obst_cfg(seed=5, margin=1))
    assert other.positions == replay.positions
    first = []
    for d in played:
        replay.apply(d)
        if replay.last_obstacles():
            first.append((replay.round, replay.last_obstacles()))
    assert first == events
    for d in played[: 3 * obs.first]:  # jusqu'au premier evenement
        other.apply(d)
    assert other.round == obs.first and other.last_obstacles()
    assert other.last_obstacles() != events[0][1]


def _played_directions(st: GameState) -> list[int]:
    """Directions jouees depuis l'etat initial, retrouvees dans l'historique."""
    return [PASS if t < 0 else st._off.index(t - h) for _, h, t, _ in st._hist]


def test_obstacles_apply_undo_copy_and_snapshot() -> None:
    from tron.core import OBSTACLE

    cfg = _obst_cfg(every=1, count=2, max=None, margin=0)
    st = GameState(cfg)
    rng = random.Random(4)
    grids = [st.grid.copy()]
    keys = [st.key]
    while not st.is_terminal():
        legal = st.legal_moves()
        st.apply(rng.choice(legal) if legal else PASS)
        assert st.key == st.compute_key()
        grids.append(st.grid.copy())
        keys.append(st.key)
        back = GameState.from_snapshot(cfg, st.to_snapshot())
        assert (back.grid == st.grid).all() and back.n_obstacles == st.n_obstacles
        assert back.key == st.key and back.last_obstacles() == st.last_obstacles()
    assert "#" in st.to_text() and "/" in st.to_snapshot()["grid"]
    cp = st.copy()
    cp.undo()
    assert st.key == keys[-1] and (st.grid == grids[-1]).all()
    while st.ply:
        st.undo()
        assert st.key == keys[st.ply] and (st.grid == grids[st.ply]).all()
    assert st.n_obstacles == 0 and not (st.grid == OBSTACLE).any()
    # Instantanes incoherents : obstacle sans regle, obstacle absent de l'historique.
    st2 = GameState(cfg)
    _walk(st2, random.Random(4), limit=12)
    assert all(st2.is_alive(p) for p in range(3)) and st2.n_obstacles > 0
    text_state = GameState.from_text(cfg, st2.to_text(), current=st2.current)
    assert text_state.n_obstacles == st2.n_obstacles
    assert (text_state.grid == st2.grid).all()  # carte ASCII : grille seule (tour 0)
    assert text_state.key == text_state.compute_key()
    snap = st2.to_snapshot()
    with pytest.raises(ValueError, match="valeur de case"):
        GameState.from_snapshot(_cfg(n=3), {**snap, "positions": snap["positions"]})
    cell = snap["grid"].index("1")
    tampered = snap["grid"][:cell] + "/" + snap["grid"][cell + 1 :]
    with pytest.raises(ValueError):
        GameState.from_snapshot(cfg, {**snap, "grid": tampered})


def test_obstacles_can_decide_the_game() -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 5,
            "seed": 1,
            "placement": {"positions": [[0, 0], [4, 4]]},
            "players": [{"name": "a", "team": "A"}, {"name": "b", "team": "B"}],
            "rules": {"obstacles": {"every": 1, "count": 25, "margin": 0}},
        }
    )
    st = GameState(cfg)
    st.apply(1)  # a -> (1, 0)
    st.apply(3)  # b -> (3, 4) : fin du tour 1, toutes les cases libres sont bloquees
    assert st.n_obstacles == 21 and not st.legal_moves()
    st.apply(PASS)
    out = st.outcome()
    assert st.is_terminal() and out is not None and out.winning_team == 1
    assert st.last_obstacles() == []  # partie finie : plus d'obstacles
    st.undo()
    st.undo()
    assert st.n_obstacles == 0 and st.legal_moves()


def test_obstacles_match_log_replay_and_checkpoint(tmp_path: Path) -> None:
    import json

    from tron.sim import GameLog, LogError, Recorder

    cfg = GameConfig.from_yaml(CONFIGS / "games" / "obstacles_14x14.yaml")
    light = GameConfig.from_dict(
        {
            **cfg.to_dict(),
            "players": [
                {"name": "a1", "team": "A", "agent": "paranoid", "depth": 2},
                {"name": "b1", "team": "B", "agent": "greedy"},
                {"name": "a2", "team": "A", "agent": "social", "depth": 2, "profile": "team"},
                {"name": "b2", "team": "B", "agent": "mcts", "simulations": 40},
            ],
        }
    )
    path = tmp_path / "o.jsonl"
    result = Match(light, observers=[Recorder(path)]).run()
    log = GameLog.load(path)
    assert result.obstacles > 0 and result.to_row()["obstacles"] == result.obstacles
    assert result.to_row()["team_scores"] == ""
    logged = [c for m in log.moves for c in m.get("obstacles", ())]
    assert len(logged) == result.obstacles
    final = log.final_state()
    assert final.n_obstacles == result.obstacles
    m = Match(light)
    while m.state.n_obstacles == 0 or m.state.ply % 4 != 1:
        m.step()
    saved = m.save_checkpoint(tmp_path / "o.json")
    resumed = tmp_path / "r.jsonl"
    r2 = Match.from_checkpoint(saved, observers=[Recorder(resumed)]).run()
    assert (r2.obstacles, r2.winning_team, r2.n_plies) == (
        result.obstacles,
        result.winning_team,
        result.n_plies,
    )

    def timeless(moves: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{k: v for k, v in mv.items() if k != "time"} for mv in moves]

    assert timeless(GameLog.load(resumed).moves) == timeless(log.moves)
    # Un journal dont les obstacles ont ete modifies est refuse au rejeu.
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next(i for i, ln in enumerate(lines) if json.loads(ln).get("obstacles") and i > 0)
    rec = json.loads(lines[idx])
    rec["obstacles"][0] = [rec["obstacles"][0][0] ^ 1, rec["obstacles"][0][1]]
    lines[idx] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LogError, match="obstacles"):
        GameLog.load(path).final_state()


def test_obstacle_rendering() -> None:
    from tron.viz import make_palette
    from tron.viz.render import color_grid

    cfg = _obst_cfg(every=1, count=4, margin=0)
    st = GameState(cfg)
    _walk(st, random.Random(3), limit=6)
    pal = make_palette(cfg.team_index)
    img = color_grid(st, pal)
    cells = [st.coords(c) for c in range(st.size) if st.grid[c] == -3]
    assert cells and all(tuple(img[y, x]) == pal.muted for x, y in cells)
