"""Tests des explications de décision et de leur visualisation."""

from __future__ import annotations

import importlib
import json
import math
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import make_config
from tron.agents import (
    DecisionContext,
    DecisionDetails,
    DiplomacySnapshot,
    GreedyAgent,
    MaxNAgent,
    ParanoidAgent,
    make_agent,
)
from tron.core import GameState
from tron.sim import GameLog, Match, Recorder
from tron.viz import LiveSource, PlaybackController, ReplaySource, make_palette, render_frame
from tron.viz.controller import UNAVAILABLE
from tron.viz.render import pv_path


def midgame(n: int = 3, size: int = 9, plies: int = 6, seed: int = 1) -> GameState:
    cfg = make_config(size, size, n, seed=seed, placement={"mode": "regular", "rotate": True})
    st = GameState(cfg)
    rng = random.Random(seed)
    for _ in range(plies):
        moves = st.legal_moves()
        if st.is_terminal() or not moves:
            break
        st.apply(rng.choice(moves))
    return st


def test_details_roundtrip_and_ranking() -> None:
    d = DecisionDetails(
        scores=((0, 0.25), (1, None), (2, 0.75), (3, math.nan)),
        visits=((0, 3), (2, 7)),
        social_weights=((0, 1.0), (2, -1.0)),
        allies=(2,),
        heuristic_weights=(("safety", 0.3), ("territory", 0.7)),
        opponents=((1, "territorial", 0.8),),
        note="essai",
    )
    data = json.loads(json.dumps(d.to_dict(), allow_nan=False))
    assert data["scores"][3] == [3, None]  # NaN -> null (JSON strict)
    back = DecisionDetails.from_dict(data)
    assert back.scores[:3] == d.scores[:3] and back.scores[3] == (3, None)
    assert (back.visits, back.allies, back.opponents, back.note) == (
        d.visits,
        d.allies,
        d.opponents,
        d.note,
    )
    assert back.ranked()[0] == (2, 0.75) and back.ranked()[-1][1] is None
    assert back.score_of(0) == 0.25 and back.score_of(1) is None
    assert DecisionDetails().to_dict() == {}


def test_greedy_shows_the_score_of_every_legal_move() -> None:
    st = midgame()
    agent = GreedyAgent(st.current, st.config, tie_break="fixed")
    dec = agent.choose_move(st)
    assert dec.details is not None
    assert [m for m, _ in dec.details.scores] == st.legal_moves()
    assert all(v is not None for _, v in dec.details.scores)
    assert dec.details.score_of(dec.move) == pytest.approx(dec.stats.value)


@pytest.mark.parametrize("cls", [ParanoidAgent, MaxNAgent])
def test_search_agents_report_only_exact_values(cls: type) -> None:
    rng = random.Random(7)
    checked = 0
    for trial in range(12):
        st = midgame(3, 9, rng.randint(3, 12), seed=trial)
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        agent = cls(st.current, st.config, depth=3)
        plain = cls(st.current, st.config, depth=3, pruning=False)
        dec = agent.choose_move(st)
        full = plain.run_search(st, 3)
        assert dec.details is not None and dec.stats.pv and dec.stats.pv[0] == dec.move
        for m, v in dec.details.scores:
            if v is not None:  # valeur exacte : identique à la recherche sans élagage
                assert v == pytest.approx(full.root_values[m], abs=1e-12)
        assert dec.details.score_of(dec.move) is not None
        checked += 1
    assert checked >= 6


def test_mcts_social_and_adaptive_details() -> None:
    st = midgame(3, 10, 6, seed=3)
    mcts = make_agent("mcts", st.current, st.config, simulations=40)
    d = mcts.choose_move(st).details
    assert d is not None and sum(k for _, k in d.visits) == 40
    assert {m for m, _ in d.scores} == {m for m, _ in d.visits}
    cfg = make_config(9, 9, 3, teams=["A", "B", "A"], seed=2)
    st2 = GameState(cfg)
    social = make_agent("social", 0, cfg, profile="team", depth=2)
    n = 3
    zeros = tuple(tuple(0 for _ in range(n)) for _ in range(n))
    snap = DiplomacySnapshot(0, ((0, 1),), (), tuple((1.0,) * n for _ in range(n)), zeros)
    ds = social.choose_move(st2, 0, DecisionContext(0, snap)).details
    assert ds is not None and ds.social_weights == ((0, 1.0), (2, 1.0)) and ds.allies == (1,)
    adaptive = make_agent("adaptive", 0, cfg, depth=2)
    da = adaptive.choose_move(st2).details
    assert da is not None and dict(da.heuristic_weights)["territory"] == 1.0
    assert {q for q, _, _ in da.opponents} == {1}
    assert da.scores


def _config(seed: int = 2):
    return make_config(
        10,
        10,
        4,
        teams=["A", "B", "A", "B"],
        agents=["paranoid", "maxn", "greedy", "mcts"],
        depth=[2, 2, 1, 1],
        params=[{}, {}, {}, {"simulations": 20}],
        seed=seed,
        placement={"mode": "regular", "rotate": True},
    )


def test_replay_restores_the_same_details(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    match = Match(_config(), observers=[Recorder(path)])
    match.run()
    log = GameLog.load(path)
    assert len(log.moves) == len(match.events)
    for i, event in enumerate(match.events):
        got = log.details(i)
        if event.details is None:
            assert got is None
        else:
            assert got is not None and got.to_dict() == event.details.to_dict()
    live = PlaybackController(LiveSource(_config()), paused=True)
    replay = PlaybackController(ReplaySource(log), paused=True)
    for _ in range(9):
        live.step_forward()
        replay.step_forward()
        a, b = live.decision_panel(), replay.decision_panel()
        assert a is not None and b is not None
        assert a.lines() == b.lines()
    assert replay.step_backward()
    panel = replay.decision_panel()
    assert panel is not None and panel.ply == 7


def test_missing_details_never_block_play_or_replay(tmp_path: Path) -> None:
    path = tmp_path / "sans.jsonl"
    Match(_config(), observers=[Recorder(path, details=False)]).run()
    log = GameLog.load(path)
    assert all("details" not in m for m in log.moves) and log.details(0) is None
    ctl = PlaybackController(ReplaySource(log), paused=True)
    assert ctl.decision_panel() is None  # aucun coup joué
    ctl.step_forward()
    panel = ctl.decision_panel()
    assert panel is not None and f"coups : {UNAVAILABLE}" in panel.lines()
    # Journal antérieur sans pv, leaves ni prunes.
    lines = path.read_text(encoding="utf-8").splitlines()
    old = []
    for line in lines:
        rec = json.loads(line)
        for key in ("pv", "leaves", "prunes"):
            rec.pop(key, None)
        old.append(json.dumps(rec))
    old_path = tmp_path / "ancien.jsonl"
    old_path.write_text("\n".join(old) + "\n", encoding="utf-8")
    ctl_old = PlaybackController(ReplaySource(GameLog.load(old_path)), paused=True)
    ctl_old.step_forward()
    text = ctl_old.decision_panel().lines()  # type: ignore[union-attr]
    assert any(UNAVAILABLE in line for line in text)
    ctl_old.toggle_decision()
    assert ctl_old.decision_path() == []


def test_pv_path_and_headless_rendering() -> None:
    cfg = make_config(6, 5, 2, placement={"positions": [[1, 1], [4, 3]]})
    st = GameState(cfg)
    path = pv_path(st, [1, 0, 1])  # J0 : E ; J1 : N ; J0 : E
    assert path == [(0, 2, 1), (1, 4, 2), (0, 3, 1)]
    assert st.ply == 0  # état intact
    assert pv_path(st, [7]) == []  # coup impossible : arrêt sans erreur
    pal = make_palette(cfg.team_index)
    img = render_frame(st, pal, cell=12, labels=False, path=path)
    assert img.shape == (60, 72, 3) and img.dtype == np.uint8
    assert tuple(img[1 * 12 + 6, 2 * 12 + 6]) == pal.head[0]
    plain = render_frame(st, pal, cell=12, labels=False)
    assert tuple(plain[1 * 12 + 6, 2 * 12 + 6]) == pal.surface


# fenêtre (mock API)
class _Stub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.x = args[1] if len(args) > 1 else 0
        self.y = args[2] if len(args) > 2 else 0
        self.text = args[0] if args else ""
        self.drawn: list[str] = []

    def draw(self) -> None:
        self.drawn.append(self.text)


class _SpriteList(list):
    def draw(self) -> None:
        pass


class _Window:
    def __init__(self, width: int, height: int, title: str, update_rate: float = 1 / 60) -> None:
        self.width, self.height = width, height

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_decision_toggle_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("arcade")
    calls: list[str] = []
    mod.Window = _Window  # type: ignore[attr-defined]
    mod.SpriteList = _SpriteList  # type: ignore[attr-defined]
    mod.SpriteSolidColor = _Stub  # type: ignore[attr-defined]
    mod.Text = _Stub  # type: ignore[attr-defined]
    mod.draw_lrbt_rectangle_filled = lambda *a, **k: calls.append("rect")  # type: ignore[attr-defined]
    mod.draw_lrbt_rectangle_outline = lambda *a, **k: calls.append("outline")  # type: ignore[attr-defined]
    mod.draw_line = lambda *a, **k: calls.append("line")  # type: ignore[attr-defined]
    mod.key = types.SimpleNamespace(  # type: ignore[attr-defined]
        SPACE=1, RIGHT=2, LEFT=3, UP=4, DOWN=5, V=6, N=7, HOME=8, END=9, S=10, ESCAPE=11, Q=12, D=13
    )
    monkeypatch.setitem(sys.modules, "arcade", mod)
    app = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    try:
        ctl = PlaybackController(LiveSource(_config()), paused=True)
        win = app.TronWindow(ctl, cell=16)
        win.on_key_press(13, 0)  # D
        assert ctl.show_decision
        win.on_draw()  # avant le premier coup : message d'attente
        assert any("aucune" in t for t in win.t_decision[0].drawn)
        for _ in range(3):
            win.on_key_press(2, 0)  # pas suivant
        win.on_draw()
        assert any(t.startswith("Décision") for t in win.t_decision[0].drawn)
        win.on_key_press(13, 0)
        assert not ctl.show_decision
    finally:
        monkeypatch.delitem(sys.modules, "arcade")
        importlib.reload(app)
