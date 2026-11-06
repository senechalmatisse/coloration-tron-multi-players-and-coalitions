"""Tests de la visualisation (jalon J2) : palette, rendu, contrôle de lecture, exports."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import make_config
from tron.core import GameState
from tron.sim import GameLog, Match, Recorder
from tron.viz import (
    LiveSource,
    PlaybackController,
    ReplaySource,
    export_animation,
    log_frames,
    make_palette,
    render_frame,
    save_png,
)
from tron.viz.palette import hex_to_rgb, label_ink, mix, rgb_to_hex, shade


def _config(seed: int = 2):
    return make_config(
        10,
        10,
        4,
        teams=["A", "B", "A", "B"],
        agents=["paranoid", "maxn", "greedy", "random"],
        depth=[2, 2, 1, 1],
        seed=seed,
        placement={"mode": "regular", "rotate": True},
    )


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    path = tmp_path / "partie.jsonl"
    Match(_config(), observers=[Recorder(path)]).run()
    return path


def test_palette() -> None:
    pal = make_palette([0, 1, 0, 1])
    assert len(pal.team) == 2 and pal.team[0] != pal.team[1]
    assert pal.player[0] != pal.player[2]
    assert all(d != p for d, p in zip(pal.dead, pal.player, strict=True))
    assert rgb_to_hex(hex_to_rgb("#3987e5")) == "#3987e5"
    assert mix((0, 0, 0), (255, 255, 255), 0.5) == (128, 128, 128)
    assert shade((100, 100, 100), -1.0) == (0, 0, 0)
    assert label_ink((255, 255, 0)) != label_ink((0, 0, 80))
    light = make_palette([0, 1], dark=False)
    assert light.surface == hex_to_rgb("#fcfcfb")


def test_render_frame() -> None:
    cfg = make_config(6, 5, 2, placement={"positions": [[1, 1], [4, 3]]})
    st = GameState(cfg)
    st.apply(1)  # J0 en (2, 1)
    pal = make_palette(cfg.team_index)
    img = render_frame(st, pal, cell=10, labels=False)
    assert img.shape == (50, 60, 3) and img.dtype == np.uint8
    assert tuple(img[15, 15]) == pal.player[0]  # mur (1, 1)
    assert tuple(img[10, 20]) == pal.ink  # anneau de la tête (2, 1)
    assert tuple(img[45, 5]) == pal.surface
    owners = np.full((5, 6), 1)
    over = render_frame(st, pal, cell=10, owners=owners, labels=False)
    assert tuple(over[45, 5]) == pal.territory[1]
    st.apply(-1)  # J1 éliminé
    dead = render_frame(st, pal, cell=10, labels=False)
    assert tuple(dead[30, 40]) == pal.ink  # croix sur la tête éliminée
    labelled = render_frame(st, pal, cell=16)
    assert labelled.shape == (80, 96, 3)
    tiny = render_frame(st, pal, cell=3)
    assert tiny.shape == (15, 18, 3)


def test_live_controller() -> None:
    ctl = PlaybackController(LiveSource(_config()), speed=10)
    assert ctl.speed == 10
    assert ctl.update(0.35) == 3
    ctl.toggle_pause()
    assert ctl.update(1.0) == 0
    assert ctl.step_forward() and ctl.source.state.ply == 4
    assert not ctl.step_backward()
    for _ in range(20):
        ctl.faster()
    assert ctl.speed == 1000
    for _ in range(20):
        ctl.slower()
    assert ctl.speed == 0.5
    panel = ctl.panel()
    assert panel.ply == 4 and len(panel.players) == 4 and len(panel.teams) == 2
    assert sum(p.share for p in panel.players) == pytest.approx(1.0)
    assert any(p.nodes is not None for p in panel.players)
    assert ctl.owners() is None
    ctl.toggle_voronoi()
    owners = ctl.owners()
    assert owners is not None and owners.shape == (10, 10)
    seed0 = ctl.source.config.seed
    ctl.new_game()
    assert ctl.source.config.seed != seed0 and ctl.source.config.first_player == 1
    assert ctl.source.state.ply == 0 and "partie 2" in ctl.source.title
    ctl.paused = False
    ctl.speed_index = len(ctl_speeds()) - 1
    while not ctl.source.finished:
        ctl.update(1.0)
    assert "Victoire" in ctl.panel().status


def ctl_speeds() -> tuple[float, ...]:
    from tron.viz.controller import SPEEDS

    return SPEEDS


def test_replay_controller(log_path: Path) -> None:
    log = GameLog.load(log_path)
    src = ReplaySource(log)
    ctl = PlaybackController(src, paused=True)
    assert src.can_go_back is False
    assert ctl.step_forward() and ctl.step_forward()
    key_after_two = src.state.key
    assert ctl.step_backward() and src.index == 1
    ctl.step_forward()
    assert src.state.key == key_after_two
    src.seek(len(log.moves))
    assert src.finished and not ctl.step_forward()
    final = log.final_state()
    assert np.array_equal(src.state.grid, final.grid)
    panel = ctl.panel()
    assert all(p.rank is not None for p in panel.players)
    ctl.new_game()
    assert src.index == 0 and src.state.ply == 0
    assert "Rejeu" in src.title and src.agent_names


def test_exports(tmp_path: Path, log_path: Path) -> None:
    import imageio.v3 as iio

    log = GameLog.load(log_path)
    frames = list(log_frames(log, cell=6, every=5, voronoi=True, hold_end=2))
    assert len(frames) >= 3 and frames[0].shape == (60, 60, 3)
    gif = export_animation(frames, tmp_path / "partie.gif", fps=12)
    # Pillow fusionne les images identiques consécutives (pause finale) en une seule.
    assert len(frames) - 2 <= iio.imread(gif, index=None).shape[0] <= len(frames)
    png = save_png(frames[-1], tmp_path / "derniere.png")
    assert iio.imread(png).shape[:2] == (60, 60)
    with pytest.raises(ValueError):
        export_animation(frames, tmp_path / "partie.avi")
    mp4 = export_animation(frames, tmp_path / "partie.mp4", fps=12)
    assert mp4.exists() and mp4.stat().st_size > 0


# --------------------------------------------- fenêtre arcade (bouchon d'API)
class _Stub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.x = args[1] if len(args) > 1 else 0
        self.y = args[2] if len(args) > 2 else 0
        self.text = args[0] if args else ""

    def draw(self) -> None:
        pass


class _SpriteList(list):
    def draw(self) -> None:
        pass


class _Window:
    def __init__(self, width: int, height: int, title: str, update_rate: float = 1 / 60) -> None:
        self.width, self.height, self.title = width, height, title
        self.closed = False

    def clear(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Image:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"png")


def _stub_arcade() -> types.ModuleType:
    mod = types.ModuleType("arcade")
    mod.Window = _Window  # type: ignore[attr-defined]
    mod.SpriteList = _SpriteList  # type: ignore[attr-defined]
    mod.SpriteSolidColor = _Stub  # type: ignore[attr-defined]
    mod.Text = _Stub  # type: ignore[attr-defined]
    calls: list[str] = []
    mod.calls = calls  # type: ignore[attr-defined]
    mod.draw_lrbt_rectangle_filled = lambda *a, **k: calls.append("rect")  # type: ignore[attr-defined]
    mod.draw_lrbt_rectangle_outline = lambda *a, **k: calls.append("outline")  # type: ignore[attr-defined]
    mod.draw_line = lambda *a, **k: calls.append("line")  # type: ignore[attr-defined]
    mod.get_image = lambda *a, **k: _Image()  # type: ignore[attr-defined]
    mod.run = lambda: calls.append("run")  # type: ignore[attr-defined]
    keys = types.SimpleNamespace(
        SPACE=1, RIGHT=2, LEFT=3, UP=4, DOWN=5, V=6, N=7, HOME=8, END=9, S=10, ESCAPE=11, Q=12
    )
    mod.key = keys  # type: ignore[attr-defined]
    return mod


def test_arcade_window_with_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, log_path: Path
) -> None:
    stub = _stub_arcade()
    monkeypatch.setitem(sys.modules, "arcade", stub)
    app = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    try:
        assert app.auto_cell(20, 20) == 36 and app.auto_cell(100, 50) == 8
        ctl = PlaybackController(LiveSource(_config()), speed=20)
        win = app.TronWindow(ctl, cell=16, screenshot_dir=tmp_path / "shots")
        assert len(win.cells) == 100
        win.on_update(0.5)
        win.on_draw()
        assert "outline" in stub.calls and "rect" in stub.calls
        k = stub.key
        for symbol in (k.SPACE, k.RIGHT, k.LEFT, k.UP, k.DOWN, k.V, k.N, k.HOME, k.END):
            win.on_key_press(symbol, 0)
            win.on_draw()
        win.on_key_press(k.S, 0)
        assert len(list((tmp_path / "shots").glob("*.png"))) == 1
        win.on_key_press(k.ESCAPE, 0)
        assert win.closed
        rep = PlaybackController(ReplaySource(GameLog.load(log_path)), paused=True)
        win2 = app.TronWindow(rep, cell=8)
        win2.on_key_press(k.END, 0)
        win2.on_draw()
        win2.on_key_press(k.LEFT, 0)
        win2.on_key_press(k.HOME, 0)
        win2.on_draw()
        assert "line" in stub.calls
        app.run_replay(GameLog.load(log_path), cell=8)
        app.run_live(_config(), cell=8)
        assert stub.calls.count("run") == 2
    finally:
        monkeypatch.delitem(sys.modules, "arcade")
        importlib.reload(app)
