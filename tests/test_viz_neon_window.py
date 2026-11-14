"""Fenêtres néon (temps réel, rejeu, comparaison) avec un bouchon d'API Arcade.

Les tests vérifient la logique : sprites des traces synchronisés avec la scène,
modes de mélange, touches, journal, tremblement, bloom et son repli. Le module
``arcade_preview`` dessine réellement les appels (aperçu simulé) pour contrôler
la mise en page sans écran ; la validation sur un poste graphique reste
décrite dans ``docs/validation-gui.md``.
"""

from __future__ import annotations

import importlib
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import arcade_preview
import arcade_stub
from conftest import make_config
from tron.core.config import GameConfig
from tron.sim import GameLog, Match, Recorder
from tron.viz.controller import LiveSource, PlaybackController, ReplaySource
from tron.viz.effects.bloom import Bloom, shader_source
from tron.viz.theme import QUALITIES

ROOT = Path(__file__).resolve().parents[1]
ADDITIVE = arcade_stub.Context.BLEND_ADDITIVE


def _config(seed: int = 3) -> GameConfig:
    return make_config(
        10, 10, 4, teams=["A", "B", "A", "B"], agents=["greedy", "random", "greedy", "random"],
        seed=seed, placement={"mode": "regular", "rotate": True},
    )  # fmt: skip


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    path = tmp_path / "partie.jsonl"
    Match(_config(), observers=[Recorder(path)]).run()
    return path


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    stub = arcade_stub.install(monkeypatch)
    module = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    module.stub = stub  # type: ignore[attr-defined]
    try:
        yield module
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def _trail_matches_scene(win: object) -> None:
    arena = win.arena  # type: ignore[attr-defined]
    tracker = arena.tracker
    assert len(arena.glow) == len(arena.core) == sum(len(p) for p in tracker.paths)
    for p, items in enumerate(arena._trail):
        assert [cell for cell, *_ in items] == tracker.paths[p]
        assert [mask for _, mask, *_ in items] == [tracker.mask(p, i) for i in range(len(items))]
        for cell, mask, glow, core in items:
            assert (glow.center_x, glow.center_y) == arena.center(*cell)
            assert glow.texture is arena._glow_tex(mask) and core.texture is arena._core_tex(mask)


def test_neon_live_window(app: types.ModuleType, tmp_path: Path) -> None:
    stub = app.stub  # type: ignore[attr-defined]
    ctl = PlaybackController(LiveSource(_config()), speed=10)
    win = app.TronWindow(ctl, cell=16, screenshot_dir=tmp_path / "shots", theme="neon")
    assert win.neon and len(win.cells) == 100 and len(win.arena.bikes) == 4
    assert win.width == 10 * 16 + 2 * win.margin + app.PANEL_WIDTH and win.height >= 640
    for _ in range(12):
        win.on_update(0.1)
        win.on_draw()
    _trail_matches_scene(win)
    arena = win.arena
    assert arena.glow.draws[-1] == {"blend_function": ADDITIVE}
    assert arena.halos.draws[-1] == {"blend_function": ADDITIVE}
    assert arena.core.draws[-1] == {} and arena.background.draws
    assert win.hud.t_title.text == "TRON // COALITIONS"
    assert win.hud.t_status.text.startswith("Lecture") and "Demi-coup" in win.hud.t_ply.text
    assert {"rect", "outline", "line"} <= set(stub.calls)
    k = stub.key
    win.on_key_press(k.V, 0)
    win.on_draw()
    assert any(s.color[3] == 40 for s in win.cells)  # territoires de Voronoi teintés
    assert "lines" in stub.calls  # frontières
    win.on_key_press(k.D, 0)
    win.on_key_press(k.A, 0)
    win.on_draw()
    assert win.show_team_links and any("Décision" in t.text for t in win.t_decision)
    win.on_key_press(k.S, 0)
    assert len(list((tmp_path / "shots").glob("*.png"))) == 1
    win.on_key_press(k.ESCAPE, 0)
    assert win.closed


def test_neon_keys_quality_theme_and_bloom(app: types.ModuleType) -> None:
    stub = app.stub  # type: ignore[attr-defined]
    ctl = PlaybackController(LiveSource(_config()), speed=10)
    win = app.TronWindow(ctl, cell=12, theme="neon", quality="medium")
    k = stub.key
    for _ in range(5):
        win.on_update(0.1)
    win.on_key_press(k.G, 0)
    assert win.quality.name == "ultra" and win.bloom_on
    win.on_update(0.05)
    win.on_draw()
    assert win.bloom is not None and win.bloom.enabled
    assert "use:fbo" in win.ctx.log and "use:screen" in win.ctx.log
    _trail_matches_scene(win)
    win.on_key_press(k.B, 0)
    assert not win.bloom_on
    win.on_key_press(k.G, 0)
    assert win.quality.name == "low" and not win.bloom_on
    win.on_draw()
    _trail_matches_scene(win)
    bank = win._bank
    textures = len(bank)
    win.on_key_press(k.G, 0)
    win.on_key_press(k.G, 0)
    win.on_key_press(k.G, 0)  # retour à « low » : textures réutilisées, l'atlas ne grossit pas
    win.on_draw()
    assert win.quality.name == "low" and len(bank) <= textures + 40
    win.on_key_press(k.T, 0)  # thème classique (palette validée pour les daltoniens)
    assert not win.neon and len(win.cells) == 100
    win.on_draw()
    win.on_key_press(k.T, 0)
    assert win.neon
    win.on_draw()
    _trail_matches_scene(win)
    win.on_key_press(k.N, 0)  # nouvelle partie : scène et journal remis à zéro
    win.on_update(0.0)
    win.on_draw()
    assert len(win.arena.glow) == 4 and not win.event_log.recent()


def test_neon_replay_log_rewind_and_effects(app: types.ModuleType, log_path: Path) -> None:
    stub = app.stub  # type: ignore[attr-defined]
    log = GameLog.load(log_path)
    ctl = PlaybackController(ReplaySource(log), paused=True)
    win = app.TronWindow(ctl, cell=16, theme="neon")
    k = stub.key
    win.on_draw()
    first_elim = next(i for i, rec in enumerate(log.moves) if rec["elim"])
    for _ in range(first_elim + 1):  # pas à pas jusqu'à la première élimination
        win.on_key_press(k.RIGHT, 0)
        win.on_update(1 / 60)
    arena = win.arena
    assert arena.effects.particles.count > 0 and arena.effects.active_rings()
    assert win._camera is not None
    win.on_draw()
    assert "circle_outline" in stub.calls and win._camera.used >= 1  # onde de choc et tremblement
    assert arena.particles.draws[-1] == {"blend_function": ADDITIVE}
    texts = [line.text for line in win.event_log.recent(20)]
    assert any("ÉLIMINÉ" in t for t in texts)
    win.on_key_press(k.END, 0)
    win.on_update(1 / 60)
    win.on_draw()
    texts = [line.text for line in win.event_log.recent(20)]
    assert texts[-1].startswith(("VICTOIRE", "ÉGALITÉ"))
    assert win.hud.t_cards[0][1].text in ("VAINQUEUR", "VAINCU", "RANG 2", "RANG 3", "RANG 4")
    _trail_matches_scene(win)
    win.on_key_press(k.LEFT, 0)
    win.on_update(1 / 60)
    assert not any(t.startswith("VICTOIRE") for t in (ln.text for ln in win.event_log.recent(20)))
    assert arena.effects.particles.count == 0  # retour arrière : effets effacés
    _trail_matches_scene(win)
    win.on_key_press(k.HOME, 0)
    win.on_update(1 / 60)
    win.on_draw()
    assert len(arena.glow) == 4 and not win.event_log.recent()
    _trail_matches_scene(win)


def test_bloom_pipeline_and_fallback(app: types.ModuleType) -> None:
    ctl = PlaybackController(LiveSource(_config()), speed=10)
    win = app.TronWindow(ctl, cell=12, theme="neon", quality="ultra")
    win.on_update(0.2)
    win.on_draw()
    bloom = win.bloom
    assert bloom is not None and bloom.enabled and bloom.error is None
    programs = win.ctx.programs
    assert len(programs) == 3 and programs[0]["source"] == 0 and programs[0]["threshold"] == 0.55
    assert programs[2]["scene"] == 0 and programs[2]["bloom"] == 1
    assert len(bloom.quad.renders) == 1 + 2 * bloom.passes + 1
    assert win.ctx.log[-1] == "use:screen" or "use:screen" in win.ctx.log
    # contexte OpenGL défaillant : le bloom se désactive et le rendu continue
    win2 = app.TronWindow(ctl, cell=12, theme="neon", quality="ultra")
    win2.ctx.fail = True
    win2.on_draw()
    assert win2.bloom is not None and not win2.bloom.enabled and "RuntimeError" in win2.bloom.error
    win2.on_draw()
    rows = [t[1].text for t in win2.hud.t_stats]
    assert "ultra" in rows and "ultra · bloom" not in rows
    assert Bloom(win2).end() is False  # rien à composer sans begin()


def test_shader_sources_match_uniforms() -> None:
    vertex = shader_source("fullscreen.vert")
    assert "in_vert" in vertex and "in_uv" in vertex
    for name, uniforms in {
        "bloom_extract.frag": ("source", "threshold"),
        "blur.frag": ("source", "direction"),
        "bloom_combine.frag": ("scene", "bloom", "intensity"),
    }.items():
        text = shader_source(name)
        assert text.startswith("#version 330") and "fragColor" in text
        for uniform in uniforms:
            assert "uniform" in text and f" {uniform};" in text


def test_neon_compare_window(
    monkeypatch: pytest.MonkeyPatch, log_path: Path, tmp_path: Path
) -> None:
    stub = arcade_stub.install(monkeypatch)
    arcade_app = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    capp = importlib.reload(importlib.import_module("tron.viz.comparison_app"))
    try:
        from tron.viz.comparison import ComparisonSession

        other = tmp_path / "autre.jsonl"
        Match(_config(seed=8), observers=[Recorder(other)]).run()
        session = ComparisonSession(GameLog.load(log_path), GameLog.load(other))
        win = capp.CompareWindow(session, cell=10, theme="neon", quality="low")
        assert win.boards is not None and len(win.arenas) == 2
        assert len(win.boards[0].cells) == 100
        k = stub.key
        win.on_draw()
        win.on_key_press(k.SPACE, 0)
        win.on_update(0.5)
        win.on_draw()
        assert session.ply == 5
        for symbol in (k.V, k.END, k.LEFT, k.HOME, k.END):
            win.on_key_press(symbol, 0)
            win.on_update(1 / 60)
            win.on_draw()
        for side, arena in enumerate(win.arenas):
            assert arena.tracker.paths[0][-1] == session.sources[side].state.head(0)
        assert "outline" in stub.calls
        capp.run_compare(session, cell=10, theme="neon")
        assert stub.calls.count("run") == 1
    finally:
        monkeypatch.undo()
        importlib.reload(arcade_app)
        importlib.reload(capp)


def test_run_entry_points_with_theme(app: types.ModuleType, log_path: Path) -> None:
    stub = app.stub  # type: ignore[attr-defined]
    app.run_live(_config(), cell=8, theme="neon", quality="low")
    app.run_replay(GameLog.load(log_path), cell=8, theme="neon")
    assert stub.calls.count("run") == 2
    with pytest.raises(ValueError, match="thème"):
        app.run_live(_config(), theme="rose")


def test_simulated_preview_layout(monkeypatch: pytest.MonkeyPatch, log_path: Path) -> None:
    """Aperçu simulé : la fenêtre se dessine réellement (PIL) et sa mise en page tient."""
    preview = arcade_preview.install(monkeypatch)
    module = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    try:
        ctl = PlaybackController(ReplaySource(GameLog.load(log_path)), paused=True, voronoi=True)
        win = module.TronWindow(ctl, cell=20, theme="neon")
        ctl.source.seek(len(ctl.source.log.moves))
        win.on_update(1 / 60)
        win.on_draw()
        image = preview.canvas().image()
        assert image.shape == (win.height, win.width, 3)
        theme = win.neon_theme
        panel = image[win.height // 2, win.width - 20].astype(int)
        assert np.abs(panel - np.array(theme.panel[:3])).max() < 40  # fond du HUD
        arena = win.arena
        row = win.height - round(arena.bottom + arena.pixel_height / 2)
        edge = image[row, round(arena.left) - 2 : round(arena.left) + 2].astype(int)
        assert edge[:, 2].max() > 150  # liseré cyan de l'arène
        bottom = win.height - round(arena.bottom) + 4
        assert bottom < win.height and arena.bottom >= win.margin
        for text in win.hud.t_help:  # l'aide tient dans le panneau
            assert len(text.text) < 60
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_quality_presets_are_ordered() -> None:
    low, medium, ultra = (QUALITIES[q] for q in ("low", "medium", "ultra"))
    assert low.max_particles < medium.max_particles < ultra.max_particles
    assert not low.halo and medium.halo and ultra.bloom and not medium.bloom
