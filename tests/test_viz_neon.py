"""Thème néon sans affichage : thème, scène, tuiles, rendu NumPy, effets, HUD, exports."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import make_config
from tron.agents.context import AllianceEvent
from tron.core import GameState
from tron.core.config import GameConfig
from tron.core.directions import EAST, NORTH, PASS, SOUTH, WEST
from tron.sim import GameLog, Match, Recorder
from tron.viz import log_frames
from tron.viz.controller import LiveSource, PlaybackController, ReplaySource
from tron.viz.effects import EffectDirector, ParticleSystem, zone_color, zone_controllers
from tron.viz.hud import (
    EventLog,
    describe_event,
    fit,
    format_clock,
    hud_stats,
    player_cards,
    stamp,
)
from tron.viz.neon import (
    NeonRenderer,
    arena_background,
    bike_image,
    default_heading,
    frame_size,
    line_peak,
    neon_frames,
    obstacle_tile,
    render_neon,
    soft_dot,
    stroke_mask,
    trail_tiles,
)
from tron.viz.scene import (
    EAST_BIT,
    NORTH_BIT,
    SOUTH_BIT,
    WEST_BIT,
    PlyExtras,
    SceneTracker,
    VisualEvent,
    alliance_pairs,
    arena_stats,
    heading_between,
    opposite,
    path_mask,
    record_extras,
    team_links,
    territory_edges,
)
from tron.viz.theme import (
    NEON_COLORS,
    QUALITIES,
    arena_padding,
    major_step,
    make_neon_theme,
    next_quality,
    pulse,
    resolve_quality,
    resolve_theme,
    team_hues,
    trail_metrics,
    with_alpha,
)

ROOT = Path(__file__).resolve().parents[1]


def _duel(width: int = 6, height: int = 6) -> GameConfig:
    return make_config(width, height, 2, agents="greedy")


def _scripted() -> tuple[GameState, list[int]]:
    """Deux joueurs en (1, 1) et (4, 4) ; coups alternés, avec virages."""
    st = GameState(_duel(), positions=[(1, 1), (4, 4)])
    dirs = [EAST, WEST, SOUTH, NORTH, SOUTH, NORTH]
    for d in dirs:
        st.apply(d)
    return st, dirs


def _extras(dirs: list[int]):
    return lambda ply: PlyExtras(direction=dirs[ply] if ply < len(dirs) else None)


# thème
def test_theme_colors_and_quality() -> None:
    theme = make_neon_theme([0, 1, 0, 1])
    assert len(theme.team) == 2 and theme.team[0] != theme.team[1]
    assert theme.player[0] != theme.player[2]  # nuances d'une même équipe
    for p in range(4):
        assert sum(theme.dead[p]) < sum(theme.player[p])  # trace éteinte plus sombre
        assert sum(theme.core[p]) > sum(theme.player[p])  # cœur plus clair
    assert theme.trail_color(0, True) == theme.player[0]
    assert theme.trail_color(0, False, 1.0) == theme.dead[0]
    assert theme.trail_color(0, False, 0.0) == theme.player[0]
    assert theme.territory(1, 40)[3] == 40 and with_alpha((1, 2, 3), 2.0) == (1, 2, 3, 255)
    assert resolve_quality("ULTRA").bloom and not resolve_quality("medium").bloom
    assert resolve_quality(QUALITIES["low"]) is QUALITIES["low"]
    with pytest.raises(ValueError, match="qualité"):
        resolve_quality("max")
    assert resolve_theme("Neon") == "neon"
    with pytest.raises(ValueError, match="thème"):
        resolve_theme("rose")
    names = [next_quality(QUALITIES[q]).name for q in ("low", "medium", "ultra")]
    assert names == ["medium", "ultra", "low"]
    assert pulse(0.0) == 1.0 and pulse(1.0, enabled=False) == 1.0
    assert all(0.8 - 1e-9 <= pulse(t / 10) <= 1.2 + 1e-9 for t in range(100))
    assert major_step(20, 20) == 5 and major_step(40, 20) == 10
    assert arena_padding(4) == 6 and arena_padding(40) == 30


def test_team_hues_follow_color_names() -> None:
    assert team_hues(["Rouge", "Bleu", "Vert", "Jaune"]) == [5, 7, 3, 6]
    assert team_hues(["A", "B"]) == [0, 1]
    assert team_hues(["Bleu", "A", "B"]) == [7, 0, 1]  # pas de doublon de teinte
    assert team_hues(["Rouge", "red"]) == [5, 0]
    assert team_hues([], 3) == [0, 1, 2]
    theme = make_neon_theme([0, 1], ["Vert", "Rose"])
    assert theme.team == (
        tuple(int(NEON_COLORS[3][i : i + 2], 16) for i in (1, 3, 5)),
        tuple(int(NEON_COLORS[1][i : i + 2], 16) for i in (1, 3, 5)),
    )


def test_trail_metrics_scale_with_cell() -> None:
    small, big = trail_metrics(8), trail_metrics(32)
    assert big.body > small.body and big.core < big.body
    for m in (small, big, trail_metrics(20, "low")):
        assert m.margin >= 3 * max(m.sigma_tight, m.sigma_wide)
        assert m.tile == m.cell + 2 * m.margin
    low = trail_metrics(20, "low")
    assert low.sigma_wide == 0 and low.gain_wide == 0 and low.tile < trail_metrics(20).tile


# scène
def test_tracker_paths_masks_and_events() -> None:
    st, dirs = _scripted()
    tracker = SceneTracker()
    result = tracker.sync(st, _extras(dirs))
    assert result.reset and len(result.added) == 6
    assert tracker.paths[0] == [(1, 1), (2, 1), (2, 2), (2, 3)]
    assert tracker.paths[1] == [(4, 4), (3, 4), (3, 3), (3, 2)]
    assert tracker.mask(0, 0) == EAST_BIT
    assert tracker.mask(0, 1) == WEST_BIT | SOUTH_BIT
    assert tracker.mask(0, 2) == NORTH_BIT | SOUTH_BIT
    assert tracker.mask(0, 3) == NORTH_BIT
    assert tracker.headings == [SOUTH, NORTH] and tracker.head(1) == (3, 2)
    kinds = [ev.kind for ev in result.events]
    assert kinds == ["move", "move", "turn", "turn", "move", "move"]
    assert tracker.sync(st, _extras(dirs)).changed is False
    st.undo()
    st.undo()
    back = tracker.sync(st)
    assert back.rewound and len(back.removed) == 2 and not back.events
    assert tracker.paths[0][-1] == (2, 2) and tracker.headings == [SOUTH, NORTH]
    st.undo()
    tracker.sync(st)
    assert tracker.headings == [SOUTH, WEST]  # le 4e coup (1 vers le nord) est défait
    assert heading_between((0, 0), (5, 5)) == -1 and opposite(-1) == -1
    assert path_mask([(0, 0)], 0) == 0


def test_tracker_elimination_causes_and_victory() -> None:
    cfg = make_config(5, 5, 3, agents="greedy")
    # 0 sort par la bordure, 1 heurte la trace de 2, 2 passe son tour (bloqué)
    st = GameState(cfg, positions=[(0, 0), (2, 2), (3, 2)])
    dirs = [WEST, EAST]
    tracker = SceneTracker()
    for d in dirs:
        st.apply(d)
    events = tracker.sync(st, _extras(dirs)).events
    elim = [ev for ev in events if ev.kind == "elimination"]
    assert [(e.player, e.cause, e.other) for e in elim] == [(0, "bordure", -1), (1, "trace", 2)]
    assert elim[0].point == (0.0, 0.5)
    assert any(ev.kind == "victory" and ev.other == 2 for ev in events)
    assert tracker.alive == [False, False, True]
    st2 = GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    st2.apply(EAST)
    st2.apply(PASS)
    st2.apply(NORTH)
    st2.apply(WEST)  # 0 revient sur sa propre trace
    dirs2 = [EAST, PASS, NORTH, WEST]
    tr2 = SceneTracker()
    causes = [e.cause for e in tr2.sync(st2, _extras(dirs2)).events if e.kind == "elimination"]
    assert causes == ["bloqué", "propre trace"]
    unknown = SceneTracker().sync(st2, lambda ply: PlyExtras())
    assert [e.cause for e in unknown.events if e.kind == "elimination"] == ["", ""]


def test_tracker_reset_and_grid_mode() -> None:
    st, _ = _scripted()
    tracker = SceneTracker()
    tracker.sync(st, events=False)
    other = GameState(_duel(), positions=[(0, 0), (5, 5)])
    assert tracker.sync(other).reset and tracker.paths == [[(0, 0)], [(5, 5)]]
    copy = st.copy(history=False)  # historique perdu : cases lues sur la grille
    result = tracker.sync(copy)
    assert result.reset and tracker.grid_mode and not result.events
    assert sorted(tracker.paths[0]) == sorted([(1, 1), (2, 1), (2, 2), (2, 3)])
    assert tracker.paths[0][-1] == copy.head(0)
    i = tracker.paths[0].index((2, 1))
    assert tracker.mask(0, i) == WEST_BIT | SOUTH_BIT
    copy.apply(SOUTH)
    tracker.sync(copy)
    assert tracker.grid_mode and copy.head(0) in tracker.paths[0]
    text = GameState.from_text(_duel(5, 5), "Aa...\n.....\n.....\n.....\n...bB")
    tr = SceneTracker()
    tr.sync(text)
    assert tr.grid_mode and len(tr.paths[0]) == 2


def test_state_history_is_read_only_view() -> None:
    st, dirs = _scripted()
    assert len(st.history) == len(dirs) == st.ply
    p, h, t, _ = st.history[0]
    assert p == 0 and st.coords(h) == (1, 1) and st.coords(t) == (2, 1)
    assert st.copy(history=False).history == []


def test_alliances_links_edges_and_stats() -> None:
    events = [
        AllianceEvent(1, 0, 1, "formed"),
        AllianceEvent(2, 2, 3, "formed"),
        AllianceEvent(3, 1, 2, "formed"),
        AllianceEvent(4, 3, 2, "broken"),
        AllianceEvent(5, 1, 0, "dissolved"),
        AllianceEvent(6, 0, 3, "proposed"),
    ]
    assert alliance_pairs(events[:3]) == ((0, 1), (1, 2), (2, 3))
    assert alliance_pairs(events) == ()
    cfg = make_config(6, 6, 4, teams=["A", "B", "A", "B"])
    st = GameState(cfg)
    assert team_links(st) == ((0, 2), (1, 3))
    owners = np.array([[0, 0, 1], [0, -1, 1]])
    edges = territory_edges(owners)
    assert (2.0, 0.0, 2.0, 1.0) in edges and (1.0, 1.0, 2.0, 1.0) in edges
    assert len(edges) == 4
    stats = arena_stats(st, alliances=2)
    assert stats.active == 4 and stats.alliances == 2 and math.isclose(stats.occupied, 4 / 36)
    rec = {
        "diplomacy": [{"ply": 1, "player": 0, "target": 1, "kind": "formed"}, {"bad": 1}],
        "zone": [[0, 1]],
        "obstacles": [[2, 3]],
    }
    extras = record_extras(rec, EAST)
    assert extras.direction == EAST and extras.zone == ((0, 1),) and extras.obstacles == ((2, 3),)
    assert [e.kind for e in extras.diplomacy] == ["formed"]


# tuiles
def test_stroke_tiles_are_disjoint_and_sum_to_a_line() -> None:
    cell, width, margin = 10, 4.0, 10
    size = cell + 2 * margin
    canvas = np.zeros((size, size + 2 * cell))
    for i, mask in enumerate([EAST_BIT, WEST_BIT | EAST_BIT, WEST_BIT]):
        canvas[:, i * cell : i * cell + size] += stroke_mask(size, margin, cell, mask, width)
    line = np.zeros_like(canvas)
    line[13:17, 13:37] = 1.0  # centres des cases en x = 15, 25, 35 ; bouts carrés de ±2
    assert np.allclose(canvas, line) and canvas.max() <= 1.0 + 1e-9
    corner = stroke_mask(10, 0, 10, NORTH_BIT | EAST_BIT, 2.0)
    assert math.isclose(corner.sum(), 2 * 2 + 4 * 2 + 4 * 2)


def test_glow_tiles_are_linear_and_symmetric() -> None:
    metrics = trail_metrics(12)
    tiles = trail_tiles(metrics)
    assert trail_tiles(metrics) is tiles  # mis en cache
    s, m, c = metrics.tile, metrics.margin, metrics.cell
    horiz = tiles.glow[EAST_BIT | WEST_BIT]
    assert np.allclose(np.rot90(horiz), tiles.glow[NORTH_BIT | SOUTH_BIT], atol=1e-6)
    assert all(t.min() >= 0.0 and t.max() <= 1.0 for t in tiles.glow)
    # une longue trace droite : la somme des tuiles culmine à 1 au centre
    n = 9
    canvas = np.zeros((s, (n - 1) * c + s), dtype=np.float64)
    for i in range(n):
        canvas[:, i * c : i * c + s] += horiz
    mid = canvas[:, (n // 2) * c + m : (n // 2) * c + m + c]
    assert 0.97 < mid.max() <= 1.05
    assert line_peak(metrics) > metrics.gain_body
    assert tiles.core[0].sum() == pytest.approx(metrics.core**2, rel=0.05)


def test_bike_orientation_and_assets() -> None:
    theme = make_neon_theme([0, 1])
    for heading, (dy, dx) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
        img = bike_image(theme.player[0], theme.head[0], 20, heading)
        assert img.shape == (40, 40, 4) and img.dtype == np.uint8
        white = (img[..., :3].astype(int).sum(axis=2) > 740) & (img[..., 3] > 200)
        ys, xs = np.nonzero(white)
        assert ys.size, "phare absent"
        cy, cx = ys.mean() - 19.5, xs.mean() - 19.5
        assert cy * dy + cx * dx > 5  # le phare est à l'avant
    dark = bike_image(theme.dead[0], theme.dead[0], 20, 1, lit=False)
    assert not ((dark[..., :3].astype(int).sum(axis=2) > 740) & (dark[..., 3] > 200)).any()
    assert bike_image(theme.player[0], theme.head[0], 6, 0).shape == (12, 12, 4)
    dot = soft_dot(5)
    assert dot.shape == (11, 11) and dot[5, 5] == 1.0 and dot[0, 0] < 0.05
    tile = obstacle_tile(16, theme.obstacle, theme.obstacle_edge)
    assert tile.shape == (16, 16, 4) and tile[8, 8, 3] == 255


def test_arena_background() -> None:
    theme = make_neon_theme([0, 1])
    bg = arena_background(10, 8, 12, theme, QUALITIES["medium"])
    pad = arena_padding(12)
    assert bg.shape == (8 * 12 + 2 * pad, 10 * 12 + 2 * pad, 3)
    border = bg[pad + 40, pad - 1].astype(int)
    inner = bg[pad + 40, pad + 30].astype(int)
    assert border[2] > 150 and border.sum() > inner.sum() + 200  # liseré cyan lumineux
    low = arena_background(10, 8, 12, theme, QUALITIES["low"])
    assert low.shape == bg.shape and not np.array_equal(low, bg)


# rendu NumPy
def test_render_neon_colors_and_size() -> None:
    st, _ = _scripted()
    img = render_neon(st, cell=20, labels=False)
    pad = arena_padding(20)
    assert img.shape == (*frame_size(st.config, 20), 3) == (6 * 20 + 2 * pad,) * 2 + (3,)
    theme = make_neon_theme(st.config.team_index)
    core = img[pad + 1 * 20 + 10, pad + 2 * 20 + 10].astype(int)  # trace de 0 en (2, 1)
    assert np.abs(core - np.array(theme.core[0])).max() < 40  # cœur blanc teinté
    empty = img[pad + 5 * 20 + 10, pad + 0 * 20 + 10].astype(int)
    assert empty.sum() < 150  # case libre : fond sombre
    assert np.array_equal(img, render_neon(st, cell=20, labels=False))  # déterministe


def test_incremental_render_matches_fresh_render() -> None:
    st, dirs = _scripted()
    renderer = NeonRenderer(st.config, cell=14, labels=False)
    for _ in range(3):
        st.undo()
    renderer.render(st)
    for d in dirs[3:]:
        st.apply(d)
        renderer.render(st)
    st.undo()
    incremental = renderer.render(st)
    fresh = NeonRenderer(st.config, cell=14, labels=False).render(st)
    assert np.abs(incremental.astype(int) - fresh.astype(int)).max() <= 1


def test_render_dead_player_dims_and_marks() -> None:
    cfg = make_config(5, 5, 3, agents="greedy")
    st = GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    st.apply(EAST)
    st.apply(PASS)  # 1 éliminé
    img = render_neon(st, cell=20, labels=True)
    alive = render_neon(GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)]), cell=20)
    pad = arena_padding(20)
    head = (slice(pad + 2 * 20 + 2, pad + 3 * 20 - 2), slice(pad + 2 * 20 + 2, pad + 3 * 20 - 2))
    assert img[head].astype(int).sum() != alive[head].astype(int).sum()
    theme = make_neon_theme(cfg.team_index)
    red = np.array(theme.danger)
    assert (np.abs(img[head].astype(int) - red).sum(axis=2) < 120).any()  # croix rouge


def test_render_overlays() -> None:
    cfg = GameConfig.from_yaml(ROOT / "configs/games/zone_control_16x16.yaml")
    match = Match(cfg)
    for _ in range(40):
        match.step()
    st = match.state
    renderer = NeonRenderer(cfg, cell=10)
    owners = np.full((st.height, st.width), -1)
    owners[:, : st.width // 2] = 0
    owners[:, st.width // 2 :] = 1
    plain = renderer.render(st)
    with_owners = renderer.render(st, owners=owners, path=[(0, 3, 3)], links=[(0, 1)], t=1.0)
    assert plain.shape == with_owners.shape and not np.array_equal(plain, with_owners)
    assert len(zone_controllers(st)) == len(cfg.rules.zones)
    theme = renderer.theme
    assert zone_color(st, theme, -1) == theme.text_dim
    assert zone_color(st, theme, 0) == theme.team[cfg.team_index[0]]
    assert default_heading(GameState(_duel(), positions=[(0, 0), (5, 5)]), 0) in (EAST, SOUTH)


def test_render_obstacles() -> None:
    cfg = GameConfig.from_yaml(ROOT / "configs/games/obstacles_14x14.yaml")
    match = Match(cfg)
    while match.state.n_obstacles == 0:
        match.step()
    img = NeonRenderer(cfg, cell=12).render(match.state)
    x, y = next((int(x), int(y)) for y, x in np.argwhere(match.state.grid2d() == -3))
    pad = arena_padding(12)
    pix = img[pad + y * 12 + 6, pad + x * 12 + 1].astype(int)
    assert pix[0] > pix[2]  # liseré orangé de l'obstacle


# effets
def test_particles_deterministic_and_bounded() -> None:
    a, b = ParticleSystem(10, seed=3), ParticleSystem(10, seed=3)
    for ps in (a, b):
        assert ps.emit(6, 1.0, 1.0, (255, 0, 0)) == 6
        assert ps.emit(6, 2.0, 2.0, [(0, 255, 0), (0, 0, 255)]) == 4
        ps.update(0.1)
    assert a.dropped == 2 and len(a) == 10
    va, vb = a.view(), b.view()
    assert np.array_equal(va.x, vb.x) and np.array_equal(va.color, vb.color)
    assert (va.alpha > 0).all() and (va.alpha <= 1).all()
    a.update(5.0)
    assert len(a) == 0 and len(a.view()) == 0
    assert ParticleSystem(0).emit(3, 0, 0, (1, 2, 3)) == 0


def _director_match(cfg: GameConfig) -> tuple[EffectDirector, SceneTracker, GameState, list[int]]:
    st = GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    theme = make_neon_theme(cfg.team_index)
    tracker = SceneTracker()
    tracker.sync(st)  # première synchronisation : état initial, sans effet
    return EffectDirector(theme, "medium", seed=1), tracker, st, []


def test_effect_director_elimination_rewind_and_fast_forward() -> None:
    cfg = make_config(5, 5, 3, agents="greedy")
    director, tracker, st, _ = _director_match(cfg)
    dirs = [EAST, PASS]
    for d in dirs:
        st.apply(d)
    director.handle(tracker.sync(st, _extras(dirs)), tracker, st, speed=5)
    assert len(director.particles) > 20 and director.active_rings()
    assert director.flash_alpha() == 1.0 and director.shake_offset() != (0.0, 0.0)
    assert director.fade(1) == 0.0 and director.busy
    director.update(0.3)
    assert 0.0 < director.fade(1) <= 1.0 and director.fade(0) == 0.0
    director.update(1.0)
    assert director.fade(1) == 1.0 and director.shake_offset() == (0.0, 0.0)
    st.undo()
    director.handle(tracker.sync(st), tracker, st)
    assert len(director.particles) == 0 and 1 not in director.deaths
    fast = EffectDirector(director.theme, "medium")
    tr = SceneTracker()
    big = GameState(make_config(10, 10, 2, agents="greedy"), positions=[(0, 0), (9, 9)])
    tr.sync(big)
    for d in [EAST, WEST] * 8 + [PASS]:
        big.apply(d)
    fast.handle(tr.sync(big), tr, big, speed=1000)
    assert len(fast.particles) == 0 and fast.fade(0) == 1.0  # trop de demi-coups : pas d'effet
    calm = EffectDirector(director.theme, "medium")
    st3 = GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    tr3 = SceneTracker()
    tr3.sync(st3)
    st3.apply(EAST)
    st3.apply(PASS)
    calm.handle(tr3.sync(st3, _extras(dirs)), tr3, st3, speed=500)
    assert calm.flash_alpha() == 0.0 and calm.shake_offset() == (0.0, 0.0)  # trop rapide
    low = EffectDirector(director.theme, "low")
    tr4, st4 = SceneTracker(), GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    tr4.sync(st4)
    st4.apply(EAST)
    st4.apply(PASS)
    low.handle(tr4.sync(st4, _extras(dirs)), tr4, st4, speed=1)
    assert low.flash_alpha() == 0.0 and 0 < len(low.particles) < len(director.theme.player) * 40
    low.set_quality("ultra")
    assert low.particles.capacity == QUALITIES["ultra"].max_particles


def test_effect_director_other_events() -> None:
    cfg = GameConfig.from_yaml(ROOT / "configs/games/zone_control_16x16.yaml")
    st = GameState(cfg)
    director = EffectDirector(make_neon_theme(cfg.team_index), "ultra")
    tracker = SceneTracker()
    tracker.sync(st)
    from tron.viz.scene import SyncResult

    events = [
        VisualEvent("alliance", 0, player=0, other=1),
        VisualEvent("rupture", 0, player=0, other=1),
        VisualEvent("zone", 0, other=0, index=0),
        VisualEvent("zone", 0, other=0, index=99),
        VisualEvent("obstacles", 0, cells=((3, 3), (4, 4))),
        VisualEvent("alliance", 0, player=0, other=99),
    ]
    director.handle(SyncResult(events=events), tracker, st)
    assert len(director.active_links()) == 1 and len(director.active_rings()) == 3
    link = director.active_links()[0]
    assert link.progress(director.time) == 0.0 and link.alpha(director.time) == 1.0
    director.update(2.0)
    assert not director.active_links() and not director.active_rings()


# HUD
def test_hud_cards_stats_and_log() -> None:
    cfg = make_config(5, 5, 3, agents="greedy", teams=["A", "B", "A"])
    st = GameState(cfg, positions=[(0, 0), (2, 2), (4, 4)])
    ctl = PlaybackController(LiveSource(cfg), paused=True)
    info = ctl.panel()
    theme = make_neon_theme(cfg.team_index, cfg.team_names)
    cards = player_cards(info, theme, alliances=[(0, 2)])
    assert [c.badge for c in cards] == ["ACTIF"] * 3
    assert "allié de 2" in cards[0].detail and cards[info.current].current
    end = player_cards(info, theme, winners=[0, 2])
    assert [c.badge for c in end] == ["VAINQUEUR", "VAINCU", "VAINQUEUR"]
    rows = dict(hud_stats(st, alliances=1, elapsed=75.0, speed=10))
    assert rows["joueurs actifs"] == "3 / 3" and rows["temps"] == "01:15"
    assert rows["vitesse"] == "×10" and rows["alliances"] == "1"
    assert format_clock(3725) == "62:05" and fit("abcdef", 4) == "abc…" and fit("ab", 4) == "ab"
    log = EventLog(size=2)
    log.add(
        [
            VisualEvent("move", 1, player=0),
            VisualEvent("elimination", 3, player=1, cause="trace", other=0),
            VisualEvent("alliance", 4, player=0, other=2),
            VisualEvent("victory", 5, other=0),
        ],
        st,
        theme,
    )
    texts = [line.text for line in log.recent(5)]
    assert texts == ["JOUEUR 1 (J1) ÉLIMINÉ · trace de 0", "ALLIANCE 0 + 2", "VICTOIRE · A"]
    assert len(log.recent()) == 2 and log.recent(0) == []
    assert stamp(log.recent(1)[0]) == "[005] VICTOIRE · A"
    log.prune(4)
    assert [line.ply for line in log.recent(5)] == [3]
    log.clear()
    assert not log.recent()


def test_describe_every_event_kind() -> None:
    cfg = GameConfig.from_yaml(ROOT / "configs/games/zone_control_16x16.yaml")
    st = GameState(cfg)
    theme = make_neon_theme(cfg.team_index)
    cases = {
        VisualEvent("elimination", 1, player=0, cause="bordure"): "· bordure",
        VisualEvent("elimination", 1, player=0, cause="propre trace"): "sa propre trace",
        VisualEvent("elimination", 1, player=0): "ÉLIMINÉ",
        VisualEvent("rupture", 1, player=0, other=1): "RUPTURE 0 × 1",
        VisualEvent("zone", 1, other=0, index=1): "ZONE 2 → ",
        VisualEvent("zone", 1, other=-1, index=0): "ZONE 1 NEUTRE",
        VisualEvent("obstacles", 1, cells=((1, 1),)): "1 OBSTACLE",
        VisualEvent("obstacles", 1, cells=((1, 1), (2, 2))): "2 OBSTACLES",
        VisualEvent("victory", 1, other=-1): "ÉGALITÉ",
    }
    for ev, expected in cases.items():
        line = describe_event(ev, st, theme)
        assert line is not None and expected in line.text, (ev, line)
    assert describe_event(VisualEvent("turn", 1, player=0), st, theme) is None


# contrôleur, exports
@pytest.fixture
def neon_log(tmp_path: Path) -> Path:
    path = tmp_path / "neon.jsonl"
    cfg = make_config(
        8, 8, 4, teams=["A", "B", "A", "B"], agents=["greedy", "random", "greedy", "random"],
        seed=4,
    )  # fmt: skip
    Match(cfg, observers=[Recorder(path)]).run()
    return path


def test_controller_extras_and_alliances(neon_log: Path) -> None:
    log = GameLog.load(neon_log)
    ctl = PlaybackController(ReplaySource(log), paused=True)
    for _ in range(3):
        ctl.step_forward()
    assert ctl.ply_extras(0).direction == log.directions[0]
    assert ctl.ply_extras(10_000) == PlyExtras() and ctl.alliances() == ()
    cfg = GameConfig.from_yaml(ROOT / "configs/games/diplomacy_4p_20x20.yaml")
    live = PlaybackController(LiveSource(cfg))
    formed = False
    for _ in range(200):
        if not live.step_forward():
            break
        if live.alliances():
            formed = True
            break
    assert live.ply_extras(0).direction is not None
    assert live.ply_extras(-5) == PlyExtras()
    if formed:  # alliance formée : la paire est active dans l'état affiché
        assert all(0 <= a < b < cfg.n_players for a, b in live.alliances())


def test_replay_alliances_from_log(tmp_path: Path) -> None:
    cfg = GameConfig.from_yaml(ROOT / "configs/games/diplomacy_4p_20x20.yaml")
    path = tmp_path / "dipl.jsonl"
    Match(cfg, observers=[Recorder(path)]).run()
    log = GameLog.load(path)
    ctl = PlaybackController(ReplaySource(log), paused=True)
    seen = set()
    while ctl.step_forward():
        seen.update(ctl.alliances())
    expected = set()
    for i in range(len(log.moves)):
        expected.update(alliance_pairs(log.diplomacy_events()[: i + 1]))
    assert seen <= expected


def test_neon_frames_and_export(neon_log: Path, tmp_path: Path) -> None:
    log = GameLog.load(neon_log)
    frames = list(neon_frames(log, cell=8, hold_end=4, fps=10))
    assert len(frames) == len(log.moves) + 1 + 4
    assert frames[0].shape == (*frame_size(log.config, 8), 3)
    again = list(neon_frames(log, cell=8, hold_end=4, fps=10))
    assert all(np.array_equal(a, b) for a, b in zip(frames, again, strict=True))
    quiet = list(neon_frames(log, cell=8, hold_end=4, fps=10, effects=False))
    assert any(not np.array_equal(a, b) for a, b in zip(frames, quiet, strict=True))
    sparse = list(log_frames(log, cell=8, every=5, theme="neon", hold_end=0, voronoi=True))
    assert len(sparse) == len(range(0, len(log.moves) + 1, 5)) + (len(log.moves) % 5 != 0)
    classic = next(iter(log_frames(log, cell=8)))
    assert classic.shape == (8 * 8, 8 * 8, 3)  # le rendu classique reste le défaut
    from tron.viz.export import export_animation

    out = export_animation(iter(frames[:6]), tmp_path / "neon.gif", fps=10)
    assert out.exists() and out.stat().st_size > 0


def test_comparison_frame_neon(neon_log: Path) -> None:
    from tron.viz.comparison import ComparisonSession

    log = GameLog.load(neon_log)
    session = ComparisonSession(log, log)
    session.seek(4)
    img = session.frame(8, theme="neon", quality="low")
    h, w = frame_size(log.config, 8)
    assert img.shape == (h, 2 * w + 12, 3)
    frames = session.frames(8, every=7, hold_end=1, theme="neon")
    assert frames[0].shape == img.shape and session.ply == 4


def test_cli_theme_options(neon_log: Path, tmp_path: Path) -> None:
    from tron.cli.main import build_parser, main

    parser = build_parser()
    args = parser.parse_args(["replay", str(neon_log), "--no-gui"])
    assert args.theme == "neon" and args.quality == "medium"
    args = parser.parse_args(["compare", "a", "b", "--theme", "classic", "--quality", "ultra"])
    assert (args.theme, args.quality) == ("classic", "ultra")
    with pytest.raises(SystemExit):
        parser.parse_args(["play", "x.yaml", "--quality", "max"])
    gif = tmp_path / "rejeu.gif"
    assert main(["replay", str(neon_log), "--gif", str(gif), "--cell", "6", "--every", "4"]) == 0
    assert gif.exists()
    png = tmp_path / "cmp.png"
    code = main(["compare", str(neon_log), str(neon_log), "--png", str(png), "--no-gui"])
    assert code == 0 and png.exists()
