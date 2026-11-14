"""Rendu néon : images partagées (tuiles de traces, motos, fond) et rendu NumPy des exports.

**Traces lumineuses.** Chaque case d'une trace est une *tuile* : le corps de la
trace (carré central et bras vers les cases voisines de la trace, rectangles
disjoints rastérisés exactement) puis sa lueur, somme de flous gaussiens. Le
flou étant linéaire, la somme des tuiles dessinées en mélange additif est
exactement la lueur de la trace entière : la fenêtre Arcade dessine ces tuiles
comme des sprites (une mise à jour par demi-coup), le rendu NumPy les cumule
dans un calque par joueur. Les deux rendus sont donc identiques, au
quantifiage 8 bits près. Le cœur blanc des traces suit le même principe, sans
flou, en mélange alpha.

**Motos.** Silhouettes vues de dessus (roues, carénage, cockpit, phare)
dessinées par champs de distance signée, anticrénelées, une image par joueur
et par orientation.

**Fond.** Dégradé radial, grille fine, lignes majeures lumineuses toutes les 5
ou 10 cases, liseré lumineux autour de l'arène.

Le rendu classique (:mod:`tron.viz.render`) reste inchangé.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from tron.agents.context import AllianceEvent
from tron.core.config import GameConfig
from tron.core.rules import OBSTACLE
from tron.core.state import GameState
from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.recorder import GameLog
from tron.viz.effects.director import EffectDirector, zone_color, zone_controllers
from tron.viz.palette import RGB, mix
from tron.viz.scene import (
    EAST_BIT,
    NORTH_BIT,
    SOUTH_BIT,
    WEST_BIT,
    Cell,
    PlyExtras,
    SceneTracker,
    alliance_pairs,
    record_extras,
    territory_edges,
)
from tron.viz.theme import (
    FLASH_ALPHA,
    LINK_FLOW_SPEED,
    SCAN_PERIOD,
    NeonTheme,
    Quality,
    TrailMetrics,
    arena_padding,
    major_step,
    make_neon_theme,
    pulse,
    resolve_quality,
    trail_metrics,
)

try:  # Pillow sert seulement à inscrire les numéros des joueurs
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

_WHITE = (255, 255, 255)
_DARK = (8, 12, 20)
_GLASS = (12, 22, 38)


# ------------------------------------------------------------ rastérisation
def _stamp_rect(canvas: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> None:
    """Ajoute à ``canvas`` la couverture exacte (filtre boîte) du rectangle donné."""
    if x1 <= x0 or y1 <= y0:
        return
    h, w = canvas.shape
    ix0, ix1 = max(0, math.floor(x0)), min(w, math.ceil(x1))
    iy0, iy1 = max(0, math.floor(y0)), min(h, math.ceil(y1))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    xs = np.arange(ix0, ix1, dtype=np.float64)
    ys = np.arange(iy0, iy1, dtype=np.float64)
    cx = np.clip(np.minimum(x1, xs + 1) - np.maximum(x0, xs), 0.0, 1.0)
    cy = np.clip(np.minimum(y1, ys + 1) - np.maximum(y0, ys), 0.0, 1.0)
    canvas[iy0:iy1, ix0:ix1] += np.outer(cy, cx)


def stroke_mask(size: int, origin: float, cell: int, mask: int, width: float) -> np.ndarray:
    """Couverture du tracé d'une case : carré central et bras vers ses voisines.

    La case occupe ``[origin, origin + cell)`` sur les deux axes d'un canevas de
    ``size`` pixels. Les rectangles sont disjoints : la somme des tuiles de
    cases voisines reconstitue exactement le tracé continu.
    """
    canvas = np.zeros((size, size), dtype=np.float64)
    c0 = float(origin)
    mid = c0 + cell / 2
    hw = width / 2
    _stamp_rect(canvas, mid - hw, mid + hw, mid - hw, mid + hw)
    if mask & NORTH_BIT:
        _stamp_rect(canvas, mid - hw, mid + hw, c0, mid - hw)
    if mask & SOUTH_BIT:
        _stamp_rect(canvas, mid - hw, mid + hw, mid + hw, c0 + cell)
    if mask & EAST_BIT:
        _stamp_rect(canvas, mid + hw, c0 + cell, mid - hw, mid + hw)
    if mask & WEST_BIT:
        _stamp_rect(canvas, c0, mid - hw, mid - hw, mid + hw)
    return canvas


def line_peak(metrics: TrailMetrics) -> float:
    """Intensité au centre d'une trace droite infinie (avant normalisation)."""
    w = metrics.body

    def erf_term(sigma: float) -> float:
        return math.erf(w / (2 * math.sqrt(2) * sigma)) if sigma > 0 else 0.0

    return (
        metrics.gain_body
        + metrics.gain_tight * erf_term(metrics.sigma_tight)
        + metrics.gain_wide * erf_term(metrics.sigma_wide)
    )


@dataclass(frozen=True, slots=True)
class TrailTiles:
    """Tuiles des 16 configurations de connexion (bits N, E, S, O) d'une case de trace."""

    metrics: TrailMetrics
    glow: tuple[np.ndarray, ...]
    """Intensité de lueur (corps compris), float32 dans [0, 1], mélange additif."""
    core: tuple[np.ndarray, ...]
    """Couverture du cœur blanc, float32 dans [0, 1], mélange alpha."""


@lru_cache(maxsize=16)
def trail_tiles(metrics: TrailMetrics) -> TrailTiles:
    """Tuiles de trace pour une géométrie donnée (mises en cache)."""
    size, m, c = metrics.tile, metrics.margin, metrics.cell
    scale = 1.0 / line_peak(metrics)
    glow = []
    core = []
    for mask in range(16):
        body = stroke_mask(size, m, c, mask, metrics.body)
        tile = metrics.gain_body * body
        tile = tile + metrics.gain_tight * gaussian_filter(
            body, metrics.sigma_tight, mode="constant", truncate=3.0
        )
        if metrics.sigma_wide > 0:
            tile = tile + metrics.gain_wide * gaussian_filter(
                body, metrics.sigma_wide, mode="constant", truncate=3.0
            )
        glow.append(np.clip(tile * scale, 0.0, 1.0).astype(np.float32))
        core.append(stroke_mask(size, m, c, mask, metrics.core).astype(np.float32))
    return TrailTiles(metrics, tuple(glow), tuple(core))


# ------------------------------------------------------------ formes (SDF)
def _sd_round_box(
    u: np.ndarray, v: np.ndarray, cu: float, cv: float, hu: float, hv: float, r: float
) -> np.ndarray:
    r = min(r, hu, hv)
    qu = np.abs(u - cu) - hu + r
    qv = np.abs(v - cv) - hv + r
    outside = np.hypot(np.maximum(qu, 0.0), np.maximum(qv, 0.0))
    inside = np.minimum(np.maximum(qu, qv), 0.0)
    return np.asarray(outside + inside - r)


def _sd_ellipse(
    u: np.ndarray, v: np.ndarray, cu: float, cv: float, ru: float, rv: float
) -> np.ndarray:
    k = np.hypot((u - cu) / ru, (v - cv) / rv)
    return np.asarray((k - 1.0) * min(ru, rv))


def _fill(d: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(0.5 - d, 0.0, 1.0))


def _stroke(d: np.ndarray, width: float) -> np.ndarray:
    return np.asarray(np.clip(width / 2 + 0.5 - np.abs(d), 0.0, 1.0))


def _over(canvas: np.ndarray, cover: np.ndarray, rgb: Sequence[int], alpha: float = 1.0) -> None:
    """Compose (prémultiplié) une couche unie de couverture ``cover`` sur ``canvas``."""
    a = cover * alpha
    canvas[..., :3] = np.asarray(rgb, dtype=np.float32) * a[..., None] + canvas[..., :3] * (
        1.0 - a[..., None]
    )
    canvas[..., 3] = a + canvas[..., 3] * (1.0 - a)


def _rotate(img: np.ndarray, heading: int) -> np.ndarray:
    """Oriente une image dessinée vers l'est selon ``heading`` (0 N, 1 E, 2 S, 3 O)."""
    k = {0: 1, 1: 0, 2: -1, 3: 2}.get(heading, 0)
    return np.ascontiguousarray(np.rot90(img, k=k))


def _to_rgba8(premul: np.ndarray) -> np.ndarray:
    a = premul[..., 3:4]
    rgb = np.where(a > 1e-6, premul[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([rgb, a * 255.0], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@lru_cache(maxsize=256)
def bike_image(color: RGB, highlight: RGB, cell: int, heading: int, lit: bool = True) -> np.ndarray:
    """Moto lumineuse vue de dessus, RGBA ``uint8`` de côté ``2 · cell`` centrée sur la case.

    Grandes cases : deux roues cerclées de lumière, carénage, cockpit, liseré
    clair, phare et feu arrière. Petites cases : carénage et phare seulement.
    ``lit=False`` : moto éteinte (joueur éliminé), sans phare.
    """
    c = max(4, int(cell))
    size = max(8, 2 * c)
    px = np.arange(size, dtype=np.float32) + 0.5 - size / 2
    u, v = np.meshgrid(px, px)
    nose = 0.66 * c
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    detailed = c >= 12
    body_half = 0.21 * c if detailed else 0.30 * c
    if detailed:
        rim = max(1.0, 0.075 * c)
        for cu in (-0.40 * c, 0.40 * c):
            d = _sd_round_box(u, v, cu, 0.0, 0.23 * c, 0.26 * c, 0.20 * c)
            _over(canvas, _fill(d), _DARK)
            _over(canvas, _stroke(d, rim), mix(color, _WHITE, 0.25), 0.95)
    d_body = _sd_round_box(u, v, 0.0, 0.0, nose, body_half, 0.85 * body_half)
    _over(canvas, _fill(d_body), color)
    if detailed:
        _over(canvas, _stroke(d_body + 0.5, max(1.0, 0.05 * c)), highlight, 0.7)
        d_can = _sd_ellipse(u, v, 0.02 * c, 0.0, 0.20 * c, 0.55 * body_half)
        _over(canvas, _fill(d_can), _GLASS)
        _over(canvas, _stroke(d_can, max(1.0, 0.04 * c)), highlight, 0.85)
        d_line = _sd_round_box(u, v, -0.30 * c, 0.0, 0.20 * c, 0.45, 0.45)
        _over(canvas, _fill(d_line), highlight, 0.9)
    if lit:
        r_head = max(0.9, 0.085 * c)
        d_head = np.hypot(u - (nose - 1.2 * r_head), v) - r_head
        _over(canvas, _fill(np.asarray(d_head)), _WHITE)
        d_tail = _sd_round_box(
            u, v, -nose + max(0.6, 0.03 * c), 0.0, max(0.6, 0.03 * c), body_half * 0.75, 0.3
        )
        _over(canvas, _fill(d_tail), mix(color, _WHITE, 0.5))
    return _rotate(_to_rgba8(canvas), heading)


@lru_cache(maxsize=64)
def soft_dot(radius: int) -> np.ndarray:
    """Point lumineux gaussien (float32) de rayon visible ``radius`` pixels, pic 1."""
    r = max(1, int(radius))
    size = 2 * r + 1
    px = np.arange(size, dtype=np.float32) - r
    xx, yy = np.meshgrid(px, px)
    sigma = max(0.6, r / 2.6)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)


@lru_cache(maxsize=16)
def obstacle_tile(cell: int, fill: RGB, edge: RGB) -> np.ndarray:
    """Bloc d'obstacle (RGBA ``uint8``) : métal sombre, liseré orangé, hachure."""
    c = max(2, int(cell))
    px = np.arange(c, dtype=np.float32) + 0.5 - c / 2
    u, v = np.meshgrid(px, px)
    canvas = np.zeros((c, c, 4), dtype=np.float32)
    d = _sd_round_box(u, v, 0.0, 0.0, c / 2 - 0.5, c / 2 - 0.5, 0.12 * c)
    _over(canvas, _fill(d), fill)
    if c >= 8:
        hatch = (np.mod(u + v, max(3.0, c / 3)) < 1.0).astype(np.float32) * _fill(d + 0.18 * c)
        _over(canvas, hatch, edge, 0.25)
    _over(canvas, _stroke(d + 0.5, max(1.0, 0.08 * c)), edge, 0.85)
    return _to_rgba8(canvas)


# ------------------------------------------------------------------- fond
@lru_cache(maxsize=8)
def arena_background(
    width: int, height: int, cell: int, theme: NeonTheme, quality: Quality
) -> np.ndarray:
    """Fond de l'arène (RGB ``uint8``) : dégradé, grille, lignes majeures, liseré lumineux."""
    pad = arena_padding(cell)
    w_px, h_px = width * cell + 2 * pad, height * cell + 2 * pad
    ys = (np.arange(h_px, dtype=np.float32) + 0.5 - h_px / 2) / (h_px / 2)
    xs = (np.arange(w_px, dtype=np.float32) + 0.5 - w_px / 2) / (w_px / 2)
    xx, yy = np.meshgrid(xs, ys)
    k = np.clip(np.hypot(xx, yy) / 1.25, 0.0, 1.0)[..., None]
    bg = np.asarray(theme.background, dtype=np.float32)
    alt = np.asarray(theme.background_alt, dtype=np.float32)
    img = alt + (bg - alt) * k**1.4
    inside = np.zeros((h_px, w_px), dtype=bool)
    inside[pad : pad + height * cell, pad : pad + width * cell] = True
    img[~inside] *= 0.72
    if cell >= 6:  # grille fine, à la limite gauche et haute de chaque case
        minor = np.zeros((h_px, w_px), dtype=np.float32)
        for k_x in range(1, width):
            minor[pad : pad + height * cell, pad + k_x * cell] = 1.0
        for k_y in range(1, height):
            minor[pad + k_y * cell, pad : pad + width * cell] = 1.0
        a = 0.55 * minor[..., None]
        img = img * (1 - a) + np.asarray(theme.grid, dtype=np.float32) * a
    if quality.grid_major:
        step = major_step(width, height)
        major = np.zeros((h_px, w_px), dtype=np.float32)
        thick = 2 if cell >= 24 else 1
        for k_x in range(step, width, step):
            x0 = pad + k_x * cell
            major[pad : pad + height * cell, x0 : x0 + thick] = 1.0
        for k_y in range(step, height, step):
            y0 = pad + k_y * cell
            major[y0 : y0 + thick, pad : pad + width * cell] = 1.0
        color = np.asarray(theme.grid_major, dtype=np.float32)
        a = 0.70 * major[..., None]
        img = img * (1 - a) + color * a
        img += 0.30 * gaussian_filter(major, 1.6, mode="constant")[..., None] * color
    border = np.zeros((h_px, w_px), dtype=np.float32)
    t = 2 if cell >= 10 else 1
    x0, x1 = pad - t, pad + width * cell + t
    y0, y1 = pad - t, pad + height * cell + t
    border[y0:y1, x0 : x0 + t] = border[y0:y1, x1 - t : x1] = 1.0
    border[y0 : y0 + t, x0:x1] = border[y1 - t : y1, x0:x1] = 1.0
    edge = np.asarray(theme.border, dtype=np.float32)
    img = img * (1 - 0.85 * border[..., None]) + edge * 0.85 * border[..., None]
    sigma = max(1.5, 0.35 * cell) if quality.halo else max(1.0, 0.15 * cell)
    img += 0.55 * gaussian_filter(border, sigma, mode="constant")[..., None] * edge
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ----------------------------------------------------------- composition
def _add_patch(
    img: np.ndarray, patch: np.ndarray, color: Sequence[float], x0: int, y0: int, gain: float = 1.0
) -> None:
    """``img[y0:, x0:] += gain · patch · color`` (patch 2D), avec découpe aux bords."""
    h, w = patch.shape[:2]
    H, W = img.shape[:2]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    sub = patch[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
    img[ay0:ay1, ax0:ax1] += (gain * sub)[..., None] * np.asarray(color, dtype=np.float32)


def _add_scalar(layer: np.ndarray, patch: np.ndarray, x0: int, y0: int, sign: float) -> None:
    h, w = patch.shape
    H, W = layer.shape
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    layer[ay0:ay1, ax0:ax1] += sign * patch[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]


def _over_rgba(img: np.ndarray, rgba: np.ndarray, x0: int, y0: int, alpha: float = 1.0) -> None:
    """Compose une image RGBA ``uint8`` (alpha non prémultiplié) sur ``img`` (float)."""
    h, w = rgba.shape[:2]
    H, W = img.shape[:2]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    sub = rgba[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0].astype(np.float32)
    a = sub[..., 3:4] / 255.0 * alpha
    img[ay0:ay1, ax0:ax1] = sub[..., :3] * a + img[ay0:ay1, ax0:ax1] * (1 - a)


def _segment_cover(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Couverture d'un segment épais sur sa boîte englobante et abscisse curviligne."""
    pad = width + 2
    x0 = max(0, math.floor(min(p0[0], p1[0]) - pad))
    y0 = max(0, math.floor(min(p0[1], p1[1]) - pad))
    x1 = min(shape[1], math.ceil(max(p0[0], p1[0]) + pad))
    y1 = min(shape[0], math.ceil(max(p0[1], p1[1]) + pad))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32), x0, y0
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length2 = dx * dx + dy * dy
    if length2 <= 1e-9:
        t = np.zeros_like(xx)
    else:
        t = np.clip(((xx - p0[0]) * dx + (yy - p0[1]) * dy) / length2, 0.0, 1.0)
    d = np.hypot(xx - (p0[0] + t * dx), yy - (p0[1] + t * dy))
    cover = np.clip(width / 2 + 0.5 - d, 0.0, 1.0)
    return cover.astype(np.float32), (t * math.sqrt(length2)).astype(np.float32), x0, y0


def dash_pattern(s: np.ndarray, period: float, phase: float) -> np.ndarray:
    """Pointillés lumineux défilants : 1 sur la moitié de chaque période, adoucis."""
    k = np.mod(s / period - phase, 1.0)
    return np.asarray(np.clip(1.6 - 3.2 * np.abs(k - 0.5), 0.25, 1.0))


def default_heading(state: GameState, player: int) -> int:
    """Orientation d'une moto qui n'a pas encore bougé : vers le centre de l'arène."""
    x, y = state.head(player)
    dx, dy = state.width / 2 - (x + 0.5), state.height / 2 - (y + 0.5)
    if abs(dx) >= abs(dy):
        return 1 if dx >= 0 else 3
    return 2 if dy >= 0 else 0


# ------------------------------------------------------------- rendu NumPy
class NeonRenderer:
    """Rendu néon d'une partie en images RGB ``uint8`` (exports GIF/MP4, captures).

    Les traces sont cumulées tuile par tuile dans un calque par joueur, mis à
    jour de façon incrémentale d'une image à la suivante.
    """

    def __init__(
        self,
        config: GameConfig,
        cell: int = 16,
        theme: NeonTheme | None = None,
        quality: Quality | str = "medium",
        labels: bool = True,
    ) -> None:
        self.config = config
        self.cell = max(4, int(cell))
        self.theme = theme or make_neon_theme(config.team_index, config.team_names)
        self.quality = resolve_quality(quality)
        self.labels = labels
        self.pad = arena_padding(self.cell)
        self.metrics = trail_metrics(self.cell, self.quality)
        self.tiles = trail_tiles(self.metrics)
        bg = arena_background(config.width, config.height, self.cell, self.theme, self.quality)
        self.background = bg.astype(np.float32)
        self.shape = bg.shape[:2]
        n = config.n_players
        self._glow = [np.zeros(self.shape, dtype=np.float32) for _ in range(n)]
        self._core = [np.zeros(self.shape, dtype=np.float32) for _ in range(n)]
        self._stamped: list[list[tuple[Cell, int]]] = [[] for _ in range(n)]
        self._tracker: SceneTracker | None = None
        self._own: SceneTracker | None = None
        self._version = -1

    # ------------------------------------------------------------ repères
    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Position en pixels d'un point exprimé en cases."""
        return self.pad + x * self.cell, self.pad + y * self.cell

    # ------------------------------------------------------------- traces
    def _stamp(self, player: int, cell: Cell, mask: int, sign: float) -> None:
        m = self.metrics.margin
        x0 = self.pad + cell[0] * self.cell - m
        y0 = self.pad + cell[1] * self.cell - m
        _add_scalar(self._glow[player], self.tiles.glow[mask], x0, y0, sign)
        _add_scalar(self._core[player], self.tiles.core[mask], x0, y0, sign)

    def sync(self, tracker: SceneTracker) -> None:
        """Met les calques de traces à jour (seule la fin des traces change d'ordinaire)."""
        if tracker is self._tracker and tracker.version == self._version:
            return
        if tracker is not self._tracker:
            for layer in (*self._glow, *self._core):
                layer.fill(0.0)
            self._stamped = [[] for _ in self._stamped]
        for p in range(min(tracker.n, len(self._stamped))):
            path = tracker.paths[p]
            desired = [(cell, tracker.mask(p, i)) for i, cell in enumerate(path)]
            stamped = self._stamped[p]
            k, lim = 0, min(len(stamped), len(desired))
            while k < lim and stamped[k] == desired[k]:
                k += 1
            for cell, mask in stamped[k:]:
                self._stamp(p, cell, mask, -1.0)
            for cell, mask in desired[k:]:
                self._stamp(p, cell, mask, 1.0)
            self._stamped[p] = desired
        self._tracker = tracker
        self._version = tracker.version

    # -------------------------------------------------------------- image
    def render(
        self,
        state: GameState,
        tracker: SceneTracker | None = None,
        owners: np.ndarray | None = None,
        path: Sequence[tuple[int, int, int]] = (),
        effects: EffectDirector | None = None,
        t: float = 0.0,
        links: Sequence[tuple[int, int]] = (),
    ) -> np.ndarray:
        """Image de l'état (RGB ``uint8``, marge comprise pour le liseré de l'arène).

        Sans ``tracker``, le rendu suit lui-même l'état (de façon incrémentale tant
        que c'est le même objet, comme lors d'un rejeu).
        """
        if tracker is None:
            if self._own is None:
                self._own = SceneTracker()
            tracker = self._own
            tracker.sync(state, events=False)
        self.sync(tracker)
        theme, c = self.theme, self.cell
        img = self.background.copy()
        if self.quality.animated:
            self._scan(img, t)
        if owners is not None:
            self._territories(img, state, owners)
        self._obstacles(img, state)
        self._zones(img, state)
        fades = [effects.fade(p) if effects is not None else 1.0 for p in range(state.n)]
        for p in range(state.n):
            alive = state.is_alive(p)
            fade = 0.0 if alive else fades[p]
            color = theme.trail_color(p, alive, fade)
            gain = 1.0 - 0.45 * fade
            img += self._glow[p][..., None] * (np.asarray(color, dtype=np.float32) * gain)
        for p in range(state.n):
            alive = state.is_alive(p)
            strength = 1.0 if alive else 1.0 - fades[p]
            if strength <= 0.0:
                continue
            a = (self._core[p] * strength)[..., None]
            img[:] = img * (1 - a) + np.asarray(theme.core[p], dtype=np.float32) * a
        self._links(img, state, links, effects, t)
        if path:
            self._path(img, path)
        self._bikes(img, state, tracker, t)
        if effects is not None:
            self._effects(img, effects)
        out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if self.labels and c >= 14 and Image is not None:
            out = self._labels(out, state)
        return out

    def _scan(self, img: np.ndarray, t: float) -> None:
        h = self.config.height * self.cell
        y = self.pad + (t / SCAN_PERIOD % 1.0) * h
        rows = np.arange(img.shape[0], dtype=np.float32)
        band = np.exp(-(((rows - y) / max(4.0, 0.06 * h)) ** 2))
        x0, x1 = self.pad, self.pad + self.config.width * self.cell
        y0, y1 = self.pad, self.pad + h
        color = np.asarray(self.theme.accent, dtype=np.float32) * 0.035
        img[y0:y1, x0:x1] += band[y0:y1, None, None] * color

    def _cells_slice(self, x: int, y: int) -> tuple[slice, slice]:
        c = self.cell
        return (
            slice(self.pad + y * c, self.pad + (y + 1) * c),
            slice(self.pad + x * c, self.pad + (x + 1) * c),
        )

    def _territories(self, img: np.ndarray, state: GameState, owners: np.ndarray) -> None:
        theme, c = self.theme, self.cell
        g = state.grid2d()
        h, w = owners.shape
        tint = np.zeros((h, w, 3), dtype=np.float32)
        alpha = np.zeros((h, w), dtype=np.float32)
        free = g == -1
        for p in range(state.n):
            sel = free & (owners == p)
            tint[sel] = theme.player[p]
            alpha[sel] = 0.16
        big_t = np.repeat(np.repeat(tint, c, axis=0), c, axis=1)
        big_a = np.repeat(np.repeat(alpha, c, axis=0), c, axis=1)[..., None]
        region = img[self.pad : self.pad + h * c, self.pad : self.pad + w * c]
        region[:] = region * (1 - big_a) + big_t * big_a
        edge = np.asarray(mix(theme.text_dim, theme.accent, 0.3), dtype=np.float32)
        for x0, y0, x1, y1 in territory_edges(owners):
            px0, py0 = self.to_px(x0, y0)
            px1, py1 = self.to_px(x1, y1)
            ix0, iy0 = round(px0), round(py0)
            if x0 == x1:  # frontière verticale
                img[iy0 : round(py1), max(0, ix0 - 1) : ix0 + 1] = (
                    img[iy0 : round(py1), max(0, ix0 - 1) : ix0 + 1] * 0.55 + edge * 0.45
                )
            else:
                img[max(0, iy0 - 1) : iy0 + 1, ix0 : round(px1)] = (
                    img[max(0, iy0 - 1) : iy0 + 1, ix0 : round(px1)] * 0.55 + edge * 0.45
                )

    def _obstacles(self, img: np.ndarray, state: GameState) -> None:
        g = state.grid2d()
        cells = np.argwhere(g == OBSTACLE)
        if cells.size == 0:
            return
        tile = obstacle_tile(self.cell, self.theme.obstacle, self.theme.obstacle_edge)
        for y, x in cells:
            px, py = self.to_px(int(x), int(y))
            _over_rgba(img, tile, round(px), round(py))

    def _zones(self, img: np.ndarray, state: GameState) -> None:
        zones = state.config.rules.zones
        if not zones:
            return
        controllers = zone_controllers(state)
        edge = 2 if self.cell >= 10 else 1
        for zone, team in zip(zones, controllers, strict=True):
            color = np.asarray(zone_color(state, self.theme, team), dtype=np.float32)
            x0, y0 = (round(v) for v in self.to_px(zone.x, zone.y))
            x1, y1 = (round(v) for v in self.to_px(zone.x + zone.width, zone.y + zone.height))
            fill = 0.10 if team >= 0 else 0.04
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - fill) + color * fill
            for sl in (
                (slice(y0, y0 + edge), slice(x0, x1)),
                (slice(y1 - edge, y1), slice(x0, x1)),
                (slice(y0, y1), slice(x0, x0 + edge)),
                (slice(y0, y1), slice(x1 - edge, x1)),
            ):
                img[sl] = img[sl] * 0.2 + color * 0.8

    def _links(
        self,
        img: np.ndarray,
        state: GameState,
        links: Sequence[tuple[int, int]],
        effects: EffectDirector | None,
        t: float,
    ) -> None:
        drawn: list[tuple[int, int, float, float]] = [(a, b, 1.0, 1.0) for a, b in links]
        if effects is not None:
            for lk in effects.active_links():
                drawn.append((lk.a, lk.b, lk.progress(effects.time), lk.alpha(effects.time)))
        c = self.cell
        phase = t * LINK_FLOW_SPEED if self.quality.animated else 0.0
        for a, b, progress, alpha in drawn:
            if not (state.is_alive(a) and state.is_alive(b)) or alpha <= 0:
                continue
            ax, ay = state.head(a)
            bx, by = state.head(b)
            p0 = self.to_px(ax + 0.5, ay + 0.5)
            end = self.to_px(bx + 0.5, by + 0.5)
            p1 = (p0[0] + (end[0] - p0[0]) * progress, p0[1] + (end[1] - p0[1]) * progress)
            color = mix(self.theme.player[a], self.theme.player[b], 0.5)
            cover, s, x0, y0 = _segment_cover(img.shape[:2], p0, p1, max(1.5, 0.12 * c))
            if cover.size == 0:
                continue
            dashes = dash_pattern(s, max(4.0, 0.9 * c), phase)
            _add_patch(img, cover * dashes, color, x0, y0, 0.85 * alpha)
            glow, _, gx0, gy0 = _segment_cover(img.shape[:2], p0, p1, max(3.0, 0.45 * c))
            _add_patch(img, glow, color, gx0, gy0, 0.12 * alpha)

    def _path(self, img: np.ndarray, path: Sequence[tuple[int, int, int]]) -> None:
        c = self.cell
        r = max(1.5, 0.22 * c)
        size = math.ceil(2 * r) + 2
        px = np.arange(size, dtype=np.float32) + 0.5 - size / 2
        xx, yy = np.meshgrid(px, px)
        diamond = np.clip(r + 0.5 - (np.abs(xx) + np.abs(yy)), 0.0, 1.0)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        for p, x, y in path:
            rgba[..., :3] = self.theme.head[p]
            rgba[..., 3] = np.rint(diamond * 220).astype(np.uint8)
            cx, cy = self.to_px(x + 0.5, y + 0.5)
            _over_rgba(img, rgba, round(cx - size / 2), round(cy - size / 2))
            dot = soft_dot(max(2, round(0.45 * c)))
            _add_patch(
                img, dot, self.theme.player[p], round(cx) - dot.shape[1] // 2,
                round(cy) - dot.shape[0] // 2, 0.25,
            )  # fmt: skip

    def _bikes(self, img: np.ndarray, state: GameState, tracker: SceneTracker, t: float) -> None:
        theme, c = self.theme, self.cell
        glow = pulse(t, self.quality.animated)
        for p in range(state.n):
            x, y = state.head(p)
            cx, cy = self.to_px(x + 0.5, y + 0.5)
            alive = state.is_alive(p)
            heading = tracker.headings[p] if p < tracker.n else -1
            if heading < 0:
                heading = default_heading(state, p)
            if alive:
                dot = soft_dot(max(2, round(1.1 * c * glow)))
                _add_patch(
                    img,
                    dot,
                    theme.player[p],
                    round(cx) - dot.shape[1] // 2,
                    round(cy) - dot.shape[0] // 2,
                    0.30 + 0.25 * glow,
                )
                bike = bike_image(theme.player[p], theme.head[p], c, heading, True)
            else:
                bike = bike_image(theme.dead[p], mix(theme.dead[p], _WHITE, 0.2), c, heading, False)
            size = bike.shape[0]
            _over_rgba(img, bike, round(cx - size / 2), round(cy - size / 2))
            if not alive and c >= 6:
                half = 0.32 * c
                for sx in (-1, 1):
                    cover, _, x0, y0 = _segment_cover(
                        img.shape[:2],
                        (cx - half, cy - half * sx),
                        (cx + half, cy + half * sx),
                        max(1.0, 0.09 * c),
                    )
                    if cover.size:
                        a = (cover * 0.85)[..., None]
                        sub = img[y0 : y0 + cover.shape[0], x0 : x0 + cover.shape[1]]
                        sub[:] = sub * (1 - a) + np.asarray(theme.danger, dtype=np.float32) * a

    def _effects(self, img: np.ndarray, effects: EffectDirector) -> None:
        c = self.cell
        view = effects.particles.view()
        for i in range(len(view)):
            radius = max(1, round(float(view.size[i]) * c * 1.6))
            dot = soft_dot(radius)
            px, py = self.to_px(float(view.x[i]), float(view.y[i]))
            _add_patch(
                img,
                dot,
                view.color[i],
                round(px) - radius,
                round(py) - radius,
                float(view.alpha[i]) * 0.9,
            )
        now = effects.time
        for ring in effects.active_rings():
            r = ring.radius(now) * c
            alpha = ring.alpha(now)
            cx, cy = self.to_px(ring.x, ring.y)
            width = max(1.0, 0.12 * c)
            x0, y0 = math.floor(cx - r - width - 2), math.floor(cy - r - width - 2)
            size = math.ceil(2 * (r + width + 2))
            xs = np.arange(size, dtype=np.float32) + x0 + 0.5
            xx, yy = np.meshgrid(xs, np.arange(size, dtype=np.float32) + y0 + 0.5)
            d = np.abs(np.hypot(xx - cx, yy - cy) - r)
            cover = np.clip(width / 2 + 0.5 - d, 0.0, 1.0)
            _add_patch(img, cover, ring.color, x0, y0, 0.9 * alpha)
        flash = effects.flash_alpha()
        if flash > 0:
            img += flash * FLASH_ALPHA * 0.6

    def _labels(self, img: np.ndarray, state: GameState) -> np.ndarray:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        font = _font(max(9, int(self.cell * 0.5)))
        for p in range(state.n):
            x, y = state.head(p)
            px, py = self.to_px(x + 1.0, y)
            color = self.theme.text if state.is_alive(p) else self.theme.text_muted
            draw.text(
                (px - 2, py - self.cell * 0.55),
                str(p),
                fill=color,
                font=font,
                stroke_width=2,
                stroke_fill=self.theme.background,
            )
        return np.asarray(pil)


@lru_cache(maxsize=8)
def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow ancien
        return ImageFont.load_default()


def render_neon(
    state: GameState,
    cell: int = 16,
    quality: Quality | str = "medium",
    owners: np.ndarray | None = None,
    path: Sequence[tuple[int, int, int]] = (),
    labels: bool = True,
) -> np.ndarray:
    """Image néon d'un état isolé (sans effets), RGB ``uint8``."""
    renderer = NeonRenderer(state.config, cell, quality=quality, labels=labels)
    return renderer.render(state, owners=owners, path=path)


def neon_frames(
    log: GameLog,
    cell: int = 12,
    every: int = 1,
    voronoi: bool = False,
    hold_end: int = 15,
    quality: Quality | str = "medium",
    fps: float = 20.0,
    effects: bool = True,
) -> Iterator[np.ndarray]:
    """Images néon successives d'une partie journalisée, effets animés compris.

    Le temps des effets avance de ``1 / fps`` par image ; les dernières images
    (``hold_end``) laissent les effets de fin de partie se dérouler.
    """
    cfg = log.config
    renderer = NeonRenderer(cfg, cell, quality=quality)
    director = EffectDirector(renderer.theme, renderer.quality, seed=cfg.seed) if effects else None
    tracker = SceneTracker()
    heur = VoronoiHeuristic(cfg.width, cfg.height, cfg.n_players)
    directions = log.directions
    total = len(log.moves)

    def extras(ply: int) -> PlyExtras:
        if 0 <= ply < total:
            return record_extras(log.moves[ply], directions[ply])
        return PlyExtras()

    diplomacy: list[AllianceEvent] = []
    dt = 1.0 / max(1e-6, fps)
    speed = fps * max(1, every)
    t = 0.0
    frame: np.ndarray | None = None
    last: GameState | None = None
    for i, st in enumerate(log.states()):
        result = tracker.sync(st, extras)
        if i > 0:
            diplomacy.extend(extras(i - 1).diplomacy)
        if director is not None:
            director.handle(result, tracker, st, speed=speed)
        if i % every and i != total:
            continue
        if director is not None:
            director.update(dt)
        owners = heur.owner_map(st) if voronoi and not st.is_terminal() else None
        frame = renderer.render(
            st, tracker, owners=owners, effects=director, t=t, links=alliance_pairs(diplomacy)
        )
        last = st
        t += dt
        yield frame
    for _ in range(hold_end if frame is not None else 0):
        if director is not None and last is not None and director.busy:
            director.update(dt)
            frame = renderer.render(
                last, tracker, effects=director, t=t, links=alliance_pairs(diplomacy)
            )
        t += dt
        yield frame  # type: ignore[misc]


def frame_size(config: GameConfig, cell: int) -> tuple[int, int]:
    """Taille (hauteur, largeur) en pixels d'une image néon."""
    pad = arena_padding(cell)
    return config.height * cell + 2 * pad, config.width * cell + 2 * pad


ExtrasFn = Callable[[int], PlyExtras]

__all__ = [
    "NeonRenderer",
    "TrailTiles",
    "arena_background",
    "bike_image",
    "dash_pattern",
    "default_heading",
    "frame_size",
    "line_peak",
    "neon_frames",
    "obstacle_tile",
    "render_neon",
    "soft_dot",
    "stroke_mask",
    "trail_tiles",
]
