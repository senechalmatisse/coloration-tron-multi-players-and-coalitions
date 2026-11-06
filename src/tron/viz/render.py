"""Rendu d'un état en image RGB (NumPy), indépendant de toute bibliothèque graphique.

Sert à l'export GIF/MP4, aux captures d'écran et aux tests ; la fenêtre arcade
dessine la même scène avec ses propres primitives.
"""

from __future__ import annotations

import numpy as np

from tron.core.state import FREE, GameState
from tron.viz.palette import Palette, label_ink

try:  # Pillow sert seulement à inscrire les numéros des joueurs
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


def color_grid(state: GameState, palette: Palette, owners: np.ndarray | None = None) -> np.ndarray:
    """Couleur de chaque case, tableau (H, W, 3) ``uint8`` (une case = un pixel)."""
    g = state.grid2d().astype(np.int64)
    h, w = g.shape
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = palette.surface
    if owners is not None:
        free = g == FREE
        for p in range(state.n):
            img[free & (owners == p)] = palette.territory[p]
    for p in range(state.n):
        img[g == p] = palette.player[p] if state.is_alive(p) else palette.dead[p]
    for p in range(state.n):
        x, y = state.head(p)
        img[y, x] = palette.head[p] if state.is_alive(p) else palette.dead[p]
    return img


def render_frame(
    state: GameState,
    palette: Palette,
    cell: int = 16,
    owners: np.ndarray | None = None,
    grid_lines: bool = True,
    labels: bool = True,
) -> np.ndarray:
    """Image (H·cell, W·cell, 3) de l'état : murs, têtes, éliminés, territoires éventuels."""
    base = color_grid(state, palette, owners)
    img = np.repeat(np.repeat(base, cell, axis=0), cell, axis=1)
    if grid_lines and cell >= 6:
        img[::cell, :, :] = palette.grid
        img[:, ::cell, :] = palette.grid
    ring = 2 if cell >= 10 else 1
    for p in range(state.n):
        x, y = state.head(p)
        x0, y0 = x * cell, y * cell
        block = img[y0 : y0 + cell, x0 : x0 + cell]
        if state.is_alive(p):
            if cell >= 6:
                block[:ring, :] = palette.ink
                block[-ring:, :] = palette.ink
                block[:, :ring] = palette.ink
                block[:, -ring:] = palette.ink
        elif cell >= 6:
            idx = np.arange(cell)
            block[idx, idx] = palette.ink
            block[idx, cell - 1 - idx] = palette.ink
    if labels and cell >= 14 and Image is not None:
        img = _draw_labels(img, state, palette, cell)
    return img


def _draw_labels(img: np.ndarray, state: GameState, palette: Palette, cell: int) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for p in range(state.n):
        if not state.is_alive(p):
            continue
        x, y = state.head(p)
        text = str(p)
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tx = x * cell + (cell - (right - left)) / 2 - left
        ty = y * cell + (cell - (bottom - top)) / 2 - top
        draw.text((tx, ty), text, fill=label_ink(palette.head[p]))
    return np.asarray(pil)
