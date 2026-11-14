"""Rendu d'un état en image RGB (NumPy), indépendant de toute bibliothèque graphique.

Sert à l'export GIF/MP4, aux captures d'écran et aux tests ; la fenêtre arcade
dessine la même scène avec ses propres primitives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tron.core.directions import PASS
from tron.core.state import FREE, OBSTACLE, GameState
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
    img[g == OBSTACLE] = palette.muted  # obstacles dynamiques
    for p in range(state.n):
        img[g == p] = palette.player[p] if state.is_alive(p) else palette.dead[p]
    for p in range(state.n):
        x, y = state.head(p)
        img[y, x] = palette.head[p] if state.is_alive(p) else palette.dead[p]
    return img


def pv_path(state: GameState, moves: Sequence[int]) -> list[tuple[int, int, int]]:
    """Cases (joueur, x, y) parcourues par la suite de coups ``moves`` depuis ``state``.

    Sert à dessiner la variante principale d'une décision. ``state`` n'est pas modifié ;
    le parcours s'arrête à la fin de partie ou à un coup impossible.
    """
    st = state.copy(history=False)
    out: list[tuple[int, int, int]] = []
    for m in moves:
        if st.is_terminal():
            break
        p = st.current
        if not 0 <= m <= 3 or m not in st.legal_moves():
            st.apply(PASS)
            continue
        st.apply(m)
        x, y = st.head(p)
        out.append((p, x, y))
    return out


def render_frame(
    state: GameState,
    palette: Palette,
    cell: int = 16,
    owners: np.ndarray | None = None,
    grid_lines: bool = True,
    labels: bool = True,
    path: Sequence[tuple[int, int, int]] = (),
) -> np.ndarray:
    """Image (H·cell, W·cell, 3) de l'état : murs, têtes, éliminés, territoires éventuels.

    ``path`` (voir :func:`pv_path`) marque d'un carré central, à la couleur de tête
    du joueur, les cases d'une variante principale.
    """
    base = color_grid(state, palette, owners)
    img = np.repeat(np.repeat(base, cell, axis=0), cell, axis=1)
    if grid_lines and cell >= 6:
        img[::cell, :, :] = palette.grid
        img[:, ::cell, :] = palette.grid
    if cell >= 4:
        _draw_zones(img, state, palette, cell)
    if path and cell >= 4:
        lo, hi = cell // 3, cell - cell // 3
        for p, x, y in path:
            img[y * cell + lo : y * cell + hi, x * cell + lo : x * cell + hi] = palette.head[p]
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


def _draw_zones(img: np.ndarray, state: GameState, palette: Palette, cell: int) -> None:
    """Contour des zones du mode contrôle de zone, à l'encre secondaire."""
    edge = 2 if cell >= 10 else 1
    for z in state.config.rules.zones:
        x0, y0 = z.x * cell, z.y * cell
        x1, y1 = (z.x + z.width) * cell, (z.y + z.height) * cell
        img[y0 : y0 + edge, x0:x1] = palette.ink_secondary
        img[y1 - edge : y1, x0:x1] = palette.ink_secondary
        img[y0:y1, x0 : x0 + edge] = palette.ink_secondary
        img[y0:y1, x1 - edge : x1] = palette.ink_secondary


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
