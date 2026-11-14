"""Rendu hors écran : ``render_frame`` sans fenêtre ni Arcade.

Ces tests valident l'image NumPy (dimensions, type, couleurs des murs, des têtes,
des joueurs éliminés et des territoires). Ils ne valident pas la fenêtre Arcade
elle-même : voir la procédure manuelle de ``docs/validation-gui.md``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tron.core import GameConfig, GameState
from tron.eval import VoronoiHeuristic
from tron.viz import make_palette, render_frame
from tron.viz.render import color_grid

ROOT = Path(__file__).resolve().parents[1]


def _state() -> GameState:
    cfg = GameConfig.from_dict(
        {
            "grid": {"width": 9, "height": 6},
            "placement": {"positions": [[1, 1], [7, 4], [4, 2]]},
            "players": [
                {"name": "a", "team": "A"},
                {"name": "b", "team": "B"},
                {"name": "c", "team": "A"},
            ],
        }
    )
    st = GameState(cfg)
    st.apply(1)  # a -> (2, 1)
    st.apply(3)  # b -> (6, 4)
    st.apply(2)  # c -> (4, 3)
    return st


def _cell_center(img: np.ndarray, x: int, y: int, cell: int) -> tuple[int, ...]:
    return tuple(int(v) for v in img[y * cell + cell // 2, x * cell + cell // 2])


def test_dimensions_dtype_and_determinism() -> None:
    st = _state()
    pal = make_palette(st.config.team_index)
    for cell in (1, 3, 6, 10, 16, 24):
        img = render_frame(st, pal, cell=cell)
        assert img.shape == (6 * cell, 9 * cell, 3) and img.dtype == np.uint8
        assert np.array_equal(img, render_frame(st, pal, cell=cell))  # déterministe
    grid = color_grid(st, pal)
    assert grid.shape == (6, 9, 3) and grid.dtype == np.uint8


def test_wall_head_dead_and_territory_colours() -> None:
    st = _state()
    pal = make_palette(st.config.team_index)
    cell = 12
    img = render_frame(st, pal, cell=cell, labels=False)
    for p, wall in ((0, (1, 1)), (1, (7, 4)), (2, (4, 2))):
        assert _cell_center(img, *wall, cell) == pal.player[p]  # mur (case de départ)
        hx, hy = st.head(p)
        assert _cell_center(img, hx, hy, cell) == pal.head[p]  # intérieur de la tête
        assert tuple(img[hy * cell, hx * cell + cell // 2]) == pal.ink  # anneau
    assert _cell_center(img, 0, 5, cell) == pal.surface
    st.apply(-1)  # a est éliminé (passe)
    dead = render_frame(st, pal, cell=cell, labels=False)
    assert _cell_center(dead, 1, 1, cell) == pal.dead[0]
    hx, hy = st.head(0)
    assert tuple(dead[hy * cell, hx * cell]) == pal.ink  # croix de la tête éliminée
    owners = VoronoiHeuristic(9, 6, 3).owner_map(st)
    over = render_frame(st, pal, cell=cell, owners=owners, labels=False)
    free = [(x, y) for y in range(6) for x in range(9) if st.grid2d()[y, x] == -1]
    for x, y in free[:10]:
        owner = int(owners[y, x])
        expected = pal.territory[owner] if owner >= 0 else pal.surface
        assert _cell_center(over, x, y, cell) == expected


def test_labels_and_rendering_never_import_arcade() -> None:
    code = (
        "import sys; from tron.core import GameConfig, GameState;"
        "from tron.viz import make_palette, render_frame;"
        "cfg = GameConfig.from_dict({'grid': 12, 'players': [{'name': f'J{i}', 'team': 'AB'[i % 2]}"
        " for i in range(8)]});"
        "st = GameState(cfg); img = render_frame(st, make_palette(cfg.team_index), cell=20);"
        "print(img.shape, 'arcade' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
    )
    assert out.stdout.strip() == "(240, 240, 3) False"
