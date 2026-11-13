"""Fenêtre Arcade de comparaison de deux rejeux (P17).

Deux plateaux côte à côte, synchronisés par :class:`ComparisonSession`, une ligne
de temps d'événements sous chaque plateau et un panneau de comparaison. Touches :
``Espace`` lecture / pause, ``→`` / ``←`` demi-coup suivant / précédent,
``↑`` / ``↓`` vitesse, ``Début`` / ``Fin``, ``V`` territoires de Voronoi,
``Échap`` / ``Q`` quitter. Toute la logique (curseur, données, avertissement
d'appariement) est dans :mod:`tron.viz.comparison`, testée sans fenêtre.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tron.viz.arcade_app import FONT, MARGIN, _require_arcade, auto_cell
from tron.viz.comparison import ComparisonSession
from tron.viz.controller import SPEEDS
from tron.viz.render import color_grid

try:
    import arcade
except ImportError:  # pragma: no cover - dépend de l'installation
    arcade = None

TIMELINE_HEIGHT = 46
PANEL_HEIGHT = 250
_HELP = "Espace lecture · → ← pas · ↑ ↓ vitesse · Début Fin · V Voronoi · Échap"


class _Board:
    """Grille de cases colorées d'un plateau (une ``SpriteList``)."""

    def __init__(self, arc: Any, left: float, bottom: float, width: int, height: int, cell: int):
        self.left, self.bottom = left, bottom
        self.width, self.height, self.cell = width, height, cell
        self.cells = arc.SpriteList()
        gap = 1 if cell >= 6 else 0
        for y in range(height):
            for x in range(width):
                cx, cy = self.center(x, y)
                self.cells.append(
                    arc.SpriteSolidColor(cell - gap, cell - gap, center_x=cx, center_y=cy)
                )
        self.colors = np.full((height, width, 3), -1, dtype=np.int16)

    def center(self, x: int, y: int) -> tuple[float, float]:
        cx = self.left + x * self.cell + self.cell / 2
        cy = self.bottom + (self.height - 1 - y) * self.cell + self.cell / 2
        return cx, cy

    def sync(self, colors: np.ndarray) -> None:
        colors = colors.astype(np.int16)
        for y, x in np.argwhere((colors != self.colors).any(axis=2)):
            self.cells[int(y) * self.width + int(x)].color = tuple(int(c) for c in colors[y, x])
        self.colors = colors


class CompareWindow(arcade.Window if arcade is not None else object):  # type: ignore[misc]
    """Deux rejeux côte à côte pilotés par une :class:`ComparisonSession`."""

    def __init__(
        self, session: ComparisonSession, cell: int | None = None, speed: float = 10.0
    ) -> None:
        arc = _require_arcade()
        self._arc = arc
        self.session = session
        cfgs = [log.config for log in session.logs]
        self.cell = cell or auto_cell(
            sum(c.width for c in cfgs), max(c.height for c in cfgs), target=1100
        )
        grid_h = max(c.height for c in cfgs) * self.cell
        widths = [c.width * self.cell for c in cfgs]
        width = sum(widths) + 3 * MARGIN
        height = grid_h + TIMELINE_HEIGHT + PANEL_HEIGHT + 3 * MARGIN
        super().__init__(width, height, "Tron — comparaison de rejeux", update_rate=1 / 60)
        self.palettes = session.palettes()
        self.background_color = self.palettes[0].surface
        bottom = PANEL_HEIGHT + TIMELINE_HEIGHT + 2 * MARGIN
        lefts = [MARGIN, 2 * MARGIN + widths[0]]
        self.boards = [
            _Board(arc, lefts[i], bottom, cfgs[i].width, cfgs[i].height, self.cell)
            for i in range(2)
        ]
        self.paused = True
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.show_voronoi = False
        self._acc = 0.0
        ink = self.palettes[0].ink
        muted = self.palettes[0].ink_secondary
        self.t_titles = [
            arc.Text(
                "",
                lefts[i],
                height - MARGIN - 4,
                color=ink,
                font_size=12,
                font_name=FONT,
                bold=True,
            )
            for i in range(2)
        ]
        self.t_events = [
            arc.Text(
                "", lefts[i], PANEL_HEIGHT + MARGIN + 4, color=muted, font_size=9, font_name=FONT
            )
            for i in range(2)
        ]
        self.t_warning = arc.Text(
            "",
            MARGIN,
            PANEL_HEIGHT - 6,
            color=(220, 90, 60),
            font_size=11,
            font_name=FONT,
            bold=True,
        )
        self.t_rows = [
            arc.Text("", MARGIN, PANEL_HEIGHT - 26 - 17 * i, color=ink, font_size=9, font_name=FONT)
            for i in range(13)
        ]
        self.t_help = arc.Text(_HELP, MARGIN, 8, color=muted, font_size=9, font_name=FONT)

    # -------------------------------------------------------------- dessin
    def on_draw(self) -> None:
        arc = self._arc
        self.clear()
        s = self.session
        for side, board in enumerate(self.boards):
            st = s.sources[side].state
            owners = s.owners(side) if self.show_voronoi else None
            board.sync(color_grid(st, self.palettes[side], owners))
            board.cells.draw()
            half = board.cell / 2
            for p in range(st.n):
                cx, cy = board.center(*st.head(p))
                if st.is_alive(p):
                    arc.draw_lrbt_rectangle_outline(
                        cx - half,
                        cx + half,
                        cy - half,
                        cy + half,
                        self.palettes[side].ink,
                        border_width=2,
                    )
                else:
                    arc.draw_line(
                        cx - half, cy - half, cx + half, cy + half, self.palettes[side].ink, 2
                    )
            if s.frozen(side):  # partie terminée : affichage figé, signalé par un cadre
                arc.draw_lrbt_rectangle_outline(
                    board.left - 3,
                    board.left + board.width * board.cell + 3,
                    board.bottom - 3,
                    board.bottom + board.height * board.cell + 3,
                    self.palettes[side].muted,
                    border_width=3,
                )
            self._draw_timeline(side, board)
            view = s.view(side)
            title = f"{view.label} · {view.name} · demi-coup {view.shown_ply}/{view.length}"
            self.t_titles[side].text = title + (" · figé" if view.frozen else "")
            self.t_titles[side].draw()
        self.t_warning.text = s.pairing.warning
        if s.pairing.warning:
            self.t_warning.draw()
        for text, (label, va, vb) in zip(self.t_rows, s.rows(), strict=False):
            text.text = f"{label} : {va}" if va == vb else f"{label} : {va} | {vb}"
            text.draw()
        self.t_help.draw()

    def _draw_timeline(self, side: int, board: _Board) -> None:
        """Ligne de temps des événements du journal et curseur commun."""
        arc = self._arc
        s = self.session
        left = board.left
        right = board.left + board.width * board.cell
        y0 = PANEL_HEIGHT + MARGIN + 20
        arc.draw_lrbt_rectangle_filled(left, right, y0, y0 + 4, self.palettes[side].grid)
        total = max(1, s.max_ply)
        for ev in s.timelines[side]:
            x = left + (right - left) * ev.ply / total
            color = self.palettes[side].ink_secondary
            arc.draw_line(x, y0 - 4, x, y0 + 10, color, 1)
        x = left + (right - left) * s.ply / total
        arc.draw_line(x, y0 - 8, x, y0 + 14, self.palettes[side].ink, 2)
        recent = s.view(side).recent
        self.t_events[side].text = " · ".join(f"{e.ply}: {e.text}" for e in recent) or "—"
        self.t_events[side].draw()

    # ---------------------------------------------------------- événements
    def on_update(self, delta_time: float) -> None:
        if self.paused:
            return
        self._acc += delta_time * SPEEDS[self.speed_index]
        while self._acc >= 1.0:
            self._acc -= 1.0
            if not self.session.forward():
                self.paused = True
                break

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        key = self._arc.key
        s = self.session
        if symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.RIGHT:
            self.paused = True
            s.forward()
        elif symbol == key.LEFT:
            self.paused = True
            s.backward()
        elif symbol == key.UP:
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
        elif symbol == key.DOWN:
            self.speed_index = max(self.speed_index - 1, 0)
        elif symbol == key.HOME:
            s.start()
        elif symbol == key.END:
            s.end()
        elif symbol == key.V:
            self.show_voronoi = not self.show_voronoi
        elif symbol in (key.ESCAPE, key.Q):
            self.close()


def run_compare(session: ComparisonSession, cell: int | None = None, speed: float = 10) -> None:
    """Ouvre la fenêtre de comparaison."""
    arc = _require_arcade()
    CompareWindow(session, cell=cell, speed=speed)
    arc.run()


__all__ = ["CompareWindow", "run_compare"]
