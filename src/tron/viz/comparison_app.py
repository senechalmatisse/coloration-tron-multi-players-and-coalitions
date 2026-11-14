"""Fenêtre Arcade de comparaison de deux rejeux.

Deux plateaux côte à côte, synchronisés par :class:`ComparisonSession`, une ligne
de temps d'événements sous chaque plateau et un panneau de comparaison. Touches :
``Espace`` lecture / pause, ``→`` / ``←`` demi-coup suivant / précédent,
``↑`` / ``↓`` vitesse, ``Début`` / ``Fin``, ``V`` territoires de Voronoi,
``Échap`` / ``Q`` quitter. Toute la logique (curseur, données, avertissement
d'appariement) est dans :mod:`tron.viz.comparison`, testée sans fenêtre.

Thème néon : deux arènes de :mod:`tron.viz.neon_arcade` (traces lumineuses,
motos, effets) à la place des grilles de cases colorées.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tron.viz.arcade_app import FONT, MARGIN, _require_arcade, auto_cell
from tron.viz.comparison import ComparisonSession
from tron.viz.controller import SPEEDS
from tron.viz.hud import fit
from tron.viz.neon_arcade import NeonArena, TextureBank
from tron.viz.render import color_grid
from tron.viz.theme import (
    DEFAULT_QUALITY,
    Quality,
    arena_padding,
    make_neon_theme,
    resolve_quality,
    resolve_theme,
)

try:
    import arcade
except ImportError:  # pragma: no cover - dépend de l'installation
    arcade = None

TIMELINE_HEIGHT = 46
TITLE_HEIGHT = 24
"""Bandeau des titres au-dessus des plateaux."""
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
        self,
        session: ComparisonSession,
        cell: int | None = None,
        speed: float = 10.0,
        theme: str = "classic",
        quality: str | Quality = DEFAULT_QUALITY,
    ) -> None:
        arc = _require_arcade()
        self._arc = arc
        self.session = session
        self.theme_name = resolve_theme(theme)
        self.quality = resolve_quality(quality)
        cfgs = [log.config for log in session.logs]
        self.cell = cell or auto_cell(
            sum(c.width for c in cfgs), max(c.height for c in cfgs), target=1100
        )
        margin = max(MARGIN, arena_padding(self.cell)) if self.neon else MARGIN
        self.margin = margin
        grid_h = max(c.height for c in cfgs) * self.cell
        widths = [c.width * self.cell for c in cfgs]
        width = sum(widths) + 3 * margin
        height = grid_h + TIMELINE_HEIGHT + PANEL_HEIGHT + TITLE_HEIGHT + 3 * margin
        super().__init__(width, height, "Tron — comparaison de rejeux", update_rate=1 / 60)
        self.palettes = session.palettes()
        self.background_color = self.palettes[0].surface
        bottom = PANEL_HEIGHT + TIMELINE_HEIGHT + 2 * margin
        lefts = [margin, 2 * margin + widths[0]]
        self.arenas: list[NeonArena] = []
        if self.neon:
            self.neon_themes = tuple(make_neon_theme(c.team_index, c.team_names) for c in cfgs)
            self.background_color = self.neon_themes[0].background
            bank = TextureBank(arc)
            self.arenas = [
                NeonArena(
                    arc,
                    bank,
                    cfgs[i],
                    lefts[i],
                    bottom,
                    self.cell,
                    self.neon_themes[i],
                    self.quality,
                    self.ctx.BLEND_ADDITIVE,
                    font=FONT,
                )
                for i in range(2)
            ]
            self.boards: list[Any] = list(self.arenas)
        else:
            self.boards = [
                _Board(arc, lefts[i], bottom, cfgs[i].width, cfgs[i].height, self.cell)
                for i in range(2)
            ]
        self._lefts = lefts
        self._title_chars = [max(10, int(w / 9.0)) for w in widths]
        self._bottom = bottom
        self._clock = 0.0
        self.paused = True
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.show_voronoi = False
        self._acc = 0.0
        ink = self.palettes[0].ink
        muted = self.palettes[0].ink_secondary
        if self.neon:
            ink = self.neon_themes[0].text
            muted = self.neon_themes[0].text_dim
        self.t_titles = [
            arc.Text(
                "",
                lefts[i],
                height - 17,
                color=ink,
                font_size=11,
                font_name=FONT,
                bold=True,
            )
            for i in range(2)
        ]
        self.t_events = [
            arc.Text(
                "", lefts[i], PANEL_HEIGHT + margin + 4, color=muted, font_size=9, font_name=FONT
            )
            for i in range(2)
        ]
        self.t_warning = arc.Text(
            "",
            margin,
            PANEL_HEIGHT - 6,
            color=(220, 90, 60),
            font_size=11,
            font_name=FONT,
            bold=True,
        )
        self.t_rows = [
            arc.Text("", margin, PANEL_HEIGHT - 26 - 17 * i, color=ink, font_size=9, font_name=FONT)
            for i in range(13)
        ]
        self.t_help = arc.Text(_HELP, margin, 8, color=muted, font_size=9, font_name=FONT)

    @property
    def neon(self) -> bool:
        return self.theme_name == "neon"

    # dessin
    def on_draw(self) -> None:
        if self.neon:
            self._draw_neon()
            return
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
            title += " · figé" if view.frozen else ""
            self.t_titles[side].text = fit(title, self._title_chars[side])
            self.t_titles[side].draw()
        self.t_warning.text = s.pairing.warning
        if s.pairing.warning:
            self.t_warning.draw()
        for text, (label, va, vb) in zip(self.t_rows, s.rows(), strict=False):
            text.text = f"{label} : {va}" if va == vb else f"{label} : {va} | {vb}"
            text.draw()
        self.t_help.draw()

    def _sync_neon(self) -> None:
        s = self.session
        speed = 1.0 if self.paused else SPEEDS[self.speed_index]
        for side, arena in enumerate(self.arenas):
            src = s.sources[side]
            owners = s.owners(side) if self.show_voronoi else None
            arena.sync(src.state, src.extras, speed, owners)

    def _draw_neon(self) -> None:
        """Deux arènes néon, cadre « figé », lignes de temps et panneau de comparaison."""
        arc = self._arc
        self._sync_neon()
        self.clear()
        s = self.session
        for side, arena in enumerate(self.arenas):
            theme = self.neon_themes[side]
            st = s.sources[side].state
            arena.draw(st, self._clock)
            if s.frozen(side):
                pad = arena_padding(self.cell) - 2
                arc.draw_lrbt_rectangle_outline(
                    arena.left - pad,
                    arena.left + arena.pixel_width + pad,
                    arena.bottom - pad,
                    arena.bottom + arena.pixel_height + pad,
                    theme.text_muted,
                    border_width=3,
                )
            self._draw_timeline(side, arena)
            view = s.view(side)
            title = f"{view.label} · {view.name} · demi-coup {view.shown_ply}/{view.length}"
            title += " · figé" if view.frozen else ""
            self.t_titles[side].text = fit(title, self._title_chars[side])
            self.t_titles[side].draw()
        self.t_warning.text = s.pairing.warning
        if s.pairing.warning:
            self.t_warning.draw()
        for text, (label, va, vb) in zip(self.t_rows, s.rows(), strict=False):
            text.text = f"{label} : {va}" if va == vb else f"{label} : {va} | {vb}"
            text.draw()
        self.t_help.draw()

    def _draw_timeline(self, side: int, board: Any) -> None:
        """Ligne de temps des événements du journal et curseur commun."""
        arc = self._arc
        s = self.session
        left = board.left
        right = board.left + board.width * board.cell
        y0 = PANEL_HEIGHT + self.margin + 20
        if self.neon:
            theme = self.neon_themes[side]
            track, tick, cursor = theme.grid_major, theme.text_dim, theme.accent
        else:
            pal = self.palettes[side]
            track, tick, cursor = pal.grid, pal.ink_secondary, pal.ink
        arc.draw_lrbt_rectangle_filled(left, right, y0, y0 + 4, track)
        total = max(1, s.max_ply)
        for ev in s.timelines[side]:
            x = left + (right - left) * ev.ply / total
            arc.draw_line(x, y0 - 4, x, y0 + 10, tick, 1)
        x = left + (right - left) * s.ply / total
        arc.draw_line(x, y0 - 8, x, y0 + 14, cursor, 2)
        recent = s.view(side).recent
        self.t_events[side].text = " · ".join(f"{e.ply}: {e.text}" for e in recent) or "—"
        self.t_events[side].draw()

    # événements
    def on_update(self, delta_time: float) -> None:
        self._clock += delta_time
        for arena in self.arenas:
            arena.update(delta_time)
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


def run_compare(
    session: ComparisonSession,
    cell: int | None = None,
    speed: float = 10,
    theme: str = "classic",
    quality: str | Quality = DEFAULT_QUALITY,
) -> None:
    """Ouvre la fenêtre de comparaison (``theme`` : ``neon`` ou ``classic``)."""
    arc = _require_arcade()
    CompareWindow(session, cell=cell, speed=speed, theme=theme, quality=quality)
    arc.run()


__all__ = ["CompareWindow", "run_compare"]
