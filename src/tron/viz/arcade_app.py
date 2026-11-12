"""Fenêtre arcade 3.3 : temps réel et rejeu (F6.1 à F6.3).

Touches :

* ``Espace`` : pause / lecture
* ``→`` et ``←`` : demi-coup suivant / précédent (le retour arrière n'existe qu'en rejeu)
* ``↑`` et ``↓`` : vitesse
* ``V`` : surcouche des territoires de Voronoi
* ``D`` : panneau de la dernière décision (coup, valeurs des alternatives,
  profondeur, nœuds, variante principale dessinée sur la grille, poids sociaux)
* ``N`` : nouvelle partie (temps réel) ou retour au début (rejeu)
* ``Début`` et ``Fin`` : début et fin de partie (rejeu)
* ``S`` : capture d'écran (PNG)
* ``Échap`` ou ``Q`` : quitter

La grille est une ``SpriteList`` de cases colorées (rendu groupé sur le GPU). Seules
les cases dont la couleur a changé sont mises à jour à chaque image. Le moteur
ignore tout de cette fenêtre : elle se contente d'observer un ``Match`` ou de lire
un journal.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from tron.core.config import GameConfig
from tron.sim.recorder import GameLog
from tron.viz.controller import LiveSource, PlaybackController, ReplaySource, Source
from tron.viz.palette import label_ink, make_palette, rgb_to_hex
from tron.viz.render import color_grid

try:
    import arcade
except ImportError:  # pragma: no cover - dépend de l'installation
    arcade = None

PANEL_WIDTH = 340
MARGIN = 14
FONT = ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
_HELP = (
    "Espace pause · → ← pas · ↑ ↓ vitesse",
    "V Voronoi · D décision · N nouvelle · S capture · Échap",
)
DECISION_LINES = 10


def _require_arcade() -> Any:
    if arcade is None:
        raise RuntimeError(
            "la bibliothèque arcade n'est pas installée : pip install 'arcade==3.3.3'"
        )
    return arcade


def auto_cell(width: int, height: int, target: int = 860) -> int:
    """Taille de case en pixels pour que la grille tienne dans ~``target`` pixels."""
    return max(4, min(36, target // max(width, height)))


class TronWindow(arcade.Window if arcade is not None else object):  # type: ignore[misc]
    """Fenêtre de visualisation pilotée par un :class:`PlaybackController`."""

    def __init__(
        self,
        controller: PlaybackController,
        cell: int | None = None,
        screenshot_dir: str | Path = "screenshots",
    ) -> None:
        arc = _require_arcade()
        self.controller = controller
        st = controller.source.state
        self.cell = cell or auto_cell(st.width, st.height)
        self.grid_w = st.width * self.cell
        self.grid_h = st.height * self.cell
        width = self.grid_w + 2 * MARGIN + PANEL_WIDTH
        height = max(self.grid_h + 2 * MARGIN, 560)
        super().__init__(width, height, "Tron — coalitions", update_rate=1 / 60)
        self.screenshot_dir = Path(screenshot_dir)
        self._arc = arc
        self._setup_scene()

    # --------------------------------------------------------------- scène
    def _setup_scene(self) -> None:
        arc = self._arc
        st = self.controller.source.state
        cfg = st.config
        self.palette = make_palette(cfg.team_index)
        self.background_color = self.palette.surface
        self._shape = (st.height, st.width)
        self.cells = arc.SpriteList()
        gap = 1 if self.cell >= 6 else 0
        size = self.cell - gap
        for y in range(st.height):
            for x in range(st.width):
                cx, cy = self._center(x, y)
                self.cells.append(
                    arc.SpriteSolidColor(
                        size, size, center_x=cx, center_y=cy, color=self.palette.surface
                    )
                )
        self._colors = np.full((st.height, st.width, 3), -1, dtype=np.int16)
        ink = self.palette.ink
        self.head_labels = [
            arc.Text(
                str(p),
                0,
                0,
                color=label_ink(self.palette.head[p]),
                font_size=max(7, int(self.cell * 0.45)),
                anchor_x="center",
                anchor_y="center",
                font_name=FONT,
                bold=True,
            )
            for p in range(cfg.n_players)
        ]
        left = self.grid_w + 2 * MARGIN
        self._panel_left = left
        self.t_title = arc.Text(
            "",
            left,
            self.height - 30,
            color=ink,
            font_size=13,
            font_name=FONT,
            bold=True,
            width=PANEL_WIDTH - MARGIN,
            multiline=True,
        )
        self.t_status = arc.Text(
            "",
            left,
            self.height - 56,
            color=self.palette.ink_secondary,
            font_size=11,
            font_name=FONT,
        )
        self.t_ply = arc.Text(
            "",
            left,
            self.height - 76,
            color=self.palette.ink_secondary,
            font_size=11,
            font_name=FONT,
        )
        self.t_players = []
        top = self.height - 108
        for p in range(cfg.n_players):
            y = top - p * 46
            self.t_players.append(
                (
                    arc.Text("", left + 22, y, color=ink, font_size=11, font_name=FONT, bold=True),
                    arc.Text(
                        "",
                        left + 22,
                        y - 17,
                        color=self.palette.ink_secondary,
                        font_size=9,
                        font_name=FONT,
                    ),
                )
            )
        self._teams_top = top - cfg.n_players * 46 - 6
        self.t_teams = [
            arc.Text(
                "", left + 22, self._teams_top - 20 * t, color=ink, font_size=10, font_name=FONT
            )
            for t in range(len(cfg.team_names))
        ]
        self.t_help = [
            arc.Text(line, left, 34 - 16 * i, color=self.palette.muted, font_size=9, font_name=FONT)
            for i, line in enumerate(_HELP)
        ]
        self.t_decision = [
            arc.Text(
                "",
                MARGIN + 8,
                MARGIN + 8 + 15 * (DECISION_LINES - 1 - i),
                color=ink,
                font_size=9,
                font_name=FONT,
            )
            for i in range(DECISION_LINES)
        ]

    def _center(self, x: int, y: int) -> tuple[float, float]:
        """Centre écran de la case (x, y) ; l'axe y d'arcade monte, celui de la grille descend."""
        st_h = self.grid_h // self.cell
        cx = MARGIN + x * self.cell + self.cell / 2
        cy = MARGIN + (st_h - 1 - y) * self.cell + self.cell / 2
        return cx, cy

    def _sync_cells(self) -> None:
        st = self.controller.source.state
        if (st.height, st.width) != self._shape:  # pragma: no cover - nouvelle grille
            self._setup_scene()
        colors = color_grid(st, self.palette, self.controller.owners()).astype(np.int16)
        changed = np.argwhere((colors != self._colors).any(axis=2))
        w = st.width
        for y, x in changed:
            self.cells[int(y) * w + int(x)].color = tuple(int(c) for c in colors[y, x])
        self._colors = colors

    # ------------------------------------------------------------- dessin
    def on_draw(self) -> None:
        arc = self._arc
        self.clear()
        self._sync_cells()
        self.cells.draw()
        st = self.controller.source.state
        half = self.cell / 2
        for zone in st.config.rules.zones:  # contrôle de zone (P09.2)
            x0, y0 = self._center(zone.x, zone.y)
            x1, y1 = self._center(zone.x + zone.width - 1, zone.y + zone.height - 1)
            arc.draw_lrbt_rectangle_outline(
                min(x0, x1) - half,
                max(x0, x1) + half,
                min(y0, y1) - half,
                max(y0, y1) + half,
                self.palette.ink_secondary,
                border_width=2,
            )
        for p in range(st.n):
            x, y = st.head(p)
            cx, cy = self._center(x, y)
            if st.is_alive(p):
                arc.draw_lrbt_rectangle_outline(
                    cx - half, cx + half, cy - half, cy + half, self.palette.ink, border_width=2
                )
                if self.cell >= 14:
                    label = self.head_labels[p]
                    label.x, label.y = cx, cy
                    label.draw()
            else:
                arc.draw_line(cx - half, cy - half, cx + half, cy + half, self.palette.ink, 2)
                arc.draw_line(cx - half, cy + half, cx + half, cy - half, self.palette.ink, 2)
        if self.controller.show_decision:
            self._draw_decision()
        self._draw_panel()

    def _draw_decision(self) -> None:
        """Variante principale sur la grille et panneau texte de la dernière décision."""
        arc = self._arc
        q = self.cell / 3
        for p, x, y in self.controller.decision_path():
            cx, cy = self._center(x, y)
            arc.draw_lrbt_rectangle_filled(
                cx - q / 2, cx + q / 2, cy - q / 2, cy + q / 2, self.palette.head[p]
            )
        panel = self.controller.decision_panel()
        lines = panel.lines() if panel is not None else ["Décision : aucune pour l'instant"]
        lines = lines[:DECISION_LINES]
        height = 15 * len(lines) + 12
        arc.draw_lrbt_rectangle_filled(
            MARGIN, MARGIN + self.grid_w, MARGIN, MARGIN + height, self.palette.grid
        )
        for i, line in enumerate(lines):
            text = self.t_decision[i]
            text.text = line
            text.y = MARGIN + 8 + 15 * (len(lines) - 1 - i)
            text.draw()

    def _draw_panel(self) -> None:
        arc = self._arc
        info = self.controller.panel()
        left = self._panel_left
        arc.draw_lrbt_rectangle_filled(
            left - MARGIN // 2, self.width, 0, self.height, self.palette.grid
        )
        self.t_title.text = info.title
        self.t_status.text = info.status
        cur = info.players[info.current].name if info.players else "-"
        self.t_ply.text = f"Demi-coup {info.ply} · {info.moves} coups · au trait : {cur}"
        for t in (self.t_title, self.t_status, self.t_ply):
            t.draw()
        for line, (t_main, t_sub) in zip(info.players, self.t_players, strict=True):
            color = self.palette.player[line.index] if line.alive else self.palette.dead[line.index]
            y = t_main.y
            arc.draw_lrbt_rectangle_filled(left, left + 14, y - 2, y + 12, color)
            state = "vivant" if line.alive else "éliminé"
            if line.rank is not None:
                state = f"rang {line.rank}"
            t_main.text = f"{line.index} · {line.name} [{line.team}] · {state}"
            extra = []
            if line.nodes is not None:
                extra.append(f"{line.nodes} nœuds")
            if line.depth is not None:
                extra.append(f"prof. {line.depth}")
            if line.value is not None:
                extra.append(f"v={line.value:.3f}")
            t_sub.text = f"{line.agent} · territoire {100 * line.share:.1f} %" + (
                " · " + ", ".join(extra) if extra else ""
            )
            t_main.draw()
            t_sub.draw()
        for (name, share, alive), text, t in zip(
            info.teams, self.t_teams, range(len(self.t_teams)), strict=True
        ):
            y = text.y
            arc.draw_lrbt_rectangle_filled(left, left + 14, y - 2, y + 10, self.palette.team[t])
            text.text = f"Équipe {name} : {100 * share:.1f} % · {alive} vivant(s)"
            text.draw()
        for t in self.t_help:
            t.draw()

    # ---------------------------------------------------------- événements
    def on_update(self, delta_time: float) -> None:
        self.controller.update(delta_time)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        key = self._arc.key
        ctl = self.controller
        if symbol == key.SPACE:
            ctl.toggle_pause()
        elif symbol == key.RIGHT:
            ctl.paused = True
            ctl.step_forward()
        elif symbol == key.LEFT:
            ctl.paused = True
            ctl.step_backward()
        elif symbol == key.UP:
            ctl.faster()
        elif symbol == key.DOWN:
            ctl.slower()
        elif symbol == key.V:
            ctl.toggle_voronoi()
        elif symbol == getattr(key, "D", None):
            ctl.toggle_decision()
        elif symbol == key.N:
            ctl.new_game()
        elif symbol == key.HOME and isinstance(ctl.source, ReplaySource):
            ctl.source.seek(0)
        elif symbol == key.END and isinstance(ctl.source, ReplaySource):
            ctl.source.seek(len(ctl.source.log.moves))
        elif symbol == key.S:
            self.screenshot()
        elif symbol in (key.ESCAPE, key.Q):
            self.close()

    def screenshot(self) -> Path:
        """Capture de la fenêtre au format PNG."""
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"tron_{time.strftime('%Y%m%d_%H%M%S')}.png"
        image = self._arc.get_image(0, 0, self.width, self.height)
        image.save(path)
        return path


def _run(source: Source, cell: int | None, speed: float, voronoi: bool, paused: bool) -> None:
    arc = _require_arcade()
    controller = PlaybackController(source, speed=speed, voronoi=voronoi, paused=paused)
    TronWindow(controller, cell=cell)
    arc.run()


def run_live(
    config: GameConfig,
    cell: int | None = None,
    speed: float = 10,
    voronoi: bool = False,
    paused: bool = False,
) -> None:
    """Ouvre la fenêtre et joue la partie en temps réel."""
    _run(LiveSource(config), cell, speed, voronoi, paused)


def run_replay(
    log: GameLog,
    cell: int | None = None,
    speed: float = 10,
    voronoi: bool = False,
    paused: bool = True,
) -> None:
    """Ouvre la fenêtre de rejeu d'un journal JSONL."""
    _run(ReplaySource(log), cell, speed, voronoi, paused)


__all__ = ["TronWindow", "auto_cell", "rgb_to_hex", "run_live", "run_replay"]
