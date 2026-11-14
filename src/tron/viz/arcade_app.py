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
* ``T`` : thème néon / classique (palette validée pour les daltoniens)
* ``G`` : qualité graphique du thème néon (low → medium → ultra)
* ``B`` : post-traitement *bloom* (thème néon)
* ``A`` : liens entre coéquipiers (thème néon ; les alliances tactiques sont toujours tracées)
* ``S`` : capture d'écran (PNG)
* ``Échap`` ou ``Q`` : quitter

Thème classique : la grille est une ``SpriteList`` de cases colorées (rendu
groupé sur le GPU) dont seules les cases changées sont mises à jour. Thème
néon : arène, traces lumineuses, motos, effets et HUD holographique de
:mod:`tron.viz.neon_arcade`, avec les images de :mod:`tron.viz.neon` (les
mêmes que celles des exports). Le moteur ignore tout de cette fenêtre : elle
se contente d'observer un ``Match`` ou de lire un journal.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from tron.core.config import GameConfig
from tron.sim.recorder import GameLog
from tron.viz.controller import LiveSource, PlaybackController, ReplaySource, Source
from tron.viz.effects.bloom import Bloom
from tron.viz.hud import EventLog, hud_stats, player_cards
from tron.viz.neon_arcade import NeonArena, NeonHud, TextureBank, victory_banner
from tron.viz.palette import label_ink, make_palette, rgb_to_hex
from tron.viz.render import color_grid
from tron.viz.scene import team_links
from tron.viz.theme import (
    DEFAULT_QUALITY,
    Quality,
    arena_padding,
    make_neon_theme,
    next_quality,
    resolve_quality,
    resolve_theme,
)

try:
    import arcade
except ImportError:  # pragma: no cover - dépend de l'installation
    arcade = None

PANEL_WIDTH = 340
MARGIN = 14
MIN_HEIGHT = 640
FONT = ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
_HELP = (
    "Espace pause · → ← pas · ↑ ↓ vitesse",
    "V Voronoi · D décision · T thème · N nouvelle · S capture · Échap",
)
_HELP_NEON = (
    "Espace pause · → ← pas · ↑ ↓ vitesse · N nouvelle",
    "V territoires · D décision · A coéquipiers",
    "T thème · G qualité · B bloom · S capture · Échap",
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
        theme: str = "classic",
        quality: str | Quality = DEFAULT_QUALITY,
    ) -> None:
        arc = _require_arcade()
        self.controller = controller
        self.theme_name = resolve_theme(theme)
        self.quality = resolve_quality(quality)
        st = controller.source.state
        self.cell = cell or auto_cell(st.width, st.height)
        self.margin = max(MARGIN, arena_padding(self.cell))
        self.grid_w = st.width * self.cell
        self.grid_h = st.height * self.cell
        width = self.grid_w + 2 * self.margin + PANEL_WIDTH
        height = max(self.grid_h + 2 * self.margin, MIN_HEIGHT)
        super().__init__(width, height, "Tron — coalitions", update_rate=1 / 60)
        self.screenshot_dir = Path(screenshot_dir)
        self._arc = arc
        self._bank: TextureBank | None = None
        self.bloom: Bloom | None = None
        self.bloom_on = self.quality.bloom
        self.show_team_links = False
        self._clock = 0.0
        self._elapsed = 0.0
        self._camera: Any = None
        self._setup_scene()

    # --------------------------------------------------------------- scène
    @property
    def neon(self) -> bool:
        return self.theme_name == "neon"

    def _setup_scene(self) -> None:
        if self.neon:
            self._setup_neon()
        else:
            self._setup_classic()

    def _setup_classic(self) -> None:
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
        left = self.grid_w + 2 * self.margin
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
        self.t_decision = self._decision_texts(ink)

    def _decision_texts(self, ink: Any) -> list[Any]:
        return [
            self._arc.Text(
                "",
                self.margin + 8,
                self.margin + 8 + 15 * (DECISION_LINES - 1 - i),
                color=ink,
                font_size=9,
                font_name=FONT,
            )
            for i in range(DECISION_LINES)
        ]

    def _setup_neon(self) -> None:
        arc = self._arc
        st = self.controller.source.state
        cfg = st.config
        theme = make_neon_theme(cfg.team_index, cfg.team_names)
        self.neon_theme = theme
        self.background_color = theme.background
        self._shape = (st.height, st.width)
        if self._bank is None:
            self._bank = TextureBank(arc)
        self._bottom = max(self.margin, (self.height - self.grid_h) // 2)
        self.arena = NeonArena(
            arc,
            self._bank,
            cfg,
            self.margin,
            self._bottom,
            self.cell,
            theme,
            self.quality,
            self.ctx.BLEND_ADDITIVE,
            font=FONT,
        )
        self.cells = self.arena.cells
        left = self.grid_w + 2 * self.margin
        self._panel_left = left
        self.hud = NeonHud(arc, theme, left + 6, self.width, self.height, cfg.n_players, _HELP_NEON)
        self.event_log = EventLog(size=len(self.hud.t_log))
        self.t_decision = self._decision_texts(theme.text)
        self.t_banner = arc.Text(
            "",
            0,
            0,
            color=theme.text,
            font_size=18,
            anchor_x="center",
            anchor_y="center",
            font_name=FONT,
            bold=True,
        )
        camera = getattr(arc, "Camera2D", None)
        if camera is not None and self._camera is None:
            try:
                self._camera = camera()
            except Exception:  # pragma: no cover - dépend du contexte OpenGL
                self._camera = None

    def _center(self, x: int, y: int) -> tuple[float, float]:
        """Centre écran de la case (x, y) ; l'axe y d'arcade monte, celui de la grille descend."""
        st_h = self.grid_h // self.cell
        cx = self.margin + x * self.cell + self.cell / 2
        cy = self.margin + (st_h - 1 - y) * self.cell + self.cell / 2
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
        if self.neon:
            self._draw_neon()
            return
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
            self.margin,
            self.margin + self.grid_w,
            self.margin,
            self.margin + height,
            self.palette.grid,
        )
        self._decision_lines(lines, self.margin)

    def _decision_lines(self, lines: list[str], bottom: float) -> None:
        for i, line in enumerate(lines):
            text = self.t_decision[i]
            text.text = line
            text.y = bottom + 8 + 15 * (len(lines) - 1 - i)
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

    # ------------------------------------------------------- dessin (néon)
    def _sync_neon(self) -> None:
        ctl = self.controller
        st = ctl.source.state
        if (st.height, st.width) != self._shape or st.config.n_players != len(
            self.arena.bikes
        ):  # pragma: no cover - autre grille
            self._setup_scene()
        speed = 1.0 if ctl.paused else ctl.speed
        result = self.arena.sync(st, ctl.ply_extras, speed, ctl.owners())
        if result.reset:
            self.event_log.clear()
            self._elapsed = 0.0
        elif result.removed:
            self.event_log.prune(st.ply)
        self.event_log.add(result.events, st, self.neon_theme)

    def _draw_neon(self) -> None:
        ctl = self.controller
        self._sync_neon()
        st = ctl.source.state
        t = self._clock
        alliances = ctl.alliances()
        links = list(alliances)
        if self.show_team_links:
            links += [pair for pair in team_links(st) if pair not in alliances]
        path = ctl.decision_path()
        self.clear()
        bloom = self._bloom() if self.bloom_on else None
        drawn_with_bloom = bloom is not None and bloom.begin(self.neon_theme.background)
        shaken = self._begin_shake()
        self.arena.draw(st, t, links, path)
        if shaken:
            self._default_camera()
        if drawn_with_bloom and bloom is not None and not bloom.end():
            self.clear()  # repli : la scène est redessinée sans bloom
            self.arena.draw(st, t, links, path)
        self._default_camera()
        if st.is_terminal():
            self._draw_banner(st)
        if ctl.show_decision:
            self._draw_decision_neon()
        self._draw_hud(st, alliances)

    def _bloom(self) -> Bloom | None:
        if self.bloom is None:
            self.bloom = Bloom(self)
        return self.bloom if self.bloom.enabled else None

    def _begin_shake(self) -> bool:
        """Tremblement léger (qualité ``medium`` ou ``ultra``) ; ``False`` si la caméra est fixe."""
        if self._camera is None or not self.quality.shake:
            return False
        dx, dy = self.arena.shake_pixels()
        if dx == 0 and dy == 0:
            return False
        try:
            self._camera.position = (self.width / 2 - dx, self.height / 2 + dy)
            self._camera.use()
            return True
        except Exception:  # pragma: no cover - API de caméra indisponible
            self._camera = None
            return False

    def _default_camera(self) -> None:
        camera = getattr(self, "default_camera", None)
        if camera is not None:
            with contextlib.suppress(Exception):  # dépend d'Arcade
                camera.use()

    def _draw_banner(self, st: Any) -> None:
        out = st.outcome()
        cfg = st.config
        if out is None or out.winning_team < 0:
            label, color = "ÉGALITÉ", self.neon_theme.text
        else:
            label = f"VICTOIRE · {cfg.victory_names[out.winning_team]}"
            winner = out.winners[0] if out.winners else 0
            color = self.neon_theme.player[winner]
        victory_banner(self._arc, self.arena, self.t_banner, label, color, self._clock)

    def _draw_decision_neon(self) -> None:
        arc = self._arc
        theme = self.neon_theme
        panel = self.controller.decision_panel()
        lines = panel.lines() if panel is not None else ["Décision : aucune pour l'instant"]
        lines = lines[:DECISION_LINES]
        height = 15 * len(lines) + 12
        x0, x1 = self.margin, self.margin + self.grid_w
        y0, y1 = self._bottom, self._bottom + height
        arc.draw_lrbt_rectangle_filled(x0, x1, y0, y1, theme.panel)
        arc.draw_lrbt_rectangle_outline(x0, x1, y0, y1, theme.panel_edge, border_width=1)
        self._decision_lines(lines, y0)

    def _draw_hud(self, st: Any, alliances: tuple[tuple[int, int], ...]) -> None:
        ctl = self.controller
        info = ctl.panel()
        cur = info.players[info.current].name if info.players else "-"
        live = isinstance(ctl.source, LiveSource)
        stats = hud_stats(
            st, len(alliances), self._elapsed if live else None, None if ctl.paused else ctl.speed
        )
        render = self.quality.name + (" · bloom" if self.bloom_on and self._bloom() else "")
        stats.append(("rendu", render))
        out = st.outcome()
        winners = out.winners if out is not None else None
        self.hud.draw(
            info.title,
            info.status,
            f"Demi-coup {info.ply} · {info.moves} coups · au trait : {cur}",
            player_cards(info, self.neon_theme, alliances, winners),
            stats,
            self.event_log.recent(),
            self._clock,
        )

    # ---------------------------------------------------------- événements
    def on_update(self, delta_time: float) -> None:
        ctl = self.controller
        ctl.update(delta_time)
        self._clock += delta_time
        if not ctl.paused and not ctl.source.finished:
            self._elapsed += delta_time
        if self.neon:
            self._sync_neon()
            self.arena.update(delta_time)

    def toggle_theme(self) -> None:
        """Bascule entre le thème néon et le thème classique."""
        self.theme_name = "classic" if self.neon else "neon"
        self._setup_scene()

    def cycle_quality(self) -> None:
        """Niveau de qualité suivant (low → medium → ultra) ; le bloom suit le niveau."""
        self.quality = next_quality(self.quality)
        self.bloom_on = self.quality.bloom
        if self.neon:
            self.arena.set_quality(self.quality)
            self.cells = self.arena.cells
            self.event_log.clear()

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
        elif symbol == getattr(key, "T", None):
            self.toggle_theme()
        elif symbol == getattr(key, "G", None):
            self.cycle_quality()
        elif symbol == getattr(key, "B", None):
            self.bloom_on = not self.bloom_on
        elif symbol == getattr(key, "A", None):
            self.show_team_links = not self.show_team_links
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


def _run(
    source: Source,
    cell: int | None,
    speed: float,
    voronoi: bool,
    paused: bool,
    theme: str,
    quality: str | Quality,
) -> None:
    arc = _require_arcade()
    controller = PlaybackController(source, speed=speed, voronoi=voronoi, paused=paused)
    TronWindow(controller, cell=cell, theme=theme, quality=quality)
    arc.run()


def run_live(
    config: GameConfig,
    cell: int | None = None,
    speed: float = 10,
    voronoi: bool = False,
    paused: bool = False,
    theme: str = "classic",
    quality: str | Quality = DEFAULT_QUALITY,
) -> None:
    """Ouvre la fenêtre et joue la partie en temps réel (``theme`` : ``neon`` ou ``classic``)."""
    _run(LiveSource(config), cell, speed, voronoi, paused, theme, quality)


def run_replay(
    log: GameLog,
    cell: int | None = None,
    speed: float = 10,
    voronoi: bool = False,
    paused: bool = True,
    theme: str = "classic",
    quality: str | Quality = DEFAULT_QUALITY,
) -> None:
    """Ouvre la fenêtre de rejeu d'un journal JSONL (``theme`` : ``neon`` ou ``classic``)."""
    _run(ReplaySource(log), cell, speed, voronoi, paused, theme, quality)


__all__ = ["TronWindow", "auto_cell", "rgb_to_hex", "run_live", "run_replay"]
