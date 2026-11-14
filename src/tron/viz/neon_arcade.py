"""Couche Arcade du thème néon : arène (fond, sol, zones, traces, motos, effets) et HUD.

Toutes les images viennent de :mod:`tron.viz.neon`, les mêmes que celles des
exports : ce module se contente de les placer sous forme de sprites.

* **Traces** : par case, un sprite de lueur (mélange additif) et un sprite de
  cœur (mélange alpha). Un demi-coup ajoute deux sprites et change la texture
  de la case précédente (coût constant) ; un retour arrière fait l'inverse.
* **Motos** : un sprite par joueur (texture selon l'orientation), halo pulsant.
* **Particules** : réserve de sprites recyclés, mélange additif.
* **Fond** : une seule texture (dégradé, grille, lignes majeures, liseré).

Les textures sont créées une fois par clé (:class:`TextureBank`) : changer de
qualité ou de thème en cours de partie ne fait pas grossir l'atlas sans fin.
Le module n'importe pas Arcade : la fenêtre lui passe le module (``arc``), ce
qui permet de le tester avec un bouchon d'API.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tron.core.config import GameConfig
from tron.core.rules import OBSTACLE
from tron.core.state import GameState
from tron.viz.effects.director import EffectDirector, zone_color, zone_controllers
from tron.viz.hud import LogLine, PlayerCard, fit, stamp
from tron.viz.neon import (
    arena_background,
    bike_image,
    default_heading,
    obstacle_tile,
    soft_dot,
    trail_tiles,
)
from tron.viz.palette import RGB, mix
from tron.viz.scene import Cell, PlyExtras, SceneTracker, SyncResult, territory_edges
from tron.viz.theme import (
    FLASH_ALPHA,
    LINK_FLOW_SPEED,
    SCAN_PERIOD,
    SHAKE_MAX_PIXELS,
    NeonTheme,
    Quality,
    arena_padding,
    pulse,
    trail_metrics,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

FONT = ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
STATS_ROWS = 8
"""Lignes du bloc de statistiques du HUD."""
PARTICLE_RADIUS = 16
"""Rayon (pixels) de la texture des particules, mise à l'échelle par sprite."""
_BLACK_CLEAR = (0, 0, 0, 0)


def glow_rgba(intensity: np.ndarray) -> np.ndarray:
    """Texture de lueur pour le mélange additif : RVB = alpha = intensité."""
    v = np.clip(np.rint(intensity * 255.0), 0, 255).astype(np.uint8)
    return np.stack([v, v, v, v], axis=-1)


def cover_rgba(cover: np.ndarray, rgb: Sequence[int] = (255, 255, 255)) -> np.ndarray:
    """Texture unie dont l'alpha est la couverture (teintée ensuite par le sprite)."""
    out = np.empty((*cover.shape, 4), dtype=np.uint8)
    out[..., :3] = np.asarray(rgb, dtype=np.uint8)
    out[..., 3] = np.clip(np.rint(cover * 255.0), 0, 255).astype(np.uint8)
    return out


def _scaled(color: Sequence[float], k: float, alpha: int = 255) -> tuple[int, int, int, int]:
    return (
        max(0, min(255, round(color[0] * k))),
        max(0, min(255, round(color[1] * k))),
        max(0, min(255, round(color[2] * k))),
        alpha,
    )


def _rgba(color: Sequence[int], alpha: float) -> tuple[int, int, int, int]:
    return int(color[0]), int(color[1]), int(color[2]), max(0, min(255, round(255 * alpha)))


def _set_text(text: Any, value: str) -> None:
    if text.text != value:
        text.text = value


class TextureBank:
    """Textures Arcade du thème, créées une seule fois par clé."""

    def __init__(self, arc: Any) -> None:
        self.arc = arc
        self._cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str, build: Callable[[], np.ndarray]) -> Any:
        tex = self._cache.get(key)
        if tex is None:
            rgba = np.ascontiguousarray(build())
            if rgba.shape[-1] == 3:
                alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
                rgba = np.concatenate([rgba, alpha], axis=-1)
            image = Image.fromarray(rgba)
            kwargs: dict[str, Any] = {"hash": f"tron-neon:{key}"}
            algo = getattr(getattr(self.arc, "hitbox", None), "algo_bounding_box", None)
            if algo is not None:
                kwargs["hit_box_algorithm"] = algo
            tex = self.arc.Texture(image, **kwargs)
            self._cache[key] = tex
        return tex


class NeonArena:
    """Une arène néon dessinée avec Arcade, coin bas gauche en ``(left, bottom)``."""

    def __init__(
        self,
        arc: Any,
        bank: TextureBank,
        config: GameConfig,
        left: float,
        bottom: float,
        cell: int,
        theme: NeonTheme,
        quality: Quality,
        additive: Any,
        labels: bool = True,
        font: Sequence[str] = FONT,
    ) -> None:
        self.arc, self.bank = arc, bank
        self.config = config
        self.left, self.bottom = float(left), float(bottom)
        self.cell = int(cell)
        self.width, self.height = config.width, config.height
        self.theme, self.quality = theme, quality
        self.additive = additive
        self.font = font
        self.show_labels = labels and self.cell >= 14
        self.metrics = trail_metrics(self.cell, quality)
        self.tiles = trail_tiles(self.metrics)
        self.tracker = SceneTracker()
        self.effects = EffectDirector(theme, quality, seed=config.seed)
        self._key = f"{hash(theme) & 0xFFFFFFFF:08x}"
        self._build()

    # repères
    @property
    def pixel_width(self) -> int:
        return self.width * self.cell

    @property
    def pixel_height(self) -> int:
        return self.height * self.cell

    def center(self, x: int, y: int) -> tuple[float, float]:
        """Centre écran de la case (x, y) ; l'axe y d'Arcade monte, celui de la grille descend."""
        c = self.cell
        return self.left + (x + 0.5) * c, self.bottom + (self.height - y - 0.5) * c

    def to_screen(self, u: float, v: float) -> tuple[float, float]:
        """Position écran d'un point exprimé en cases (y vers le bas)."""
        return self.left + u * self.cell, self.bottom + (self.height - v) * self.cell

    # textures
    def _glow_tex(self, mask: int) -> Any:
        m = self.metrics
        return self.bank.get(
            f"glow:{m.cell}:{self.quality.name}:{mask}", lambda: glow_rgba(self.tiles.glow[mask])
        )

    def _core_tex(self, mask: int) -> Any:
        m = self.metrics
        return self.bank.get(
            f"core:{m.cell}:{self.quality.name}:{mask}", lambda: cover_rgba(self.tiles.core[mask])
        )

    def _bike_tex(self, player: int, heading: int, alive: bool) -> Any:
        theme = self.theme
        color = theme.player[player] if alive else theme.dead[player]
        light = theme.head[player] if alive else mix(theme.dead[player], (255, 255, 255), 0.2)
        key = f"bike:{self.cell}:{color}:{light}:{heading}:{int(alive)}"
        return self.bank.get(key, lambda: bike_image(color, light, self.cell, heading, alive))

    def _dot_tex(self, radius: int) -> Any:
        return self.bank.get(f"dot:{radius}", lambda: glow_rgba(soft_dot(radius)))

    # scène
    def _build(self) -> None:
        arc, c = self.arc, self.cell
        theme = self.theme
        self.background = arc.SpriteList()
        bg = self.bank.get(
            f"bg:{self.width}x{self.height}:{c}:{self.quality.name}:{self._key}",
            lambda: arena_background(self.width, self.height, c, theme, self.quality),
        )
        self.background.append(
            arc.Sprite(
                bg,
                center_x=self.left + self.pixel_width / 2,
                center_y=self.bottom + self.pixel_height / 2,
            )
        )
        self.cells = arc.SpriteList()
        for y in range(self.height):
            for x in range(self.width):
                cx, cy = self.center(x, y)
                self.cells.append(
                    arc.SpriteSolidColor(c, c, center_x=cx, center_y=cy, color=_BLACK_CLEAR)
                )
        self._floor = np.zeros((self.height, self.width, 4), dtype=np.int16)
        self._floor_key: tuple[int, int, int] | None = None
        self._edges: list[tuple[float, float]] = []
        self.obstacle_list = arc.SpriteList()
        self._obstacles: dict[Cell, Any] = {}
        self._obstacle_key: tuple[int, int] | None = None
        self.glow = arc.SpriteList()
        self.core = arc.SpriteList()
        n = self.config.n_players
        self._trail: list[list[tuple[Cell, int, Any, Any]]] = [[] for _ in range(n)]
        self._tints: list[tuple[tuple[int, ...], tuple[int, ...]] | None] = [None] * n
        self.halos = arc.SpriteList()
        self.bikes = arc.SpriteList()
        self._halo_radius = max(2, round(1.1 * c))
        for _ in range(n):
            self.halos.append(arc.Sprite(self._dot_tex(self._halo_radius)))
            self.bikes.append(arc.Sprite(self._bike_tex(0, 1, True)))
        self.particles = arc.SpriteList()
        self._shown_particles = 0
        self.labels = [
            arc.Text(
                str(p),
                0,
                0,
                color=theme.text,
                font_size=max(8, int(c * 0.42)),
                anchor_x="left",
                anchor_y="bottom",
                font_name=self.font,
                bold=True,
            )
            for p in range(n)
        ]

    def set_quality(self, quality: Quality) -> None:
        """Change le niveau de qualité : textures et sprites reconstruits (la scène est rejouée)."""
        self.quality = quality
        self.metrics = trail_metrics(self.cell, quality)
        self.tiles = trail_tiles(self.metrics)
        self.effects.set_quality(quality)
        self.tracker = SceneTracker()
        self._build()

    def sync(
        self,
        state: GameState,
        extras: Callable[[int], PlyExtras] | None = None,
        speed: float = 10.0,
        owners: np.ndarray | None = None,
    ) -> SyncResult:
        """Rattrape l'état observé : traces, sol (territoires), obstacles, effets."""
        result = self.tracker.sync(state, extras)
        self.effects.handle(result, self.tracker, state, speed)
        if result.changed:
            self._sync_trails(result)
        self._sync_floor(state, owners)
        self._sync_obstacles(state)
        return result

    def update(self, dt: float) -> None:
        self.effects.update(dt)

    # traces
    def _sync_trails(self, result: SyncResult) -> None:
        arc, tr = self.arc, self.tracker
        full = result.reset or tr.grid_mode or (bool(result.added) and bool(result.removed))
        for p in range(tr.n):
            path = tr.paths[p]
            items = self._trail[p]
            keep = min(len(items), len(path))
            k = 0 if full else max(0, keep - 2)
            while k < keep and items[k][0] == path[k]:
                mask = tr.mask(p, k)
                if items[k][1] != mask:
                    cell, _, g, cs = items[k]
                    g.texture = self._glow_tex(mask)
                    cs.texture = self._core_tex(mask)
                    items[k] = (cell, mask, g, cs)
                k += 1
            for _, _, g, cs in items[k:]:
                self.glow.remove(g)
                self.core.remove(cs)
            del items[k:]
            tint = self._tints[p]
            for i in range(k, len(path)):
                mask = tr.mask(p, i)
                cx, cy = self.center(*path[i])
                g = arc.Sprite(self._glow_tex(mask), center_x=cx, center_y=cy)
                cs = arc.Sprite(self._core_tex(mask), center_x=cx, center_y=cy)
                if tint is not None:
                    g.color, cs.color = tint
                self.glow.append(g)
                self.core.append(cs)
                items.append((path[i], mask, g, cs))

    def _update_tints(self, state: GameState) -> None:
        theme = self.theme
        for p in range(min(state.n, len(self._trail))):
            alive = state.is_alive(p)
            fade = 0.0 if alive else self.effects.fade(p)
            color = theme.trail_color(p, alive, fade)
            glow = _scaled(color, 1.0 - 0.45 * fade)
            core = _rgba(theme.core[p], 1.0 - fade)
            tint = (glow, core)
            if tint == self._tints[p]:
                continue
            self._tints[p] = tint
            for _, _, g, cs in self._trail[p]:
                g.color = glow
                cs.color = core

    # sol
    def _sync_floor(self, state: GameState, owners: np.ndarray | None) -> None:
        key = (id(owners), state.key if owners is not None else 0, state.ply)
        if key == self._floor_key:
            return
        self._floor_key = key
        h, w = self.height, self.width
        floor = np.zeros((h, w, 4), dtype=np.int16)
        self._edges = []
        if owners is not None:
            free = state.grid2d() == -1
            for p in range(state.n):
                sel = free & (owners == p)
                floor[sel] = (*self.theme.player[p], 40)
            for x0, y0, x1, y1 in territory_edges(owners):
                self._edges.append(self.to_screen(x0, y0))
                self._edges.append(self.to_screen(x1, y1))
        for y, x in np.argwhere((floor != self._floor).any(axis=2)):
            self.cells[int(y) * w + int(x)].color = tuple(int(v) for v in floor[y, x])
        self._floor = floor

    def _sync_obstacles(self, state: GameState) -> None:
        key = (id(state), state.n_obstacles)
        if key == self._obstacle_key:
            return
        self._obstacle_key = key
        g = state.grid2d()
        wanted = {(int(x), int(y)) for y, x in np.argwhere(g == OBSTACLE)}
        for cell in [c for c in self._obstacles if c not in wanted]:
            self.obstacle_list.remove(self._obstacles.pop(cell))
        if wanted - self._obstacles.keys():
            tex = self.bank.get(
                f"obstacle:{self.cell}:{self._key}",
                lambda: obstacle_tile(self.cell, self.theme.obstacle, self.theme.obstacle_edge),
            )
            for cell in sorted(wanted - self._obstacles.keys()):
                cx, cy = self.center(*cell)
                sprite = self.arc.Sprite(tex, center_x=cx, center_y=cy)
                self._obstacles[cell] = sprite
                self.obstacle_list.append(sprite)

    # dessin
    def draw(
        self,
        state: GameState,
        t: float,
        links: Sequence[tuple[int, int]] = (),
        path: Sequence[tuple[int, int, int]] = (),
    ) -> None:
        """Dessine l'arène (à appeler après :meth:`sync`)."""
        self._update_tints(state)
        self.background.draw()
        if self.quality.animated:
            self._draw_scan(t)
        self.cells.draw()
        if self._edges:
            edge = mix(self.theme.text_dim, self.theme.accent, 0.3)
            self.arc.draw_lines(self._edges, _rgba(edge, 0.45), 1.5)
        self.obstacle_list.draw()
        self._draw_zones(state)
        self.glow.draw(blend_function=self.additive)
        self.core.draw()
        self._draw_links(state, links, t)
        if path:
            self._draw_path(path)
        self._sync_bikes(state, t)
        self.halos.draw(blend_function=self.additive)
        self.bikes.draw()
        self._draw_wrecks(state)
        self._draw_particles()
        self._draw_rings()
        if self.show_labels:
            self._draw_labels(state)
        flash = self.effects.flash_alpha()
        if flash > 0:
            pad = arena_padding(self.cell)
            self.arc.draw_lrbt_rectangle_filled(
                self.left - pad,
                self.left + self.pixel_width + pad,
                self.bottom - pad,
                self.bottom + self.pixel_height + pad,
                (255, 255, 255, round(flash * FLASH_ALPHA * 0.6)),
            )

    def _draw_scan(self, t: float) -> None:
        h = self.pixel_height
        y = self.bottom + h - (t / SCAN_PERIOD % 1.0) * h
        sigma = max(4.0, 0.06 * h)
        for k in range(-3, 4):
            y0 = y + (k - 0.5) * sigma * 0.6
            y1 = y0 + sigma * 0.6
            lo, hi = max(self.bottom, y0), min(self.bottom + h, y1)
            if hi <= lo:
                continue
            a = 0.035 * math.exp(-((k * 0.6) ** 2)) * 1.4
            self.arc.draw_lrbt_rectangle_filled(
                self.left, self.left + self.pixel_width, lo, hi, _rgba(self.theme.accent, a)
            )

    def _draw_zones(self, state: GameState) -> None:
        zones = state.config.rules.zones
        if not zones:
            return
        arc = self.arc
        edge = 2 if self.cell >= 10 else 1
        for zone, team in zip(zones, zone_controllers(state), strict=True):
            color = zone_color(state, self.theme, team)
            x0, y1 = self.to_screen(zone.x, zone.y)
            x1, y0 = self.to_screen(zone.x + zone.width, zone.y + zone.height)
            arc.draw_lrbt_rectangle_filled(
                x0, x1, y0, y1, _rgba(color, 0.10 if team >= 0 else 0.04)
            )
            arc.draw_lrbt_rectangle_outline(x0, x1, y0, y1, _rgba(color, 0.8), border_width=edge)

    def _draw_links(self, state: GameState, links: Sequence[tuple[int, int]], t: float) -> None:
        arc, c = self.arc, self.cell
        drawn = [(a, b, 1.0, 1.0) for a, b in links]
        now = self.effects.time
        for lk in self.effects.active_links():
            drawn.append((lk.a, lk.b, lk.progress(now), lk.alpha(now)))
        phase = t * LINK_FLOW_SPEED if self.quality.animated else 0.0
        period = max(4.0, 0.9 * c)
        for a, b, progress, alpha in drawn:
            if not (state.is_alive(a) and state.is_alive(b)) or alpha <= 0:
                continue
            x0, y0 = self.center(*state.head(a))
            bx, by = self.center(*state.head(b))
            x1, y1 = x0 + (bx - x0) * progress, y0 + (by - y0) * progress
            color = mix(self.theme.player[a], self.theme.player[b], 0.5)
            arc.draw_line(x0, y0, x1, y1, _rgba(color, 0.16 * alpha), max(3.0, 0.45 * c))
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 1:
                continue
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            points: list[tuple[float, float]] = []
            start = ((0.25 + phase) % 1.0) * period - period
            s = start
            while s < length:
                s0, s1 = max(0.0, s), min(length, s + period / 2)
                if s1 > s0:
                    points.append((x0 + ux * s0, y0 + uy * s0))
                    points.append((x0 + ux * s1, y0 + uy * s1))
                s += period
            if points:
                arc.draw_lines(points, _rgba(color, 0.9 * alpha), max(1.5, 0.12 * c))

    def _draw_path(self, path: Sequence[tuple[int, int, int]]) -> None:
        r = max(1.5, 0.22 * self.cell)
        for p, x, y in path:
            cx, cy = self.center(x, y)
            self.arc.draw_polygon_filled(
                [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
                _rgba(self.theme.head[p], 0.86),
            )

    def _sync_bikes(self, state: GameState, t: float) -> None:
        k = pulse(t, self.quality.animated)
        tr = self.tracker
        for p in range(min(state.n, len(self.bikes))):
            cx, cy = self.center(*state.head(p))
            alive = state.is_alive(p)
            heading = tr.headings[p] if p < tr.n else -1
            if heading < 0:
                heading = default_heading(state, p)
            bike = self.bikes[p]
            tex = self._bike_tex(p, heading, alive)
            if bike.texture is not tex:
                bike.texture = tex
            bike.center_x, bike.center_y = cx, cy
            halo = self.halos[p]
            halo.center_x, halo.center_y = cx, cy
            if alive:
                halo.scale = k
                halo.color = _scaled(self.theme.player[p], 0.30 + 0.25 * k)
            else:
                halo.color = (0, 0, 0, 255)

    def _draw_wrecks(self, state: GameState) -> None:
        half = 0.32 * self.cell
        width = max(1.0, 0.09 * self.cell)
        color = _rgba(self.theme.danger, 0.85)
        for p in range(state.n):
            if state.is_alive(p) or self.cell < 6:
                continue
            cx, cy = self.center(*state.head(p))
            self.arc.draw_line(cx - half, cy - half, cx + half, cy + half, color, width)
            self.arc.draw_line(cx - half, cy + half, cx + half, cy - half, color, width)

    def _draw_particles(self) -> None:
        view = self.effects.particles.view()
        n = len(view)
        sprites = self.particles
        if n > len(sprites):
            tex = self._dot_tex(PARTICLE_RADIUS)
            for _ in range(n - len(sprites)):
                sprites.append(self.arc.Sprite(tex))
        c = self.cell
        for i in range(n):
            sprite = sprites[i]
            sprite.center_x, sprite.center_y = self.to_screen(float(view.x[i]), float(view.y[i]))
            radius = max(1.0, float(view.size[i]) * c * 1.6)
            sprite.scale = radius / PARTICLE_RADIUS
            sprite.color = _scaled(view.color[i], float(view.alpha[i]) * 0.9)
        for i in range(n, self._shown_particles):
            sprites[i].color = (0, 0, 0, 0)
        self._shown_particles = n
        if n:
            sprites.draw(blend_function=self.additive)

    def _draw_rings(self) -> None:
        now = self.effects.time
        width = max(1.0, 0.12 * self.cell)
        for ring in self.effects.active_rings():
            cx, cy = self.to_screen(ring.x, ring.y)
            self.arc.draw_circle_outline(
                cx,
                cy,
                max(1.0, ring.radius(now) * self.cell),
                _rgba(ring.color, 0.9 * ring.alpha(now)),
                border_width=width,
            )

    def _draw_labels(self, state: GameState) -> None:
        c = self.cell
        for p, label in enumerate(self.labels[: state.n]):
            x, y = state.head(p)
            label.x = self.left + (x + 1) * c - 2
            label.y = self.bottom + (self.height - y) * c - 0.1 * c
            label.color = self.theme.text if state.is_alive(p) else self.theme.text_muted
            label.draw()

    def shake_pixels(self) -> tuple[float, float]:
        """Décalage de caméra du tremblement en cours, en pixels (borné)."""
        dx, dy = self.effects.shake_offset()
        lim = SHAKE_MAX_PIXELS
        return (
            max(-lim, min(lim, dx * self.cell)),
            max(-lim, min(lim, dy * self.cell)),
        )


class NeonHud:
    """Panneau holographique à droite de la fenêtre.

    Titre et état, une carte par joueur (badge ACTIF / ÉLIMINÉ / rang, barre de
    territoire), statistiques, journal horodaté des événements, aide. Les objets
    ``arcade.Text`` sont créés une fois ; seul leur contenu change.
    """

    def __init__(
        self,
        arc: Any,
        theme: NeonTheme,
        left: float,
        right: float,
        height: float,
        n_players: int,
        help_lines: Sequence[str],
        font: Sequence[str] = FONT,
    ) -> None:
        self.arc, self.theme = arc, theme
        self.left, self.right, self.height = float(left), float(right), float(height)
        self.n = n_players
        self.font = font
        ink, dim = theme.text, theme.text_dim
        top = self.height - 14
        self.t_title = arc.Text(
            "TRON // COALITIONS", left, top - 16, color=theme.accent, font_size=13,
            font_name=font, bold=True,
        )  # fmt: skip
        self.t_game = arc.Text(
            "", left, top - 34, color=dim, font_size=9, font_name=font, width=int(right - left - 8)
        )
        self.t_status = arc.Text("", left, top - 52, color=ink, font_size=10, font_name=font)
        self.t_ply = arc.Text("", left, top - 68, color=dim, font_size=9, font_name=font)
        self.header_bottom = top - 80
        help_h = 16 * len(help_lines) + 10
        stats_h = 22 + 14 * STATS_ROWS
        log_min = 22 + 14 * 3
        gaps = 3 * 10
        area = self.header_bottom - help_h - stats_h - log_min - gaps
        self.card_h = int(max(30, min(50, area / max(1, n_players))))
        y = self.header_bottom - 4
        self.cards_top = y
        self.t_cards = []
        for p in range(n_players):
            top_p = y - p * self.card_h
            self.t_cards.append(
                (
                    arc.Text("", left + 10, top_p - 13, color=ink, font_size=10,
                             font_name=font, bold=True),
                    arc.Text("", right - 12, top_p - 13, color=theme.success, font_size=8,
                             font_name=font, bold=True, anchor_x="right"),
                    arc.Text("", left + 10, top_p - 25, color=dim, font_size=8, font_name=font),
                )
            )  # fmt: skip
        self.cards_bottom = y - n_players * self.card_h
        inner = self.right - self.left - 24
        self.title_chars = max(8, int((inner - 70) / 7.2))
        self.detail_chars = max(10, int(inner / 5.4))
        self.log_chars = max(10, int(inner / 5.6))
        self.stats_top = self.cards_bottom - 10
        self.t_stats_title = arc.Text(
            "STATISTIQUES", left, self.stats_top - 14, color=theme.accent, font_size=9,
            font_name=font, bold=True,
        )  # fmt: skip
        self.t_stats = [
            (
                arc.Text("", left + 4, self.stats_top - 32 - 14 * i, color=dim, font_size=9,
                         font_name=font),
                arc.Text("", right - 12, self.stats_top - 32 - 14 * i, color=ink, font_size=9,
                         font_name=font, anchor_x="right"),
            )
            for i in range(STATS_ROWS)
        ]  # fmt: skip
        self.log_top = self.stats_top - stats_h - 10
        self.t_log_title = arc.Text(
            "ÉVÉNEMENTS", left, self.log_top - 14, color=theme.accent, font_size=9,
            font_name=font, bold=True,
        )  # fmt: skip
        n_log = int(max(0, min(8, (self.log_top - 22 - help_h - 10) // 14)))
        self.t_log = [
            arc.Text("", left + 4, self.log_top - 32 - 14 * i, color=dim, font_size=8,
                     font_name=font)
            for i in range(n_log)
        ]  # fmt: skip
        self.t_help = [
            arc.Text(line, left, 14 + 16 * (len(help_lines) - 1 - i), color=theme.text_muted,
                     font_size=8, font_name=font)
            for i, line in enumerate(help_lines)
        ]  # fmt: skip

    def _frame(self, x0: float, x1: float, y0: float, y1: float, color: Sequence[int]) -> None:
        """Cadre holographique : liseré fin et équerres aux coins."""
        arc = self.arc
        arc.draw_lrbt_rectangle_outline(x0, x1, y0, y1, _rgba(color, 0.35), border_width=1)
        k = min(10.0, (x1 - x0) / 4, (y1 - y0) / 4)
        bright = _rgba(color, 0.9)
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            arc.draw_line(cx, cy, cx + sx * k, cy, bright, 2)
            arc.draw_line(cx, cy, cx, cy + sy * k, bright, 2)

    def draw(
        self,
        title: str,
        status: str,
        ply_line: str,
        cards: Sequence[PlayerCard],
        stats: Sequence[tuple[str, str]],
        log: Sequence[LogLine],
        t: float,
    ) -> None:
        arc, theme = self.arc, self.theme
        x0 = self.left - 10
        arc.draw_lrbt_rectangle_filled(x0, self.right, 0, self.height, theme.panel)
        arc.draw_line(x0, 0, x0, self.height, theme.panel_edge, 2)
        self._frame(
            self.left - 4, self.right - 6, self.header_bottom, self.height - 8, theme.accent
        )
        _set_text(self.t_game, title)
        _set_text(self.t_status, status)
        _set_text(self.t_ply, ply_line)
        for text in (self.t_title, self.t_game, self.t_status, self.t_ply):
            text.draw()
        blink = 0.55 + 0.45 * math.sin(6.0 * t)
        for card, (t_name, t_badge, t_detail) in zip(cards, self.t_cards, strict=False):
            top = self.cards_top - card.index * self.card_h
            y0, y1 = top - self.card_h + 3, top
            arc.draw_lrbt_rectangle_filled(
                self.left - 4, self.right - 6, y0, y1, _rgba(card.color, 0.07)
            )
            arc.draw_lrbt_rectangle_filled(self.left - 4, self.left, y0, y1, card.color)
            if card.current:
                arc.draw_lrbt_rectangle_outline(
                    self.left - 4, self.right - 6, y0, y1, _rgba(card.color, 0.4 + 0.4 * blink),
                    border_width=1,
                )  # fmt: skip
            bar = self.right - 16 - (self.left + 10)
            arc.draw_lrbt_rectangle_filled(
                self.left + 10, self.left + 10 + bar, y0 + 1, y0 + 3, _rgba(theme.text_muted, 0.35)
            )
            if card.share > 0:
                arc.draw_lrbt_rectangle_filled(
                    self.left + 10, self.left + 10 + bar * min(1.0, card.share), y0 + 1, y0 + 3,
                    card.color,
                )  # fmt: skip
            _set_text(t_name, fit(card.title, self.title_chars))
            _set_text(t_badge, card.badge)
            _set_text(t_detail, fit(card.detail, self.detail_chars))
            t_badge.color = card.badge_color
            t_name.color = theme.text if card.alive else theme.text_muted
            t_name.draw()
            t_badge.draw()
            if self.card_h >= 34:
                t_detail.draw()
        self._frame(self.left - 4, self.right - 6, self.log_top + 4, self.stats_top, theme.accent)
        self.t_stats_title.draw()
        for (label, value), (t_label, t_value) in zip(stats, self.t_stats, strict=False):
            _set_text(t_label, label)
            _set_text(t_value, value)
            t_label.draw()
            t_value.draw()
        self.t_log_title.draw()
        shown = list(log)[-len(self.t_log) :] if self.t_log else []
        for line, text in zip(shown, self.t_log, strict=False):
            _set_text(text, fit(stamp(line), self.log_chars))
            text.color = line.color
            text.draw()
        for text in self.t_help:
            text.draw()


def victory_banner(arc: Any, arena: NeonArena, text: Any, label: str, color: RGB, t: float) -> None:
    """Bandeau de fin de partie au centre de l'arène."""
    cx = arena.left + arena.pixel_width / 2
    cy = arena.bottom + arena.pixel_height / 2
    half = min(arena.pixel_width / 2 - 6, 220)
    glow = 0.75 + 0.25 * math.sin(3.0 * t)
    arc.draw_lrbt_rectangle_filled(cx - half, cx + half, cy - 22, cy + 22, (3, 7, 18, 200))
    arc.draw_line(cx - half, cy - 22, cx + half, cy - 22, _rgba(color, glow), 2)
    arc.draw_line(cx - half, cy + 22, cx + half, cy + 22, _rgba(color, glow), 2)
    _set_text(text, label)
    text.x, text.y = cx, cy
    text.color = color
    text.draw()


__all__ = [
    "FONT",
    "NeonArena",
    "NeonHud",
    "TextureBank",
    "cover_rgba",
    "glow_rgba",
    "victory_banner",
]
