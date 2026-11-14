"""« Arcade logiciel » : exécute les appels de dessin des fenêtres et les rend avec NumPy et PIL.

Outil de test et de documentation : il réutilise le bouchon de
:mod:`arcade_stub` mais dessine réellement rectangles, lignes, cercles,
polygones, textes et sprites (mélanges alpha et additif) dans une image. On
obtient ainsi un **aperçu simulé** de la mise en page d'une fenêtre sans écran
ni OpenGL. Ce n'est pas le rendu d'Arcade (polices, anticrénelage, bloom) : la
validation sur un vrai poste graphique reste nécessaire (``docs/validation-gui.md``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import arcade_stub

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
_BOLD_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: float, bold: bool) -> Any:
    px = max(6, round(size * 96 / 72))
    for path in _BOLD_PATHS if bold else _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    return ImageFont.load_default(size=px)


class Canvas:
    """Image RGB flottante, origine en bas à gauche comme Arcade."""

    def __init__(self, width: int, height: int, background: Any = (0, 0, 0)) -> None:
        self.w, self.h = int(width), int(height)
        self.img = np.zeros((self.h, self.w, 3), dtype=np.float32)
        self.img[:] = np.asarray(background[:3], dtype=np.float32)

    def clear(self, color: Any) -> None:
        self.img[:] = np.asarray(color[:3], dtype=np.float32)

    def _blend(self, layer: Image.Image, x0: int, y0: int, additive: bool = False) -> None:
        arr = np.asarray(layer, dtype=np.float32)
        h, w = arr.shape[:2]
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(self.w, x0 + w), min(self.h, y0 + h)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        sub = arr[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
        dst = self.img[ay0:ay1, ax0:ax1]
        if additive:
            dst += sub[..., :3]
        else:
            a = sub[..., 3:4] / 255.0
            dst[:] = sub[..., :3] * a + dst * (1 - a)

    def _layer(self, box: tuple[float, float, float, float]) -> tuple[Image.Image, int, int, Any]:
        """Calque RGBA couvrant ``box`` (x0, y0, x1, y1 en coordonnées image, y vers le bas)."""
        x0, y0 = int(np.floor(box[0])) - 2, int(np.floor(box[1])) - 2
        x1, y1 = int(np.ceil(box[2])) + 2, int(np.ceil(box[3])) + 2
        layer = Image.new("RGBA", (max(1, x1 - x0), max(1, y1 - y0)), (0, 0, 0, 0))
        return layer, x0, y0, ImageDraw.Draw(layer)

    def flip(self, y: float) -> float:
        return self.h - y

    # -------------------------------------------------------------- formes
    def rect(
        self, left: float, r: float, b: float, t: float, color: Any, outline: float = 0
    ) -> None:
        layer, x0, y0, draw = self._layer((left, self.flip(t), r, self.flip(b)))
        top, low = self.flip(t) - y0, self.flip(b) - y0
        shape = [left - x0, top, max(left - x0, r - x0 - 1), max(top, low - 1)]
        rgba = _rgba(color)
        if outline:
            draw.rectangle(shape, outline=rgba, width=max(1, round(outline)))
        else:
            draw.rectangle(shape, fill=rgba)
        self._blend(layer, x0, y0)

    def lines(self, points: list[tuple[float, float]], color: Any, width: float) -> None:
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [self.flip(p[1]) for p in points]
        pad = width + 2
        layer, x0, y0, draw = self._layer(
            (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        )
        for i in range(0, len(points) - 1, 2):
            draw.line(
                [(xs[i] - x0, ys[i] - y0), (xs[i + 1] - x0, ys[i + 1] - y0)],
                fill=_rgba(color),
                width=max(1, round(width)),
            )
        self._blend(layer, x0, y0)

    def circle(self, cx: float, cy: float, r: float, color: Any, width: float) -> None:
        y = self.flip(cy)
        layer, x0, y0, draw = self._layer(
            (cx - r - width, y - r - width, cx + r + width, y + r + width)
        )
        draw.ellipse(
            [cx - r - x0, y - r - y0, cx + r - x0, y + r - y0],
            outline=_rgba(color),
            width=max(1, round(width)),
        )
        self._blend(layer, x0, y0)

    def polygon(self, points: list[tuple[float, float]], color: Any) -> None:
        xs = [p[0] for p in points]
        ys = [self.flip(p[1]) for p in points]
        layer, x0, y0, draw = self._layer((min(xs), min(ys), max(xs), max(ys)))
        draw.polygon([(x - x0, y - y0) for x, y in zip(xs, ys, strict=True)], fill=_rgba(color))
        self._blend(layer, x0, y0)

    def text(self, item: Any) -> None:
        value = str(item.text)
        if not value:
            return
        font = _font(getattr(item, "font_size", 12), bool(getattr(item, "bold", False)))
        left, _, right, _ = font.getbbox(value)
        w = right - left
        ascent, descent = font.getmetrics()
        ax = getattr(item, "anchor_x", "left")
        ay = getattr(item, "anchor_y", "baseline")
        x = float(item.x) - (w / 2 if ax == "center" else w if ax == "right" else 0)
        base = self.flip(float(item.y))
        if ay == "center":
            top_y = base - ascent / 2 - 1
        elif ay == "bottom":
            top_y = base - ascent - descent
        elif ay == "top":
            top_y = base
        else:
            top_y = base - ascent
        layer, x0, y0, draw = self._layer((x, top_y, x + w + 4, top_y + ascent + descent + 4))
        draw.text(
            (x - x0, top_y - y0), value, font=font, fill=_rgba(getattr(item, "color", (255,) * 3))
        )
        self._blend(layer, x0, y0)

    def sprite(self, sprite: Any, additive: bool) -> None:
        tex = getattr(sprite, "texture", None)
        color = tuple(sprite.color) + ((255,) if len(sprite.color) == 3 else ())
        if tex is not None:
            image = tex.image.convert("RGBA")
            scale = sprite.scale if isinstance(sprite.scale, (int, float)) else sprite.scale[0]
            if abs(scale - 1.0) > 1e-3:
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.BILINEAR)
            arr = np.asarray(image, dtype=np.float32)
            arr = arr * (np.asarray(color, dtype=np.float32) / 255.0)
            if additive:
                arr[..., 3] = 255.0
            layer = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            x0 = round(sprite.center_x - layer.width / 2)
            y0 = round(self.flip(sprite.center_y) - layer.height / 2)
            self._blend(layer, x0, y0, additive)
        else:  # case unie (SpriteSolidColor)
            w, h = sprite.args[0], sprite.args[1]
            cx, cy = sprite.center_x, sprite.center_y
            if color[3] > 0:
                self.rect(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2, color)

    def image(self) -> np.ndarray:
        return np.clip(np.rint(self.img), 0, 255).astype(np.uint8)


def _rgba(color: Any) -> tuple[int, int, int, int]:
    c = tuple(int(v) for v in color)
    return c if len(c) == 4 else (*c, 255)  # type: ignore[return-value]


def make_arcade() -> types.ModuleType:
    """Bouchon d'Arcade qui dessine vraiment (``module.canvas`` : image de la dernière fenêtre)."""
    mod = arcade_stub.make_arcade()
    state: dict[str, Canvas] = {}
    base_window = mod.Window

    class Window(base_window):  # type: ignore[misc, valid-type]
        def __init__(self, width: int, height: int, title: str, update_rate: float = 1 / 60):
            super().__init__(width, height, title, update_rate)
            self.background_color = (0, 0, 0)
            state["canvas"] = Canvas(width, height)

        def clear(self) -> None:
            state["canvas"].clear(getattr(self, "background_color", (0, 0, 0)))

    def canvas() -> Canvas:
        return state["canvas"]

    class SpriteList(arcade_stub.SpriteList):
        def draw(self, **kwargs: Any) -> None:
            super().draw(**kwargs)
            additive = kwargs.get("blend_function") == arcade_stub.Context.BLEND_ADDITIVE
            for sprite in self:
                canvas().sprite(sprite, additive)

    class Text(arcade_stub.Stub):
        def draw(self) -> None:
            super().draw()
            canvas().text(self)

    def draw_lines(points: list[tuple[float, float]], color: Any, line_width: float = 1) -> None:
        canvas().lines(list(points), color, line_width)

    mod.Window = Window  # type: ignore[attr-defined]
    mod.SpriteList = SpriteList  # type: ignore[attr-defined]
    mod.Text = Text  # type: ignore[attr-defined]

    def rect_filled(
        left: float, r: float, b: float, t: float, color: Any, *a: Any, **k: Any
    ) -> None:
        canvas().rect(left, r, b, t, color)

    def rect_outline(
        left: float, r: float, b: float, t: float, color: Any, border_width: float = 1, **k: Any
    ) -> None:
        canvas().rect(left, r, b, t, color, outline=border_width)

    mod.draw_lrbt_rectangle_filled = rect_filled  # type: ignore[attr-defined]
    mod.draw_lrbt_rectangle_outline = rect_outline  # type: ignore[attr-defined]
    mod.draw_line = lambda x0, y0, x1, y1, color, line_width=1, **k: canvas().lines(  # type: ignore[attr-defined]
        [(x0, y0), (x1, y1)], color, line_width
    )
    mod.draw_lines = draw_lines  # type: ignore[attr-defined]
    mod.draw_circle_outline = lambda cx, cy, r, color, border_width=1, **k: canvas().circle(  # type: ignore[attr-defined]
        cx, cy, r, color, border_width
    )
    mod.draw_polygon_filled = lambda points, color: canvas().polygon(list(points), color)  # type: ignore[attr-defined]
    mod.canvas = canvas  # type: ignore[attr-defined]
    return mod


def install(monkeypatch: Any | None = None) -> types.ModuleType:
    """Installe l'Arcade logiciel dans ``sys.modules``."""
    mod = make_arcade()
    entries = {
        "arcade": mod,
        "arcade.gl": mod.gl,  # type: ignore[attr-defined]
        "arcade.gl.geometry": mod.gl.geometry,  # type: ignore[attr-defined]
    }
    for name, value in entries.items():
        if monkeypatch is not None:
            monkeypatch.setitem(sys.modules, name, value)
        else:
            sys.modules[name] = value
    return mod
