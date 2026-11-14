"""Bouchon de l'API Arcade 3.3 pour tester les fenêtres sans écran ni OpenGL.

Il enregistre les appels de dessin (``module.calls``) et imite juste assez
d'objets (sprites, listes de sprites, textures, textes, caméra, contexte GL)
pour exécuter la logique des fenêtres classique et néon.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


class Stub:
    """Objet générique : garde ses arguments, ``draw`` note le texte dessiné."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.x = args[1] if len(args) > 1 else 0
        self.y = args[2] if len(args) > 2 else 0
        self.text = args[0] if args else ""
        self.drawn: list[str] = []

    def draw(self) -> None:
        self.drawn.append(self.text)


class SpriteList(list):  # type: ignore[type-arg]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.draws: list[dict[str, Any]] = []

    def draw(self, **kwargs: Any) -> None:
        self.draws.append(kwargs)


class Texture:
    def __init__(self, image: Any, **kwargs: Any) -> None:
        self.image = image
        self.kwargs = kwargs
        self.size = image.size


class Sprite:
    def __init__(self, texture: Any = None, scale: float = 1.0, **kwargs: Any) -> None:
        self.texture = texture
        self.scale = scale
        self.center_x = kwargs.get("center_x", 0.0)
        self.center_y = kwargs.get("center_y", 0.0)
        self.color: tuple[int, ...] = (255, 255, 255, 255)


class SolidColor(Stub):
    def __init__(self, width: int, height: int, **kwargs: Any) -> None:
        super().__init__(width, height, **kwargs)
        self.color = kwargs.get("color", (255, 255, 255, 255))


class Camera2D:
    def __init__(self) -> None:
        self.position = (0.0, 0.0)
        self.used = 0

    def use(self) -> None:
        self.used += 1


class _Target:
    def __init__(self, log: list[str], name: str) -> None:
        self.log, self.name = log, name

    def use(self, *args: Any) -> None:
        self.log.append(f"use:{self.name}")

    def clear(self, **kwargs: Any) -> None:
        self.log.append(f"clear:{self.name}")


class Program(dict):  # type: ignore[type-arg]
    pass


class Context:
    BLEND_ADDITIVE = (1, 1)
    BLEND_DEFAULT = (770, 771)
    LINEAR = 9729
    CLAMP_TO_EDGE = 33071

    def __init__(self, fail: bool = False) -> None:
        self.log: list[str] = []
        self.fail = fail
        self.screen = _Target(self.log, "screen")
        self.programs: list[Program] = []

    def texture(self, size: tuple[int, int], **kwargs: Any) -> _Target:
        if self.fail:
            raise RuntimeError("pas de contexte OpenGL")
        return _Target(self.log, f"tex{size}")

    def framebuffer(self, color_attachments: list[Any]) -> _Target:
        return _Target(self.log, "fbo")

    def program(self, vertex_shader: str, fragment_shader: str) -> Program:
        prog = Program(vertex=vertex_shader, fragment=fragment_shader)
        self.programs.append(prog)
        return prog


class Image:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"png")


def make_window_class(calls: list[str]) -> type:
    class Window:
        def __init__(self, width: int, height: int, title: str, update_rate: float = 1 / 60):
            self.width, self.height, self.title = width, height, title
            self.closed = False
            self.ctx = Context()
            self.default_camera = Camera2D()

        def clear(self) -> None:
            calls.append("clear")

        def close(self) -> None:
            self.closed = True

        def get_framebuffer_size(self) -> tuple[int, int]:
            return self.width, self.height

    return Window


def make_arcade() -> types.ModuleType:
    """Module ``arcade`` factice ; ``module.calls`` liste les appels de dessin."""
    mod = types.ModuleType("arcade")
    calls: list[str] = []
    mod.calls = calls  # type: ignore[attr-defined]
    mod.Window = make_window_class(calls)  # type: ignore[attr-defined]
    mod.SpriteList = SpriteList  # type: ignore[attr-defined]
    mod.SpriteSolidColor = SolidColor  # type: ignore[attr-defined]
    mod.Sprite = Sprite  # type: ignore[attr-defined]
    mod.Texture = Texture  # type: ignore[attr-defined]
    mod.Text = Stub  # type: ignore[attr-defined]
    mod.Camera2D = Camera2D  # type: ignore[attr-defined]
    mod.hitbox = types.SimpleNamespace(algo_bounding_box=object())  # type: ignore[attr-defined]
    for name in (
        "draw_lrbt_rectangle_filled",
        "draw_lrbt_rectangle_outline",
        "draw_line",
        "draw_lines",
        "draw_circle_outline",
        "draw_polygon_filled",
    ):
        short = (
            name.removeprefix("draw_")
            .replace("lrbt_rectangle_filled", "rect")
            .replace("lrbt_rectangle_outline", "outline")
        )
        setattr(mod, name, lambda *a, _n=short, **k: calls.append(_n))
    mod.get_image = lambda *a, **k: Image()  # type: ignore[attr-defined]
    mod.run = lambda: calls.append("run")  # type: ignore[attr-defined]
    mod.key = types.SimpleNamespace(  # type: ignore[attr-defined]
        SPACE=1, RIGHT=2, LEFT=3, UP=4, DOWN=5, V=6, N=7, HOME=8, END=9, S=10, ESCAPE=11, Q=12,
        D=13, T=14, G=15, B=16, A=17,
    )  # fmt: skip
    gl = types.ModuleType("arcade.gl")
    geometry = types.ModuleType("arcade.gl.geometry")

    class Quad:
        def __init__(self) -> None:
            self.renders: list[Any] = []

        def render(self, program: Any) -> None:
            self.renders.append(program)

    geometry.quad_2d_fs = Quad  # type: ignore[attr-defined]
    gl.geometry = geometry  # type: ignore[attr-defined]
    mod.gl = gl  # type: ignore[attr-defined]
    return mod


def install(monkeypatch: Any) -> types.ModuleType:
    """Installe le bouchon dans ``sys.modules`` (``arcade``, ``arcade.gl``, sa géométrie)."""
    mod = make_arcade()
    monkeypatch.setitem(sys.modules, "arcade", mod)
    monkeypatch.setitem(sys.modules, "arcade.gl", mod.gl)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arcade.gl.geometry", mod.gl.geometry)  # type: ignore[attr-defined]
    return mod
