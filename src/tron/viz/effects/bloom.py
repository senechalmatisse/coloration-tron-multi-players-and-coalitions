"""Post-traitement *bloom* (qualité ``ultra``) avec ``arcade.gl``, désactivé à la moindre erreur.

Étapes, après le dessin de l'arène dans une texture hors écran :

1. extraction des pixels lumineux (seuil progressif) dans une texture à demi-résolution ;
2. flou gaussien séparable, horizontal puis vertical (``passes`` allers-retours) ;
3. composition additive de la scène et du halo flouté à l'écran.

Le HUD est dessiné ensuite, directement à l'écran : il reste net. Les sources
GLSL sont dans ``tron/viz/shaders``. Toute exception (contexte OpenGL trop
ancien, pilote, API Arcade) désactive le bloom pour la suite et le rendu
reprend sans lui : :meth:`Bloom.begin` renvoie alors ``False``.
"""

from __future__ import annotations

import contextlib
import logging
from importlib import resources
from typing import Any

LOG = logging.getLogger(__name__)


def shader_source(name: str) -> str:
    """Source GLSL d'un fichier de ``tron/viz/shaders``."""
    return resources.files("tron.viz.shaders").joinpath(name).read_text(encoding="utf-8")


class Bloom:
    """Bloom d'une fenêtre Arcade (``window.ctx``)."""

    def __init__(
        self,
        window: Any,
        threshold: float = 0.55,
        intensity: float = 1.15,
        passes: int = 2,
        downscale: int = 2,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.intensity = intensity
        self.passes = max(1, passes)
        self.downscale = max(1, downscale)
        self.enabled = True
        self.error: str | None = None
        self._ready = False
        self._active = False

    # ressources
    def _setup(self) -> None:
        from arcade.gl import geometry  # noqa: PLC0415 - Arcade seulement à la demande

        ctx = self.window.ctx
        size = self.window.get_framebuffer_size()
        w, h = max(1, int(size[0])), max(1, int(size[1]))
        sw, sh = max(1, w // self.downscale), max(1, h // self.downscale)
        linear = (ctx.LINEAR, ctx.LINEAR)
        clamp = ctx.CLAMP_TO_EDGE
        self.size = (w, h)
        self.scene_tex = ctx.texture(
            (w, h), components=4, filter=linear, wrap_x=clamp, wrap_y=clamp
        )
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex])
        self.ping_tex = ctx.texture(
            (sw, sh), components=4, filter=linear, wrap_x=clamp, wrap_y=clamp
        )
        self.ping_fbo = ctx.framebuffer(color_attachments=[self.ping_tex])
        self.pong_tex = ctx.texture(
            (sw, sh), components=4, filter=linear, wrap_x=clamp, wrap_y=clamp
        )
        self.pong_fbo = ctx.framebuffer(color_attachments=[self.pong_tex])
        vertex = shader_source("fullscreen.vert")
        self.extract = ctx.program(
            vertex_shader=vertex, fragment_shader=shader_source("bloom_extract.frag")
        )
        self.blur = ctx.program(vertex_shader=vertex, fragment_shader=shader_source("blur.frag"))
        self.combine = ctx.program(
            vertex_shader=vertex, fragment_shader=shader_source("bloom_combine.frag")
        )
        self.extract["source"] = 0
        self.blur["source"] = 0
        self.combine["scene"] = 0
        self.combine["bloom"] = 1
        self.quad = geometry.quad_2d_fs()
        self.texel = (1.0 / sw, 1.0 / sh)
        self._ready = True

    def _fail(self, exc: BaseException) -> None:
        self.enabled = False
        self._active = False
        self.error = f"{type(exc).__name__}: {exc}"
        LOG.warning("bloom désactivé : %s", self.error)
        with contextlib.suppress(Exception):  # contexte peut-être inutilisable
            self.window.ctx.screen.use()

    # image
    def begin(self, clear_color: tuple[int, int, int]) -> bool:
        """Redirige le dessin vers la texture de scène ; ``False`` si le bloom est indisponible."""
        if not self.enabled:
            return False
        try:
            if not self._ready:
                self._setup()
            self.scene_fbo.use()
            self.scene_fbo.clear(color_normalized=(*(c / 255 for c in clear_color), 1.0))
            self._active = True
            return True
        except Exception as exc:
            self._fail(exc)
            return False

    def end(self) -> bool:
        """Calcule le halo et compose la scène à l'écran ; ``False`` en cas d'échec."""
        if not self._active:
            return False
        self._active = False
        try:
            ctx = self.window.ctx
            self.ping_fbo.use()
            self.ping_fbo.clear()
            self.scene_tex.use(0)
            self.extract["threshold"] = self.threshold
            self.quad.render(self.extract)
            for _ in range(self.passes):
                self.pong_fbo.use()
                self.ping_tex.use(0)
                self.blur["direction"] = (self.texel[0], 0.0)
                self.quad.render(self.blur)
                self.ping_fbo.use()
                self.pong_tex.use(0)
                self.blur["direction"] = (0.0, self.texel[1])
                self.quad.render(self.blur)
            ctx.screen.use()
            self.scene_tex.use(0)
            self.ping_tex.use(1)
            self.combine["intensity"] = self.intensity
            self.quad.render(self.combine)
            return True
        except Exception as exc:
            self._fail(exc)
            return False


__all__ = ["Bloom", "shader_source"]
