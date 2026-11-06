"""Visualisation : palette, rendu NumPy, contrôle de lecture, exports, fenêtre arcade.

La fenêtre arcade (``tron.viz.arcade_app``) est importée à la demande : le reste
du paquet, et a fortiori le moteur, fonctionne sans affichage (F6.5).
"""

from tron.viz.controller import LiveSource, PanelInfo, PlaybackController, ReplaySource
from tron.viz.export import export_animation, log_frames, save_png
from tron.viz.palette import Palette, make_palette
from tron.viz.render import color_grid, render_frame

__all__ = [
    "LiveSource",
    "Palette",
    "PanelInfo",
    "PlaybackController",
    "ReplaySource",
    "color_grid",
    "export_animation",
    "log_frames",
    "make_palette",
    "render_frame",
    "save_png",
]
