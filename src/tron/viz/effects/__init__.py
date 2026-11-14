"""Effets visuels du thème néon : particules, ondes de choc, liens, éclair, tremblement, bloom.

La logique (``particles``, ``director``) est indépendante de toute bibliothèque
graphique ; seul ``bloom`` s'appuie sur ``arcade.gl`` et n'est importé qu'à la demande par la fenêtre.
"""

from tron.viz.effects.director import EffectDirector, LinkFlash, Ring, zone_color, zone_controllers
from tron.viz.effects.particles import ParticleSystem, ParticleView

__all__ = [
    "EffectDirector",
    "LinkFlash",
    "ParticleSystem",
    "ParticleView",
    "Ring",
    "zone_color",
    "zone_controllers",
]
