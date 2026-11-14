"""Heuristiques d'évaluation et valeur des feuilles."""

from tron.eval.base import (
    HEURISTICS,
    Heuristic,
    UniformHeuristic,
    make_heuristic,
    register_heuristic,
)
from tron.eval.composite import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    CompositeFeatures,
    CompositeHeuristic,
    normalize_weights,
)
from tron.eval.valuation import PLY_CAP, WIN_BONUS, Valuation
from tron.eval.voronoi import TIE_MODES, VoronoiHeuristic, voronoi_kernel, warmup
from tron.eval.zone import ZONE_COMPONENTS, ZoneHeuristic

__all__ = [
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "HEURISTICS",
    "PLY_CAP",
    "TIE_MODES",
    "WIN_BONUS",
    "ZONE_COMPONENTS",
    "CompositeFeatures",
    "CompositeHeuristic",
    "Heuristic",
    "UniformHeuristic",
    "Valuation",
    "VoronoiHeuristic",
    "ZoneHeuristic",
    "make_heuristic",
    "normalize_weights",
    "register_heuristic",
    "voronoi_kernel",
    "warmup",
]

# Extensions installées : chargées une fois, après toutes les heuristiques intégrées.
from tron.plugins import HEURISTIC_GROUP as _HEURISTIC_GROUP
from tron.plugins import load_plugins as _load_plugins

_load_plugins(_HEURISTIC_GROUP)
