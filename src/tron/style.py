"""Constantes de couleur partagées par l'affichage (``viz``) et les figures (``analysis``).

Palette catégorielle validée pour les daltoniens (ordre fixe, paires adjacentes
séparées d'au moins 8 en ΔE OKLab ×100 sous simulation CVD), rampe séquentielle
d'une seule teinte (bleu, du clair au foncé), surfaces et encres claires et sombres.
"""

CATEGORICAL_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
CATEGORICAL_LIGHT = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

SURFACE_DARK = "#1a1a19"
GRID_DARK = "#2c2c2a"
BASELINE_DARK = "#383835"
INK_PRIMARY_DARK = "#f0efec"
INK_SECONDARY_DARK = "#c3c2b7"
INK_MUTED = "#898781"

SURFACE_LIGHT = "#fcfcfb"
GRID_LIGHT = "#e1e0d9"
BASELINE_LIGHT = "#c3c2b7"
INK_PRIMARY_LIGHT = "#0b0b0b"
INK_SECONDARY_LIGHT = "#52514e"
