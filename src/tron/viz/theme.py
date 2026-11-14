"""Thème néon « Neon Arena » : couleurs, tailles relatives, animations, niveaux de qualité.

Tout ce qui fixe l'aspect du thème néon est rassemblé ici, sans dépendance
graphique : le rendu NumPy (exports) et la fenêtre Arcade lisent les mêmes
valeurs, ce qui garantit des images identiques dans les deux cas.

* **Couleurs.** Une teinte néon par équipe, dans un ordre fixe ; les membres
  d'une même équipe en reçoivent des nuances (comme la palette classique). Le
  numéro du joueur reste inscrit près de sa moto : l'identité ne repose pas sur
  la seule couleur. Le thème ``classic`` (palette validée pour les daltoniens)
  reste disponible.
* **Tailles.** Les épaisseurs des traces et les rayons de lueur sont des
  fractions de la taille d'une case : l'aspect ne dépend pas de la résolution.
* **Qualité.** ``low`` (traces à deux couches, pas d'animation de fond, peu de
  particules), ``medium`` (halo, animations, particules CPU, léger tremblement)
  et ``ultra`` (en plus : post-traitement *bloom* et davantage de particules).
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from tron.viz.palette import RGB, hex_to_rgb, mix, shade

RGBA = tuple[int, int, int, int]

THEMES = ("neon", "classic")
"""Thèmes de rendu disponibles (``classic`` : palette d'origine, validée CVD)."""
DEFAULT_THEME = "neon"
"""Thème de la ligne de commande (les fonctions de la bibliothèque gardent ``classic``)."""
QUALITY_NAMES = ("low", "medium", "ultra")
DEFAULT_QUALITY = "medium"

NEON_COLORS = (
    "#00E5FF",  # cyan
    "#FF00D4",  # magenta
    "#FF9D00",  # orange
    "#39FF88",  # vert
    "#9747FF",  # violet
    "#FF365B",  # rouge
    "#F5FF45",  # jaune
    "#367BFF",  # bleu
)
"""Teintes des équipes, dans l'ordre des équipes de la configuration."""

_COLOR_WORDS = {
    "cyan": 0,
    "turquoise": 0,
    "magenta": 1,
    "rose": 1,
    "pink": 1,
    "fuchsia": 1,
    "orange": 2,
    "vert": 3,
    "verte": 3,
    "green": 3,
    "violet": 4,
    "violette": 4,
    "purple": 4,
    "mauve": 4,
    "rouge": 5,
    "red": 5,
    "jaune": 6,
    "yellow": 6,
    "or": 6,
    "gold": 6,
    "bleu": 7,
    "bleue": 7,
    "blue": 7,
}
"""Noms d'équipe reconnus comme des couleurs (index dans ``NEON_COLORS``)."""


def _normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def team_hues(names: Sequence[str], n_teams: int | None = None) -> list[int]:
    """Teinte (index de ``NEON_COLORS``) de chaque équipe.

    Une équipe nommée par une couleur (« Rouge », « Blue »…) reçoit cette teinte ;
    les autres reçoivent, dans l'ordre, les teintes encore libres.
    """
    n = len(names) if n_teams is None else n_teams
    out: list[int | None] = [None] * n
    taken: set[int] = set()
    for t in range(min(n, len(names))):
        hue = _COLOR_WORDS.get(_normalise(names[t]))
        if hue is not None and hue not in taken:
            out[t] = hue
            taken.add(hue)
    free = [i for i in range(len(NEON_COLORS)) if i not in taken]
    result = []
    for t in range(n):
        hue = out[t]
        if hue is None:
            hue = free.pop(0) if free else t % len(NEON_COLORS)
        result.append(hue)
    return result


BACKGROUND = "#030712"
BACKGROUND_ALT = "#071525"
GRID = "#123047"
GRID_MAJOR = "#1D5579"
CORE_WHITE = (235, 250, 255)
"""Couleur du cœur lumineux des traces (blanc légèrement bleuté)."""

_MEMBER_SHADES = (0.0, 0.32, -0.28, 0.52, -0.42, 0.18, -0.14, 0.66)
_DEAD_GRAY = (88, 98, 112)


@dataclass(frozen=True, slots=True)
class Quality:
    """Niveau de qualité graphique (voir la docstring du module)."""

    name: str
    halo: bool
    """Halo large autour des traces (troisième couche de lueur)."""
    animated: bool
    """Fond animé, pulsation des motos, flux des liens d'alliance."""
    max_particles: int
    particle_scale: float
    """Multiplicateur du nombre de particules émises par événement."""
    shake: bool
    """Léger tremblement de caméra lors d'une élimination."""
    flash: bool
    """Bref éclair lors d'une élimination ou de la victoire."""
    bloom: bool
    """Post-traitement *bloom* (Arcade seulement, désactivé en cas d'erreur)."""
    grid_major: bool
    """Lignes majeures de la grille (toutes les 5 ou 10 cases)."""


QUALITIES: dict[str, Quality] = {
    "low": Quality(
        name="low",
        halo=False,
        animated=False,
        max_particles=80,
        particle_scale=0.4,
        shake=False,
        flash=False,
        bloom=False,
        grid_major=False,
    ),
    "medium": Quality(
        name="medium",
        halo=True,
        animated=True,
        max_particles=500,
        particle_scale=1.0,
        shake=True,
        flash=True,
        bloom=False,
        grid_major=True,
    ),
    "ultra": Quality(
        name="ultra",
        halo=True,
        animated=True,
        max_particles=1500,
        particle_scale=1.6,
        shake=True,
        flash=True,
        bloom=True,
        grid_major=True,
    ),
}


def resolve_quality(value: str | Quality) -> Quality:
    """Niveau de qualité désigné par son nom (``low``, ``medium``, ``ultra``)."""
    if isinstance(value, Quality):
        return value
    try:
        return QUALITIES[value.lower()]
    except KeyError:
        names = ", ".join(QUALITY_NAMES)
        raise ValueError(f"qualité graphique inconnue : {value!r} ({names})") from None


def resolve_theme(value: str) -> str:
    """Nom de thème validé (``neon`` ou ``classic``)."""
    name = value.lower()
    if name not in THEMES:
        raise ValueError(f"thème inconnu : {value!r} ({', '.join(THEMES)})")
    return name


def next_quality(quality: Quality) -> Quality:
    """Niveau suivant, de façon circulaire (touche ``G`` de la fenêtre)."""
    i = QUALITY_NAMES.index(quality.name) if quality.name in QUALITY_NAMES else 0
    return QUALITIES[QUALITY_NAMES[(i + 1) % len(QUALITY_NAMES)]]


# ---------------------------------------------------------------- animations
PULSE_SPEED = 5.0
"""Pulsation de la lueur des motos : rayon relatif ``1 + 0,2 · sin(5 t)``."""
PULSE_DEPTH = 0.2
DEATH_FADE = 0.6
"""Durée (s) du passage d'une trace à sa couleur d'élimination."""
SHAKE_DURATION = 0.22
SHAKE_AMPLITUDE = 0.18
"""Amplitude du tremblement en cases, plafonnée à ``SHAKE_MAX_PIXELS``."""
SHAKE_MAX_PIXELS = 4.0
SHAKE_MAX_SPEED = 20.0
"""Au-delà de cette vitesse (demi-coups/s), ni tremblement ni éclair : l'écran resterait agité."""
FLASH_DURATION = 0.18
FLASH_ALPHA = 46
RING_DURATION = 0.55
LINK_GROW = 0.45
"""Durée (s) de l'animation de formation d'une alliance (lien qui se tend)."""
LINK_FLOW_SPEED = 1.4
SCAN_PERIOD = 7.0
"""Période (s) de la bande lumineuse qui balaie lentement le fond de l'arène."""
MAX_EFFECT_PLIES = 12
"""Au-delà de ce nombre de demi-coups traités d'un coup (avance rapide), pas d'effets."""


def pulse(t: float, enabled: bool = True) -> float:
    """Facteur de pulsation de la lueur d'une moto au temps ``t`` (s)."""
    return 1.0 + PULSE_DEPTH * math.sin(PULSE_SPEED * t) if enabled else 1.0


# ------------------------------------------------------------------ tailles
@dataclass(frozen=True, slots=True)
class TrailMetrics:
    """Géométrie des traces lumineuses pour une taille de case (pixels)."""

    cell: int
    body: float
    """Largeur du corps coloré de la trace."""
    core: float
    """Largeur du cœur blanc."""
    sigma_tight: float
    """Écart type de la lueur serrée."""
    sigma_wide: float
    """Écart type du halo large (0 sans halo)."""
    gain_body: float
    gain_tight: float
    gain_wide: float
    margin: int
    """Débord des tuiles de lueur autour de la case (≥ 3 écarts types)."""

    @property
    def tile(self) -> int:
        """Côté (pixels) d'une tuile de trace : la case et son débord."""
        return self.cell + 2 * self.margin


def trail_metrics(cell: int, quality: Quality | str = DEFAULT_QUALITY) -> TrailMetrics:
    """Épaisseurs et lueurs des traces pour des cases de ``cell`` pixels."""
    q = resolve_quality(quality)
    c = max(2, int(cell))
    body = max(1.5, 0.40 * c)
    core = min(body, max(1.0, 0.13 * c))
    sigma_tight = max(0.8, 0.16 * c)
    sigma_wide = max(1.6, 0.42 * c) if q.halo else 0.0
    margin = math.ceil(3.0 * max(sigma_tight, sigma_wide)) + 1
    return TrailMetrics(
        cell=c,
        body=body,
        core=core,
        sigma_tight=sigma_tight,
        sigma_wide=sigma_wide,
        gain_body=0.62,
        gain_tight=0.42 if q.halo else 0.55,
        gain_wide=0.30 if q.halo else 0.0,
        margin=margin,
    )


def arena_padding(cell: int) -> int:
    """Marge (pixels) autour de l'arène pour son liseré lumineux (rendu NumPy)."""
    return max(6, round(0.75 * cell))


def major_step(width: int, height: int) -> int:
    """Pas des lignes majeures de la grille : 5 cases, ou 10 au-delà de 30 cases."""
    return 10 if max(width, height) > 30 else 5


# ------------------------------------------------------------------- thème
@dataclass(frozen=True, slots=True)
class NeonTheme:
    """Couleurs du thème néon pour une partie (une teinte par équipe)."""

    background: RGB
    background_alt: RGB
    grid: RGB
    grid_major: RGB
    border: RGB
    text: RGB
    text_dim: RGB
    text_muted: RGB
    accent: RGB
    danger: RGB
    success: RGB
    warning: RGB
    panel: RGBA
    panel_edge: RGBA
    obstacle: RGB
    obstacle_edge: RGB
    team: tuple[RGB, ...]
    player: tuple[RGB, ...]
    head: tuple[RGB, ...]
    """Reflets clairs des motos."""
    core: tuple[RGB, ...]
    """Cœur blanc teinté de chaque trace."""
    dead: tuple[RGB, ...]
    """Couleur éteinte d'un joueur éliminé."""

    def trail_color(self, player: int, alive: bool, fade: float = 1.0) -> RGB:
        """Couleur de trace ; ``fade`` ∈ [0, 1] mesure l'extinction après élimination."""
        if alive:
            return self.player[player]
        return mix(self.player[player], self.dead[player], max(0.0, min(1.0, fade)))

    def territory(self, player: int, alpha: int = 40) -> RGBA:
        """Teinte translucide du territoire de Voronoi d'un joueur."""
        r, g, b = self.player[player]
        return (r, g, b, alpha)


def make_neon_theme(team_index: Sequence[int], team_names: Sequence[str] = ()) -> NeonTheme:
    """Thème néon pour des joueurs dont ``team_index`` donne l'équipe.

    ``team_names`` (facultatif) : une équipe nommée par une couleur en reçoit la teinte.
    """
    n_teams = max(team_index) + 1 if team_index else 0
    hues = team_hues(team_names, n_teams)
    teams = tuple(hex_to_rgb(NEON_COLORS[hues[t]]) for t in range(n_teams))
    seen = [0] * n_teams
    players: list[RGB] = []
    for t in team_index:
        players.append(shade(teams[t], _MEMBER_SHADES[seen[t] % len(_MEMBER_SHADES)]))
        seen[t] += 1
    background = hex_to_rgb(BACKGROUND)
    cyan = hex_to_rgb(NEON_COLORS[0])
    return NeonTheme(
        background=background,
        background_alt=hex_to_rgb(BACKGROUND_ALT),
        grid=hex_to_rgb(GRID),
        grid_major=hex_to_rgb(GRID_MAJOR),
        border=cyan,
        text=(226, 246, 255),
        text_dim=(140, 182, 208),
        text_muted=(82, 112, 138),
        accent=cyan,
        danger=hex_to_rgb("#FF365B"),
        success=hex_to_rgb("#39FF88"),
        warning=hex_to_rgb("#FF9D00"),
        panel=(7, 21, 37, 224),
        panel_edge=(0, 229, 255, 120),
        obstacle=(30, 40, 54),
        obstacle_edge=(255, 157, 0),
        team=teams,
        player=tuple(players),
        head=tuple(mix(c, (255, 255, 255), 0.35) for c in players),
        core=tuple(mix(c, CORE_WHITE, 0.80) for c in players),
        dead=tuple(mix(mix(c, _DEAD_GRAY, 0.62), background, 0.40) for c in players),
    )


def with_alpha(color: Sequence[int], alpha: float) -> RGBA:
    """Couleur RGBA, ``alpha`` ∈ [0, 1] borné."""
    a = round(255 * max(0.0, min(1.0, alpha)))
    return int(color[0]), int(color[1]), int(color[2]), a


__all__ = [
    "BACKGROUND",
    "CORE_WHITE",
    "DEFAULT_QUALITY",
    "DEFAULT_THEME",
    "NEON_COLORS",
    "QUALITIES",
    "QUALITY_NAMES",
    "THEMES",
    "NeonTheme",
    "Quality",
    "TrailMetrics",
    "arena_padding",
    "major_step",
    "make_neon_theme",
    "next_quality",
    "pulse",
    "resolve_quality",
    "resolve_theme",
    "team_hues",
    "trail_metrics",
    "with_alpha",
]
