"""Palette lisible par les daltoniens.

Couleurs catégorielles validées (séparation CVD des paires adjacentes ≥ 8 en
OKLab ×100, contraste ≥ 3:1 sur la surface sombre). Une teinte par équipe, dans
un ordre fixe ; les membres d'une même équipe en reçoivent des nuances plus
claires ou plus foncées. L'identité ne repose jamais sur la seule couleur : le
numéro du joueur est inscrit sur sa tête et le panneau d'information le nomme.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tron.style import (
    CATEGORICAL_DARK,
    CATEGORICAL_LIGHT,
    GRID_DARK,
    GRID_LIGHT,
    INK_MUTED,
    INK_PRIMARY_DARK,
    INK_PRIMARY_LIGHT,
    INK_SECONDARY_DARK,
    INK_SECONDARY_LIGHT,
    SURFACE_DARK,
    SURFACE_LIGHT,
)

RGB = tuple[int, int, int]

_MEMBER_SHADES = (0.0, 0.35, -0.30, 0.55, -0.45, 0.20, -0.15, 0.70)


def hex_to_rgb(value: str) -> RGB:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """Interpolation linéaire a → b (t = 0 : a, t = 1 : b)."""
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def shade(color: RGB, amount: float) -> RGB:
    """Éclaircit (amount > 0) ou assombrit (amount < 0) une couleur."""
    if amount >= 0:
        return mix(color, (255, 255, 255), amount)
    return mix(color, (0, 0, 0), -amount)


def luminance(rgb: RGB) -> float:
    def lin(c: int) -> float:
        x = c / 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def label_ink(fill: RGB) -> RGB:
    """Encre lisible sur un aplat : blanc ou quasi-noir selon la luminance."""
    return hex_to_rgb("#0b0b0b") if luminance(fill) > 0.35 else (255, 255, 255)


@dataclass(frozen=True, slots=True)
class Palette:
    """Couleurs d'une partie : surface, grille, murs, têtes, joueurs éliminés, territoires."""

    surface: RGB
    grid: RGB
    ink: RGB
    ink_secondary: RGB
    muted: RGB
    team: tuple[RGB, ...]
    player: tuple[RGB, ...]
    head: tuple[RGB, ...]
    dead: tuple[RGB, ...]
    territory: tuple[RGB, ...]


def make_palette(team_index: Sequence[int], dark: bool = True) -> Palette:
    """Palette pour des joueurs dont ``team_index`` donne l'équipe."""
    cats = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    surface = hex_to_rgb(SURFACE_DARK if dark else SURFACE_LIGHT)
    n_teams = max(team_index) + 1 if team_index else 0
    teams = tuple(hex_to_rgb(cats[t % len(cats)]) for t in range(n_teams))
    seen = [0] * n_teams
    players: list[RGB] = []
    for t in team_index:
        players.append(shade(teams[t], _MEMBER_SHADES[seen[t] % len(_MEMBER_SHADES)]))
        seen[t] += 1
    gray = hex_to_rgb(INK_MUTED)
    return Palette(
        surface=surface,
        grid=hex_to_rgb(GRID_DARK if dark else GRID_LIGHT),
        ink=hex_to_rgb(INK_PRIMARY_DARK if dark else INK_PRIMARY_LIGHT),
        ink_secondary=hex_to_rgb(INK_SECONDARY_DARK if dark else INK_SECONDARY_LIGHT),
        muted=gray,
        team=teams,
        player=tuple(players),
        head=tuple(mix(c, (255, 255, 255), 0.45) for c in players),
        dead=tuple(mix(mix(c, gray, 0.6), surface, 0.35) for c in players),
        territory=tuple(mix(surface, c, 0.28) for c in players),
    )
