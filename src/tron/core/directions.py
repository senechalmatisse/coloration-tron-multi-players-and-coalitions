"""Directions de déplacement. Origine en haut à gauche, nord = y − 1."""

from __future__ import annotations

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
PASS = -1
"""Pseudo-coup d'un joueur sans coup légal : il est éliminé."""

DIRECTIONS: tuple[int, ...] = (NORTH, EAST, SOUTH, WEST)
NAMES: tuple[str, ...] = ("N", "E", "S", "O")
DX: tuple[int, ...] = (0, 1, 0, -1)
DY: tuple[int, ...] = (-1, 0, 1, 0)

_ALIASES = {"N": 0, "E": 1, "S": 2, "O": 3, "W": 3, "-": PASS, "": PASS}


def direction_name(d: int) -> str:
    """Nom court d'une direction (``"-"`` pour l'absence de coup)."""
    return NAMES[d] if 0 <= d <= 3 else "-"


def parse_direction(value: str | int | None) -> int:
    """Convertit ``"N"``, ``"E"``, ``"S"``, ``"O"`` (ou ``"W"``), un entier ou ``None``."""
    if value is None:
        return PASS
    if isinstance(value, int):
        if value == PASS or 0 <= value <= 3:
            return value
        raise ValueError(f"direction invalide : {value}")
    key = value.strip().upper()
    if key not in _ALIASES:
        raise ValueError(f"direction invalide : {value!r}")
    return _ALIASES[key]
