"""Placement initial des joueurs (F1.6)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from tron.core.config import ConfigError, GameConfig
from tron.core.rng import STREAM_PLACEMENT, make_rng

MAX_ATTEMPTS = 20_000


def initial_positions(config: GameConfig) -> list[tuple[int, int]]:
    """Positions (x, y) de départ de chaque joueur, déterminées par la graine."""
    spec = config.placement
    rng = make_rng(config.seed, STREAM_PLACEMENT)
    if spec.mode == "manual":
        return [(int(x), int(y)) for x, y in spec.positions]
    if spec.mode == "random":
        return random_positions(
            config.width, config.height, config.n_players, spec.min_distance, rng
        )
    return regular_positions(
        config.width, config.height, config.n_players, spec.radius, rng if spec.rotate else None
    )


def regular_positions(
    width: int, height: int, n: int, radius: float, rng: np.random.Generator | None = None
) -> list[tuple[int, int]]:
    """Joueurs équirépartis sur un cercle centré (rotation aléatoire si ``rng``).

    Le joueur 0 est à gauche du centre (angle π) sauf rotation. Les coordonnées
    sont arrondies vers le centre, ce qui garde la symétrie sur les grilles paires.
    """
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    r = radius * min(width, height)
    theta0 = float(rng.uniform(0.0, 2.0 * math.pi)) if rng is not None else math.pi
    taken: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for k in range(n):
        theta = theta0 + 2.0 * math.pi * k / n
        x = _round_toward(cx, round(r * math.cos(theta), 9))
        y = _round_toward(cy, round(r * math.sin(theta), 9))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        if (x, y) in taken:
            x, y = _nearest_free(width, height, (x, y), taken)
        taken.add((x, y))
        out.append((x, y))
    return out


def random_positions(
    width: int, height: int, n: int, min_distance: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Positions uniformes avec une distance de Manhattan minimale entre joueurs."""
    out: list[tuple[int, int]] = []
    for _ in range(MAX_ATTEMPTS):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        if all(abs(x - px) + abs(y - py) >= min_distance for px, py in out):
            out.append((x, y))
            if len(out) == n:
                return out
    raise ConfigError(
        f"impossible de placer {n} joueurs à distance >= {min_distance} sur {width}x{height}"
    )


def _round_toward(center: float, delta: float) -> int:
    """Arrondit ``center + delta`` à l'entier, les demi-valeurs allant vers le centre."""
    v = center + delta
    return math.floor(v + 0.5) if delta < 0 else math.ceil(v - 0.5)


def _nearest_free(
    width: int, height: int, start: tuple[int, int], taken: set[tuple[int, int]]
) -> tuple[int, int]:
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        if (x, y) not in taken:
            return x, y
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    raise ConfigError("grille trop petite pour placer tous les joueurs")
