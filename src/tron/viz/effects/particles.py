"""Particules CPU vectorisées (NumPy), déterministes et indépendantes de l'affichage.

Les positions sont en unités de case (origine au coin haut gauche de la grille,
y vers le bas), les vitesses en cases par seconde : le rendu NumPy et la
fenêtre Arcade les convertissent chacun en pixels. L'aléa passe par
:func:`tron.core.rng.make_rng` (flux ``STREAM_EFFECTS``), sans lien avec celui
de la partie : deux exécutions de la même scène donnent les mêmes particules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tron.core.rng import STREAM_EFFECTS, make_rng

Color = Sequence[int]


@dataclass(frozen=True, slots=True)
class ParticleView:
    """Particules vivantes, prêtes à dessiner (tableaux de même longueur)."""

    x: np.ndarray
    y: np.ndarray
    size: np.ndarray
    """Rayon en cases."""
    alpha: np.ndarray
    """Opacité dans [0, 1]."""
    color: np.ndarray
    """Couleurs (n, 3) en 0..255."""

    def __len__(self) -> int:
        return int(self.x.shape[0])


class ParticleSystem:
    """Réserve de particules de capacité fixe (au-delà, les émissions sont tronquées)."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = max(0, int(capacity))
        self.rng = make_rng(seed, STREAM_EFFECTS)
        n = self.capacity
        self.x = np.zeros(n, dtype=np.float32)
        self.y = np.zeros(n, dtype=np.float32)
        self.vx = np.zeros(n, dtype=np.float32)
        self.vy = np.zeros(n, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.float32)
        self.life = np.ones(n, dtype=np.float32)
        self.size = np.zeros(n, dtype=np.float32)
        self.drag = np.zeros(n, dtype=np.float32)
        self.shrink = np.zeros(n, dtype=np.float32)
        self.color = np.zeros((n, 3), dtype=np.float32)
        self.count = 0
        self.dropped = 0
        """Particules refusées faute de place (statistique de diagnostic)."""

    def __len__(self) -> int:
        return self.count

    def clear(self) -> None:
        self.count = 0

    def emit(
        self,
        n: int,
        x: float,
        y: float,
        colors: Sequence[Color] | Color,
        speed: tuple[float, float] = (1.0, 3.0),
        life: tuple[float, float] = (0.4, 0.8),
        size: tuple[float, float] = (0.08, 0.16),
        angle: float = 0.0,
        spread: float = 2 * math.pi,
        drag: float = 2.0,
        jitter: float = 0.0,
        shrink: float = 0.6,
    ) -> int:
        """Émet ``n`` particules depuis ``(x, y)`` ; renvoie le nombre réellement émis.

        ``angle`` (radians, 0 vers l'est, sens horaire puisque y descend) et
        ``spread`` délimitent le cône d'émission ; ``jitter`` disperse les points
        de départ ; ``colors`` est une couleur ou une liste de couleurs tirées au sort.
        """
        n = int(n)
        room = self.capacity - self.count
        k = max(0, min(n, room))
        self.dropped += max(0, n) - k
        if k == 0:
            return 0
        rng = self.rng
        s = slice(self.count, self.count + k)
        theta = angle + (rng.random(k) - 0.5) * spread
        v = speed[0] + rng.random(k) * (speed[1] - speed[0])
        self.x[s] = x + (rng.random(k) - 0.5) * 2 * jitter
        self.y[s] = y + (rng.random(k) - 0.5) * 2 * jitter
        self.vx[s] = np.cos(theta) * v
        self.vy[s] = np.sin(theta) * v
        self.age[s] = 0.0
        self.life[s] = life[0] + rng.random(k) * (life[1] - life[0])
        self.size[s] = size[0] + rng.random(k) * (size[1] - size[0])
        self.drag[s] = drag
        self.shrink[s] = shrink
        palette = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        self.color[s] = palette[rng.integers(0, len(palette), k)]
        self.count += k
        return k

    def update(self, dt: float) -> None:
        """Avance de ``dt`` secondes : déplacement, freinage, vieillissement, retrait."""
        n = self.count
        if n == 0 or dt <= 0:
            return
        s = slice(0, n)
        damp = np.exp(-self.drag[s] * dt)
        self.vx[s] *= damp
        self.vy[s] *= damp
        self.x[s] += self.vx[s] * dt
        self.y[s] += self.vy[s] * dt
        self.age[s] += dt
        keep = np.flatnonzero(self.age[s] < self.life[s])
        if keep.shape[0] == n:
            return
        m = keep.shape[0]
        for arr in (
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.age,
            self.life,
            self.size,
            self.drag,
            self.shrink,
        ):
            arr[:m] = arr[keep]
        self.color[:m] = self.color[keep]
        self.count = m

    def view(self) -> ParticleView:
        """Particules vivantes : position, rayon, opacité et couleur à l'instant présent."""
        s = slice(0, self.count)
        k = np.clip(self.age[s] / self.life[s], 0.0, 1.0)
        alpha = (1.0 - k) ** 1.6
        size = self.size[s] * (1.0 - self.shrink[s] * k)
        return ParticleView(
            x=self.x[s].copy(),
            y=self.y[s].copy(),
            size=size,
            alpha=alpha,
            color=self.color[s].copy(),
        )


__all__ = ["ParticleSystem", "ParticleView"]
