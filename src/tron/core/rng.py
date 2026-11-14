"""Gestion des graines (F1.7).

Tout l'aléa d'une partie dérive d'une graine unique, par flux indépendants
(``numpy.random.SeedSequence``) : placement initial, départage des coups de
chaque agent. Les graines des parties d'un plan d'expériences dérivent de la
graine du plan et du numéro de partie, jamais de la condition : elles sont donc
appariées entre conditions.
"""

from __future__ import annotations

import numpy as np

STREAM_PLACEMENT = 1
STREAM_AGENT = 2
STREAM_PLAN = 3
STREAM_OBSTACLES = 4
STREAM_TRAIN = 5
"""Graines des épisodes d'entraînement (P10)."""
STREAM_EVAL = 6
"""Graines des parties d'évaluation (P10), disjointes de celles d'entraînement."""
STREAM_LEARNING = 7
"""Aléa propre à l'environnement d'apprentissage (conversion d'action illégale)."""
STREAM_EFFECTS = 8
"""Aléa des effets visuels (particules) : sans aucune influence sur la partie."""

_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1


def make_rng(seed: int, *keys: int) -> np.random.Generator:
    """Générateur indépendant pour le flux ``keys`` de la graine ``seed``."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=tuple(keys)))


def derive_seed(seed: int, *keys: int) -> int:
    """Graine dérivée sur 63 bits, stable d'une plateforme à l'autre."""
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(keys))
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def mix64(value: int) -> int:
    """Mélange splitmix64 d'un entier : valeur pseudo-aléatoire sur 63 bits, non nulle.

    Aléa *sans état* pour les règles consultées par la recherche (obstacles, P09.3) :
    la même entrée donne toujours la même sortie, sans générateur à copier ni à
    défaire, pour un coût de quelques opérations entières.
    """
    x = (value + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x ^= x >> 31
    return (x & _MASK63) or 1


def hash_index(seed: int, *keys: int) -> int:
    """Hachage déterministe de ``(seed, keys…)`` sur 63 bits (voir :func:`mix64`)."""
    h = mix64(seed & _MASK64)
    for k in keys:
        h = mix64(h ^ (k & _MASK64))
    return h
