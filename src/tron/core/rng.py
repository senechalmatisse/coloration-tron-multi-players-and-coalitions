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


def make_rng(seed: int, *keys: int) -> np.random.Generator:
    """Générateur indépendant pour le flux ``keys`` de la graine ``seed``."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=tuple(keys)))


def derive_seed(seed: int, *keys: int) -> int:
    """Graine dérivée sur 63 bits, stable d'une plateforme à l'autre."""
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(keys))
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
