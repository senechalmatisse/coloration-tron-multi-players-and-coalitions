"""Tables de hachage de Zobrist (F3.6).

La clé d'un état combine par XOR : les cases occupées, la position de la tête de
chaque joueur vivant, l'ensemble des joueurs vivants et le joueur au trait. Les
tables sont tirées avec une graine fixe : les clés sont identiques d'un processus
à l'autre, ce qui préserve la reproductibilité.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SEED = 0x7A0B_1257
_MAX_PLAYERS = 8
_CACHE: dict[int, ZobristTables] = {}


@dataclass(frozen=True, slots=True)
class ZobristTables:
    occ: list[int]
    head: list[list[int]]
    alive: list[int]
    turn: list[int]


def zobrist_tables(size: int) -> ZobristTables:
    """Tables pour une grille de ``size`` cases (bordure comprise), mises en cache."""
    tables = _CACHE.get(size)
    if tables is None:
        rng = np.random.default_rng([_SEED, size])
        hi = np.iinfo(np.int64).max
        occ = rng.integers(1, hi, size=size, dtype=np.int64).tolist()
        head = rng.integers(1, hi, size=(_MAX_PLAYERS, size), dtype=np.int64).tolist()
        alive = rng.integers(1, hi, size=_MAX_PLAYERS, dtype=np.int64).tolist()
        turn = rng.integers(1, hi, size=_MAX_PLAYERS, dtype=np.int64).tolist()
        tables = ZobristTables(occ=occ, head=head, alive=alive, turn=turn)
        _CACHE[size] = tables
    return tables
