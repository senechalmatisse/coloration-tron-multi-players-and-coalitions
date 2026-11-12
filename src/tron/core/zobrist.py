"""Tables de hachage de Zobrist (F3.6).

La clé d'un état combine par XOR : les cases occupées, la position de la tête de
chaque joueur vivant, l'ensemble des joueurs vivants et le joueur au trait ; dans
les modes à règles (P09), aussi le nombre de tours complets, les scores et les
obstacles. Les tables sont tirées avec une graine fixe : les clés sont identiques
d'un processus à l'autre, ce qui préserve la reproductibilité.

Les compteurs (tours, scores) ne sont pas bornés : leurs valeurs de hachage sont
calculées par :func:`tron.core.rng.hash_index` plutôt que lues dans une table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tron.core.rng import hash_index

_SEED = 0x7A0B_1257
_MAX_PLAYERS = 8
_CACHE: dict[int, ZobristTables] = {}


_ROUND_SALT = 0x52_4F_55_4E_44  # « ROUND »
_SCORE_SALT = 0x53_43_4F_52_45  # « SCORE »


@dataclass(frozen=True, slots=True)
class ZobristTables:
    occ: list[int]
    head: list[list[int]]
    alive: list[int]
    turn: list[int]
    obst: list[int]
    """Case occupée par un obstacle dynamique (P09.3), distincte d'un mur de joueur."""

    @staticmethod
    def round_key(rounds: int) -> int:
        """Valeur de hachage du nombre de tours complets (modes à règles, P09)."""
        return hash_index(_SEED, _ROUND_SALT, rounds)

    @staticmethod
    def score_key(team: int, score: int) -> int:
        """Valeur de hachage du score d'une équipe de victoire (contrôle de zone, P09.2)."""
        return hash_index(_SEED, _SCORE_SALT, team, score)


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
        # Tirée après les tables historiques : les clés du mode classique sont inchangées.
        obst = rng.integers(1, hi, size=size, dtype=np.int64).tolist()
        tables = ZobristTables(occ=occ, head=head, alive=alive, turn=turn, obst=obst)
        _CACHE[size] = tables
    return tables
