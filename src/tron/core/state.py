"""État du jeu et règles (F1.1 à F1.5, F1.8).

Représentation (§6.3) : grille aplatie ``int8`` de taille (H+2) × (W+2) avec une
bordure sentinelle (−2), −1 pour une case libre et ``i`` pour un mur du joueur i
(la tête comprise). La case (x, y) a l'indice ``(y+1)·(W+2) + (x+1)`` ; nord = y − 1.

``apply`` et ``undo`` modifient l'état en place, sans copie : un historique
permet de défaire les coups, et la clé de Zobrist ainsi que les compteurs
d'équipes vivantes sont tenus à jour de façon incrémentale. Les vues
``memoryview`` sur les tableaux NumPy rendent les accès scalaires rapides en
Python ; les tableaux eux-mêmes sont passés aux noyaux compilés.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tron.core.config import GameConfig
from tron.core.placement import initial_positions
from tron.core.zobrist import ZobristTables, zobrist_tables

FREE = -1
BORDER = -2


@dataclass(frozen=True, slots=True)
class Outcome:
    """Issue d'une partie terminée (F1.5)."""

    winning_team: int
    winners: tuple[int, ...]
    ranks: tuple[int, ...]
    """Rang de survie par joueur : 1 pour les survivants, puis ordre inverse d'élimination."""
    elimination_order: tuple[int, ...]
    n_moves: int
    n_plies: int


class GameState:
    """État mutable d'une partie : grille, têtes, joueurs vivants, joueur au trait."""

    __slots__ = (
        "_a",
        "_d",
        "_g",
        "_h",
        "_hist",
        "_off",
        "_team",
        "_z",
        "alive",
        "config",
        "current",
        "death_ply",
        "elim_order",
        "grid",
        "heads",
        "height",
        "key",
        "n",
        "n_moves",
        "n_teams",
        "offsets",
        "ply",
        "positions",
        "size",
        "stride",
        "team",
        "team_alive",
        "teams_alive",
        "width",
    )

    def __init__(
        self,
        config: GameConfig,
        positions: Sequence[tuple[int, int]] | None = None,
        first_player: int | None = None,
    ) -> None:
        w, h = config.width, config.height
        stride = w + 2
        size = (h + 2) * stride
        n = config.n_players
        self.config = config
        self.width = w
        self.height = h
        self.stride = stride
        self.size = size
        self.n = n
        grid = np.full(size, FREE, dtype=np.int8)
        g2 = grid.reshape(h + 2, stride)
        g2[0, :] = BORDER
        g2[-1, :] = BORDER
        g2[:, 0] = BORDER
        g2[:, -1] = BORDER
        self.grid = grid
        self.heads = np.zeros(n, dtype=np.int64)
        self.alive = np.ones(n, dtype=np.bool_)
        self.death_ply = np.full(n, -1, dtype=np.int64)
        self.offsets = np.array([-stride, 1, stride, -1], dtype=np.int64)
        self._off = (-stride, 1, stride, -1)
        self._bind_views()
        team = list(config.team_index)
        self._team = team
        self.team = np.asarray(team, dtype=np.int64)
        self.n_teams = len(config.team_names)
        self.team_alive = [team.count(t) for t in range(self.n_teams)]
        self.teams_alive = self.n_teams
        if positions is None:
            positions = initial_positions(config)
        if len(positions) != n:
            raise ValueError(f"{len(positions)} positions pour {n} joueurs")
        z: ZobristTables = zobrist_tables(size)
        self._z = z
        key = 0
        g = self._g
        for p, (x, y) in enumerate(positions):
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"position hors grille : ({x}, {y})")
            c = (y + 1) * stride + x + 1
            if g[c] != FREE:
                raise ValueError(f"case ({x}, {y}) déjà occupée")
            g[c] = p
            self._h[p] = c
            key ^= z.occ[c] ^ z.head[p][c] ^ z.alive[p]
        self.positions = tuple((int(x), int(y)) for x, y in positions)
        cur = config.first_player if first_player is None else first_player
        if not 0 <= cur < n:
            raise ValueError(f"premier joueur invalide : {cur}")
        self.current = cur
        self.key = key ^ z.turn[cur]
        self.ply = 0
        self.n_moves = 0
        self.elim_order: list[int] = []
        self._hist: list[tuple[int, int, int, int]] = []

    def _bind_views(self) -> None:
        self._g = memoryview(self.grid)
        self._h = memoryview(self.heads)
        self._a = memoryview(self.alive)
        self._d = memoryview(self.death_ply)

    # ------------------------------------------------------------ coordonnées
    def index(self, x: int, y: int) -> int:
        """Indice aplati de la case (x, y)."""
        return (y + 1) * self.stride + x + 1

    def coords(self, cell: int) -> tuple[int, int]:
        """Coordonnées (x, y) d'un indice aplati."""
        r, c = divmod(cell, self.stride)
        return c - 1, r - 1

    def head(self, player: int) -> tuple[int, int]:
        return self.coords(self._h[player])

    def cell(self, x: int, y: int) -> int:
        """Contenu de la case : −1 libre, i mur du joueur i."""
        return int(self._g[self.index(x, y)])

    def is_alive(self, player: int) -> bool:
        return bool(self._a[player])

    def alive_players(self) -> list[int]:
        return [p for p in range(self.n) if self._a[p]]

    def play_order(self, start: int | None = None) -> list[int]:
        """Joueurs vivants dans l'ordre de jeu, à partir de ``start`` (inclus)."""
        s = self.current if start is None else start
        return [q for k in range(self.n) if self._a[q := (s + k) % self.n]]

    # ----------------------------------------------------------------- règles
    def legal_moves(self, player: int | None = None) -> list[int]:
        """Directions menant à une case libre (F1.3)."""
        p = self.current if player is None else player
        if not self._a[p]:
            return []
        g = self._g
        h = self._h[p]
        s = self.stride
        out = []
        if g[h - s] == FREE:
            out.append(0)
        if g[h + 1] == FREE:
            out.append(1)
        if g[h + s] == FREE:
            out.append(2)
        if g[h - 1] == FREE:
            out.append(3)
        return out

    def has_legal_move(self, player: int | None = None) -> bool:
        p = self.current if player is None else player
        if not self._a[p]:
            return False
        g = self._g
        h = self._h[p]
        s = self.stride
        return g[h - s] == FREE or g[h + 1] == FREE or g[h + s] == FREE or g[h - 1] == FREE

    def apply(self, direction: int) -> bool:
        """Joue le coup du joueur au trait ; renvoie ``False`` s'il est éliminé.

        Un coup hors grille, sur une case occupée ou ``PASS`` élimine le joueur
        (F1.4) : ses murs restent en place et il sort de la rotation.
        """
        p = self.current
        h = self._h[p]
        z = self._z
        prev_key = key = self.key
        t = h + self._off[direction] if 0 <= direction <= 3 else -1
        if t >= 0 and self._g[t] == FREE:
            self._g[t] = p
            self._h[p] = t
            self.n_moves += 1
            key ^= z.occ[t] ^ z.head[p][h] ^ z.head[p][t]
        else:
            t = -1
            self._a[p] = False
            self._d[p] = self.ply
            self.elim_order.append(p)
            tm = self._team[p]
            self.team_alive[tm] -= 1
            if self.team_alive[tm] == 0:
                self.teams_alive -= 1
            key ^= z.alive[p] ^ z.head[p][h]
        self.ply += 1
        nxt = self._next_alive(p)
        self.current = nxt
        self.key = key ^ z.turn[p] ^ z.turn[nxt]
        self._hist.append((p, h, t, prev_key))
        return t >= 0

    def undo(self) -> None:
        """Défait le dernier coup (déplacement ou élimination)."""
        p, h, t, prev_key = self._hist.pop()
        if t >= 0:
            self._g[t] = FREE
            self._h[p] = h
            self.n_moves -= 1
        else:
            self._a[p] = True
            self._d[p] = -1
            self.elim_order.pop()
            tm = self._team[p]
            if self.team_alive[tm] == 0:
                self.teams_alive += 1
            self.team_alive[tm] += 1
        self.ply -= 1
        self.current = p
        self.key = prev_key

    def _next_alive(self, p: int) -> int:
        a = self._a
        n = self.n
        q = p
        for _ in range(n):
            q += 1
            if q == n:
                q = 0
            if a[q]:
                return q
        return p

    def is_terminal(self) -> bool:
        """Vrai quand il ne reste qu'une équipe en vie (F1.5)."""
        return self.teams_alive <= 1

    def winning_team(self) -> int:
        """Indice de l'équipe gagnante, ou −1 si la partie n'est pas finie."""
        if self.teams_alive != 1:
            return -1
        for p in range(self.n):
            if self._a[p]:
                return self._team[p]
        return -1  # pragma: no cover - impossible en tour par tour

    def outcome(self) -> Outcome | None:
        """Vainqueur, rangs de survie et nombre de coups ; ``None`` si la partie continue."""
        if not self.is_terminal():
            return None
        winners = tuple(p for p in range(self.n) if self._a[p])
        n_alive = len(winners)
        ranks = [1] * self.n
        order = self.elim_order
        m = len(order)
        for i, p in enumerate(order):
            ranks[p] = 1 + n_alive + (m - 1 - i)
        return Outcome(
            winning_team=self.winning_team(),
            winners=winners,
            ranks=tuple(ranks),
            elimination_order=tuple(order),
            n_moves=self.n_moves,
            n_plies=self.ply,
        )

    # ---------------------------------------------------------------- outils
    def copy(self, history: bool = True) -> GameState:
        """Copie indépendante ; sans ``history``, les coups antérieurs ne peuvent être défaits."""
        new = GameState.__new__(GameState)
        for name in (
            "config",
            "width",
            "height",
            "stride",
            "size",
            "n",
            "offsets",
            "_off",
            "_team",
            "team",
            "n_teams",
            "positions",
            "_z",
            "current",
            "key",
            "ply",
            "n_moves",
            "teams_alive",
        ):
            setattr(new, name, getattr(self, name))
        new.grid = self.grid.copy()
        new.heads = self.heads.copy()
        new.alive = self.alive.copy()
        new.death_ply = self.death_ply.copy()
        new._bind_views()
        new.team_alive = list(self.team_alive)
        new.elim_order = list(self.elim_order)
        new._hist = list(self._hist) if history else []
        return new

    def compute_key(self) -> int:
        """Recalcule la clé de Zobrist à partir de zéro (contrôle de cohérence)."""
        z = self._z
        key = 0
        for c in range(self.size):
            if self._g[c] >= 0:
                key ^= z.occ[c]
        for p in range(self.n):
            if self._a[p]:
                key ^= z.head[p][self._h[p]] ^ z.alive[p]
        return key ^ z.turn[self.current]

    def grid2d(self) -> np.ndarray:
        """Vue (H, W) de la grille sans la bordure."""
        return self.grid.reshape(self.height + 2, self.stride)[1:-1, 1:-1]

    def to_text(self) -> str:
        """Représentation ASCII : ``.`` libre, ``a``… murs, ``A``… têtes, ``x`` tête éliminée."""
        heads = {self._h[p]: p for p in range(self.n)}
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                c = self.index(x, y)
                v = self._g[c]
                if v == FREE:
                    row.append(".")
                elif heads.get(c) == v:
                    row.append(chr(ord("A") + v) if self._a[v] else "x")
                else:
                    row.append(chr(ord("a") + v))
            rows.append("".join(row))
        return "\n".join(rows)

    @classmethod
    def from_text(cls, config: GameConfig, text: str, current: int | None = None) -> GameState:
        """Construit un état à partir d'une carte ASCII (têtes ``A``…, murs ``a``…)."""
        lines = [ln.strip() for ln in text.strip().splitlines()]
        if len(lines) != config.height or any(len(ln) != config.width for ln in lines):
            raise ValueError("dimensions de la carte incompatibles avec la configuration")
        heads: dict[int, tuple[int, int]] = {}
        for y, ln in enumerate(lines):
            for x, ch in enumerate(ln):
                if "A" <= ch <= "H":
                    heads[ord(ch) - ord("A")] = (x, y)
        positions = [heads[p] for p in range(config.n_players)]
        state = cls(config, positions=positions, first_player=current)
        for y, ln in enumerate(lines):
            for x, ch in enumerate(ln):
                if "a" <= ch <= "h":
                    state._g[state.index(x, y)] = ord(ch) - ord("a")
        state.key = state.compute_key()
        return state

    # ------------------------------------------------------------ instantané
    def to_snapshot(self) -> dict[str, Any]:
        """Instantané sérialisable (JSON) de tout l'état, historique ``apply``/``undo`` compris.

        La grille (bordure comprise) est codée sur un caractère par case :
        ``0`` bordure, ``1`` libre, ``2`` + i mur du joueur i.
        """
        return {
            "width": self.width,
            "height": self.height,
            "grid": "".join(chr(48 + int(v) + 2) for v in self.grid.tolist()),
            "heads": [int(h) for h in self.heads.tolist()],
            "alive": [bool(a) for a in self.alive.tolist()],
            "death_ply": [int(d) for d in self.death_ply.tolist()],
            "positions": [list(p) for p in self.positions],
            "current": self.current,
            "ply": self.ply,
            "n_moves": self.n_moves,
            "elim_order": list(self.elim_order),
            "team_alive": list(self.team_alive),
            "teams_alive": self.teams_alive,
            "key": self.key,
            "history": [list(h) for h in self._hist],
        }

    @classmethod
    def from_snapshot(cls, config: GameConfig, data: Mapping[str, Any]) -> GameState:
        """Reconstruit un état depuis :meth:`to_snapshot` en vérifiant sa cohérence.

        Lève ``ValueError`` si les dimensions, la grille, les têtes, les joueurs
        vivants, les compteurs d'équipes ou la clé de Zobrist ne correspondent pas.
        """
        try:
            w, h = int(data["width"]), int(data["height"])
            if (w, h) != (config.width, config.height):
                raise ValueError(f"dimensions {w}x{h} incompatibles avec la configuration")
            state = cls(config, positions=[tuple(p) for p in data["positions"]])
            text = str(data["grid"])
            if len(text) != state.size:
                raise ValueError("taille de grille incohérente")
            values = [ord(ch) - 50 for ch in text]
            if any(not BORDER <= v < config.n_players for v in values):
                raise ValueError("valeur de case invalide dans la grille")
            g2 = np.asarray(values, dtype=np.int8).reshape(h + 2, w + 2)
            if (g2[0, :] != BORDER).any() or (g2[-1, :] != BORDER).any():
                raise ValueError("bordure de grille invalide")
            if (g2[:, 0] != BORDER).any() or (g2[:, -1] != BORDER).any():
                raise ValueError("bordure de grille invalide")
            if (g2[1:-1, 1:-1] == BORDER).any():
                raise ValueError("bordure à l'intérieur de la grille")
            n = config.n_players
            heads = [int(x) for x in data["heads"]]
            alive = [bool(x) for x in data["alive"]]
            death = [int(x) for x in data["death_ply"]]
            if not (len(heads) == len(alive) == len(death) == n):
                raise ValueError("nombre de joueurs incohérent")
            state.grid[:] = g2.ravel()
            state.heads[:] = heads
            state.alive[:] = alive
            state.death_ply[:] = death
            for p in range(n):
                if state.grid[heads[p]] != p:
                    raise ValueError(f"tête du joueur {p} hors de ses murs")
                if alive[p] != (death[p] < 0):
                    raise ValueError(f"élimination du joueur {p} incohérente")
            state.current = int(data["current"])
            state.ply = int(data["ply"])
            state.n_moves = int(data["n_moves"])
            state.elim_order = [int(x) for x in data["elim_order"]]
            if sorted(state.elim_order) != sorted(p for p in range(n) if not alive[p]):
                raise ValueError("ordre d'élimination incohérent")
            team = state._team
            counts = [
                sum(1 for p in range(n) if alive[p] and team[p] == t)
                for t in range(len(state.team_alive))
            ]
            if counts != [int(x) for x in data["team_alive"]]:
                raise ValueError("compteurs d'équipes vivantes incohérents")
            state.team_alive = counts
            state.teams_alive = sum(1 for c in counts if c > 0)
            if state.teams_alive != int(data["teams_alive"]):
                raise ValueError("nombre d'équipes vivantes incohérent")
            if not 0 <= state.current < n or (state.teams_alive > 1 and not alive[state.current]):
                raise ValueError("joueur au trait invalide")
            state._hist = [
                (int(a), int(b), int(c), int(d)) for a, b, c, d in data.get("history", [])
            ]
            key = int(data["key"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"instantané d'état incomplet : {exc}") from exc
        if state.compute_key() != key:
            raise ValueError("clé de Zobrist incohérente avec la grille")
        state.key = key
        return state

    def __repr__(self) -> str:
        return (
            f"GameState({self.width}x{self.height}, ply={self.ply}, "
            f"current={self.current}, alive={self.alive_players()})"
        )
