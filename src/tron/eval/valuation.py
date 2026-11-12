"""Valeur des feuilles de l'arbre de recherche (F2.2, F2.3).

Tous les moteurs partagent un même codage du vecteur v (une composante par joueur) :

* joueur vivant d'un état non terminal : v(i) = h(i), sa part de territoire
  (tête comprise, donc strictement positive) ;
* joueur éliminé : v(i) = ε·r/T ∈ [0, ε[, où r est le demi-coup de son
  élimination compté depuis la racine. Une défaite tardive vaut un peu mieux
  qu'une défaite rapide, et toute défaite vaut moins que toute part de territoire ;
* état terminal : chaque membre de l'équipe gagnante reçoit 1 + μ·(1 − r/T), où r
  est le demi-coup de fin de partie depuis la racine. C'est strictement plus que
  toute valeur heuristique (≤ 1), et une victoire rapide vaut mieux.

Toutes les composantes restent positives, ce qui préserve la validité de
l'élagage superficiel de Korf. Deux joueurs d'équipes différentes vérifient
v(p) + v(q) ≤ 1 + μ + ε, et la somme totale reste bornée par
S = T_max·(1 + μ) + n·ε, où T_max est la taille de la plus grande équipe.
L'échelle des défaites est très inférieure à celle du bonus de victoire,
lui-même inférieur à l'écart entre deux parts de territoire distinctes.
"""

from __future__ import annotations

from collections.abc import Sequence

from tron.core.config import GameConfig
from tron.core.state import GameState
from tron.eval.base import Heuristic

WIN_BONUS = 1e-3
"""μ : amplitude du bonus de victoire rapide."""
PLY_CAP = 1024
"""T : horizon de normalisation des demi-coups (bien au-delà des profondeurs utiles)."""


class Valuation:
    """Calcule le vecteur v d'une feuille à partir de l'heuristique et des règles."""

    def __init__(self, config: GameConfig, heuristic: Heuristic) -> None:
        n = config.n_players
        self.n = n
        self.team = list(config.victory_team_index)  # équipes qui gagnent (P09)
        self.heuristic = heuristic
        self.mu = WIN_BONUS
        self.cap = PLY_CAP
        self.eps = min(self.mu / (4 * n * self.cap), 1.0 / (4 * config.width * config.height))
        max_team = max(self.team.count(t) for t in set(self.team))
        self.max_component = 1.0 + self.mu
        self.pair_bound_diff = 1.0 + self.mu + self.eps
        self.pair_bound_same = 2.0 * (1.0 + self.mu)
        self.sum_bound = max(1.0 + n * self.eps, max_team * (1.0 + self.mu) + n * self.eps)

    def win_value(self, rel_ply: int) -> float:
        """Valeur d'une victoire obtenue ``rel_ply`` demi-coups après la racine."""
        r = min(max(rel_ply, 0), self.cap - 1)
        return 1.0 + self.mu * (1.0 - r / self.cap)

    def dead_value(self, rel_ply: int) -> float:
        """Valeur d'un joueur éliminé ``rel_ply`` demi-coups après la racine."""
        r = min(max(rel_ply, 0), self.cap - 1)
        return self.eps * r / self.cap

    def pair_bound(self, p: int, q: int) -> float:
        """Borne de v(p) + v(q) sur toutes les feuilles possibles."""
        return self.pair_bound_same if self.team[p] == self.team[q] else self.pair_bound_diff

    def vector(self, state: GameState, root_ply: int) -> list[float]:
        """Vecteur v de l'état ``state`` vu d'une recherche lancée au demi-coup ``root_ply``."""
        n = self.n
        d = state._d
        if state.over:
            winner = state.winning_team()
            if winner >= 0:
                win = self.win_value(state.ply - root_ply)
                team = self.team
                return [
                    win if team[p] == winner else self.dead_value(d[p] - root_ply) for p in range(n)
                ]
            # Égalité (P09) : les vivants se partagent la valeur comme sur une feuille.
        h = self.heuristic.shares(state)
        a = state._a
        return [h[p] if a[p] else self.dead_value(d[p] - root_ply) for p in range(n)]

    def scalar(self, state: GameState, root_ply: int, coeffs: Sequence[tuple[int, float]]) -> float:
        """Combinaison linéaire Σ c_j·v(j) ; évite l'heuristique si le camp est éliminé."""
        if not state.over and not any(state._a[j] for j, _ in coeffs):
            d = state._d
            return sum(w * self.dead_value(d[j] - root_ply) for j, w in coeffs)
        v = self.vector(state, root_ply)
        return sum(w * v[j] for j, w in coeffs)
