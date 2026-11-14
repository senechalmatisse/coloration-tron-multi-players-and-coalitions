"""Heuristique du mode contrôle de zone.

Pour chaque joueur vivant p, trois caractéristiques dans [0, 1] :

* ``territory`` : part de Voronoi ;
* ``proximity`` : moyenne sur les zones de 1 / (1 + d), où d est la distance de
  Manhattan de la tête de p à la zone (0 dans la zone) ;
* ``score`` : (score de l'équipe de victoire de p + 1) / Σ_équipes (score + 1),
  partagé entre les membres vivants de l'équipe.

raw(p) = Σ w_k·x_k(p), plancher 1/(4·W·H) ; parts normalisées sur les vivants
(Σ h = 1, 0 pour un éliminé). Hors du mode ``zone_control`` (aucune zone), l'heuristique vaut Voronoi.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

from tron.core.state import GameState
from tron.eval.base import Heuristic, register_heuristic
from tron.eval.voronoi import VoronoiHeuristic

ZONE_COMPONENTS = ("territory", "proximity", "score")


@register_heuristic("zone")
class ZoneHeuristic(Heuristic):
    """Territoire, proximité des zones et avance au score (voir le module)."""

    def __init__(
        self,
        width: int,
        height: int,
        n_players: int,
        *,
        territory: float = 0.5,
        proximity: float = 0.3,
        score: float = 0.2,
        tie: str = "order",
    ) -> None:
        super().__init__(width, height, n_players)
        raw: dict[str, Any] = {"territory": territory, "proximity": proximity, "score": score}
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"poids '{name}' non numérique : {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"poids '{name}' invalide : {value!r}")
        total = sum(float(v) for v in raw.values())
        if total <= 0:
            raise ValueError("la somme des poids de l'heuristique zone doit être > 0")
        self.weights = {k: float(v) / total for k, v in raw.items()}
        self.voronoi = VoronoiHeuristic(width, height, n_players, tie=tie)
        self.floor = 1.0 / (4.0 * width * height)

    def shares(self, state: GameState) -> list[float]:
        zones = state.config.rules.zones
        if not zones:
            return self.voronoi.shares(state)
        counts, total = self.voronoi.counts(state)
        team = state._team
        scores = state.scores
        denom = sum(s + 1 for s in scores)
        members = [0] * len(scores)
        for p in range(state.n):
            if state._a[p]:
                members[team[p]] += 1
        wt, wp, ws = (self.weights[k] for k in ZONE_COMPONENTS)
        raw = [0.0] * state.n
        acc = 0.0
        for p in range(state.n):
            if not state._a[p]:
                continue
            x, y = state.head(p)
            prox = 0.0
            for z in zones:
                dx = max(z.x - x, 0, x - (z.x + z.width - 1))
                dy = max(z.y - y, 0, y - (z.y + z.height - 1))
                prox += 1.0 / (1 + dx + dy)
            prox /= len(zones)
            t = team[p]
            sc = (scores[t] + 1) / denom / members[t] if denom else 0.0
            terr = counts[p] / total if total else 0.0
            r = max(wt * terr + wp * prox + ws * sc, self.floor)
            raw[p] = r
            acc += r
        return [r / acc for r in raw] if acc > 0 else raw


__all__ = ["ZONE_COMPONENTS", "ZoneHeuristic"]
