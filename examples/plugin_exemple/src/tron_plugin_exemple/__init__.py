"""Extension d'exemple (P12) : un agent et une heuristique pour Tron.

Les classes ne sont pas décorées par ``register_agent`` : c'est la déclaration
d'entry point (``pyproject.toml``) qui leur donne leur nom dans Tron.
"""

from __future__ import annotations

from typing import Any

from tron.agents import Agent, DecisionDetails, SearchStats
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState
from tron.eval import Heuristic


class PrudentAgent(Agent):
    """Joue le coup qui laisse le plus de voisins libres à sa nouvelle tête.

    Paramètre ``bonus_centre`` (≥ 0, défaut 0) : préférence pour le centre.
    """

    def __init__(
        self, player: int, config: GameConfig, *, bonus_centre: float = 0.0, **kw: Any
    ) -> None:
        super().__init__(player, config, **kw)
        if isinstance(bonus_centre, bool) or not isinstance(bonus_centre, int | float):
            raise ConfigError(f"bonus_centre doit être un nombre : {bonus_centre!r}")
        if bonus_centre < 0:
            raise ConfigError(f"bonus_centre doit être ≥ 0 : {bonus_centre}")
        self.bonus_centre = float(bonus_centre)

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        cx, cy = (state.width - 1) / 2, (state.height - 1) / 2
        scores: dict[int, float] = {}
        for m in moves:
            state.apply(m)
            try:  # contrat des agents : l'état reçu est rendu intact
                x, y = state.head(self.player)
                free = len(state.legal_moves(self.player))
                scores[m] = free - self.bonus_centre * (abs(x - cx) + abs(y - cy))
            finally:
                state.undo()
        best = max(scores.values())
        move = self.pick([m for m in moves if scores[m] == best])
        self._details = DecisionDetails(scores=tuple(scores.items()), note="voisins libres")
        return move, SearchStats(nodes=len(moves) + 1, depth=1, engine=self.name)


class CenterHeuristic(Heuristic):
    """Parts proportionnelles à la proximité du centre (contrat de ``Valuation``)."""

    def shares(self, state: GameState) -> list[float]:
        cx, cy = (state.width - 1) / 2, (state.height - 1) / 2
        raw = [0.0] * state.n
        for p in range(state.n):
            if state.is_alive(p):
                x, y = state.head(p)
                raw[p] = 1.0 / (1.0 + abs(x - cx) + abs(y - cy))
        total = sum(raw)
        return [r / total for r in raw] if total > 0 else raw
