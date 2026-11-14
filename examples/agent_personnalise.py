"""Un agent personnalisé, enregistré puis utilisé dans une partie.

L'agent « prudent » joue le coup qui laisse le plus de cases voisines libres à sa
nouvelle tête ; il respecte le contrat des agents : l'état reçu est rendu intact
(``apply``/``undo`` protégés par ``try/finally``).

    python examples/agent_personnalise.py
"""

from __future__ import annotations

from tron.api import (
    Agent,
    DecisionDetails,
    GameState,
    SearchStats,
    register_agent,
    run_game,
)


@register_agent("prudent")
class PrudentAgent(Agent):
    """Maximise le nombre de voisins libres de la tête après le coup."""

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        scores: dict[int, float] = {}
        for m in moves:
            state.apply(m)
            try:
                scores[m] = float(len(state.legal_moves(self.player)))
            finally:
                state.undo()
        best = max(scores.values())
        move = self.pick([m for m in moves if scores[m] == best])
        self._details = DecisionDetails(scores=tuple(scores.items()), note="voisins libres")
        return move, SearchStats(nodes=len(moves) + 1, depth=1, engine=self.name)


def main() -> None:
    config = {
        "name": "prudent_contre_glouton",
        "grid": 12,
        "seed": 5,
        "players": [
            {"name": "prudent", "team": "A", "agent": "prudent"},
            {"name": "glouton", "team": "B", "agent": "greedy"},
        ],
    }
    result = run_game(config)
    print(f"vainqueur : {result.winning_team or 'aucun'} en {result.n_moves} coups")


if __name__ == "__main__":
    main()
