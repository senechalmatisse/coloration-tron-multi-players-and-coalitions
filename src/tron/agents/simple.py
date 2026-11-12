"""Agents de référence (F3.2) : aléatoire et glouton."""

from __future__ import annotations

from tron.agents.base import Agent, DecisionDetails, SearchStats, register_agent
from tron.agents.search import EPS
from tron.core.state import GameState


@register_agent("random")
class RandomAgent(Agent):
    """Tire un coup sûr (case libre) au hasard."""

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        move = moves[int(self.rng.integers(len(moves)))]
        self._details = DecisionDetails(
            scores=tuple((m, None) for m in moves), note="coup sûr tiré au hasard"
        )
        return move, SearchStats(nodes=1, depth=0, ties=len(moves), engine=self.name)


@register_agent("greedy")
class GreedyAgent(Agent):
    """Profondeur 1 : maximise sa propre part de territoire après son coup."""

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        p = state.current
        root = state.ply
        values: dict[int, float] = {}
        for m in moves:
            state.apply(m)
            values[m] = self.valuation.vector(state, root)[p]
            state.undo()
        best = max(values.values())
        ties = [m for m in moves if values[m] > best - EPS]
        move = self.pick(ties)
        self._details = DecisionDetails(scores=tuple((m, values[m]) for m in moves))
        return move, SearchStats(
            nodes=len(moves) + 1,
            leaves=len(moves),
            value=values[move],
            depth=1,
            ties=len(ties),
            engine=self.name,
            pv=(move,),
        )
