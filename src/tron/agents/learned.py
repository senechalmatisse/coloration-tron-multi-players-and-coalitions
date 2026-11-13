"""Agent joué par un modèle appris (P10).

Le modèle est un fichier JSON ``tron-model/1`` (voir :mod:`tron.learning.model`) :
le charger ne demande que NumPy, jamais le code d'entraînement. Dans une
configuration de partie :

.. code-block:: yaml

    players:
      - {name: appris, team: A, agent: learned, model: models/linear_duel/model.json}

Le chemin est relatif au dossier courant. ``epsilon`` (défaut 0) joue un coup
légal au hasard avec cette probabilité (exploration, auto-jeu).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tron.agents.base import Agent, DecisionDetails, SearchStats, register_agent
from tron.agents.search import EPS
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState
from tron.learning.features import FeatureExtractor
from tron.learning.model import LinearModel, ModelError, load_model


@register_agent("learned")
class LearnedAgent(Agent):
    """Choisit le coup dont l'état après coup a la plus grande valeur selon le modèle."""

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        model: str | Path | Mapping[str, Any] | LinearModel,
        epsilon: float = 0.0,
        **kw: Any,
    ) -> None:
        super().__init__(player, config, **kw)
        if isinstance(epsilon, bool) or not isinstance(epsilon, int | float):
            raise ConfigError(f"epsilon doit être un nombre : {epsilon!r}")
        if not (math.isfinite(epsilon) and 0.0 <= epsilon <= 1.0):
            raise ConfigError(f"epsilon doit être dans [0, 1] : {epsilon}")
        try:
            self.model = load_model(model)
        except ModelError as exc:
            raise ConfigError(str(exc)) from exc
        self.epsilon = float(epsilon)
        self.extractor = FeatureExtractor(config)

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        if self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon:
            move = moves[int(self.rng.integers(len(moves)))]
            self._details = DecisionDetails(
                scores=tuple((m, None) for m in moves), note="exploration (epsilon)"
            )
            return move, SearchStats(nodes=1, depth=0, ties=len(moves), engine=self.name)
        values = self.model.action_values(state, self.player, self.extractor, moves)
        best = max(values.values())
        ties = [m for m in moves if values[m] > best - EPS]
        move = self.pick(ties)
        self._details = DecisionDetails(
            scores=tuple((m, values[m]) for m in moves), note="politique linéaire apprise"
        )
        return move, SearchStats(
            nodes=len(moves) + 1,
            leaves=len(moves),
            value=values[move],
            depth=1,
            ties=len(ties),
            engine=self.name,
            pv=(move,),
        )

    def describe(self) -> str:
        name = self.model.metadata.get("name", "?")
        return f"{self.name}({name})"


__all__ = ["LearnedAgent"]
