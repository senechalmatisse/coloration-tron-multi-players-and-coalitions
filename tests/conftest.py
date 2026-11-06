"""Outils communs aux tests."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pytest

from tron.core import GameConfig, GameState


def make_config(
    width: int = 10,
    height: int = 10,
    n: int = 2,
    *,
    teams: Sequence[str] | None = None,
    agents: Sequence[str] | str = "greedy",
    depth: int | Sequence[int] = 1,
    params: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    seed: int = 0,
    first_player: int = 0,
    placement: dict[str, Any] | str = "regular",
    heuristic: dict[str, Any] | str = "voronoi",
) -> GameConfig:
    """Configuration de test compacte."""
    players = []
    for i in range(n):
        agent = agents if isinstance(agents, str) else agents[i]
        d = depth if isinstance(depth, int) else depth[i]
        p: dict[str, Any] = {"name": f"J{i}", "agent": agent, "depth": d}
        if teams is not None:
            p["team"] = teams[i]
        if params is not None:
            p.update(params if isinstance(params, dict) else params[i])
        players.append(p)
    return GameConfig.from_dict(
        {
            "grid": [width, height],
            "players": players,
            "seed": seed,
            "first_player": first_player,
            "placement": placement,
            "heuristic": heuristic,
        }
    )


def random_playout(
    state: GameState, rng: random.Random, max_plies: int = 10_000, suicide_rate: float = 0.0
) -> int:
    """Joue des coups aléatoires jusqu'à la fin ; renvoie le nombre de plies joués."""
    plies = 0
    while not state.is_terminal() and plies < max_plies:
        moves = state.legal_moves()
        if moves and rng.random() >= suicide_rate:
            state.apply(rng.choice(moves))
        else:
            state.apply(rng.randrange(4) if rng.random() < 0.5 else -1)
        plies += 1
    return plies


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)
