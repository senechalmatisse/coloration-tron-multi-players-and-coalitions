"""Tests du modèle d'adversaire."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from conftest import make_config
from tron.agents import Agent, SearchStats, make_agent
from tron.agents.opponents import (
    BEHAVIOURS,
    PROFILES,
    OpponentModel,
    alignment,
    criteria,
    partners_of,
)
from tron.core import GameConfig, GameState
from tron.eval import CompositeHeuristic
from tron.sim import Match


class Synthetic(Agent):
    """Adversaire synthétique : maximise toujours un même critère du modèle."""

    def __init__(self, player: int, config: GameConfig, criterion: str, **kw: Any) -> None:
        super().__init__(player, config, **kw)
        self.criterion = criterion
        self.heur = CompositeHeuristic(config.width, config.height, config.n_players)

    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        values = criteria(state, state.current, moves, self.heur)[self.criterion]
        assert values is not None
        best = max(values)
        return self.pick([m for m, v in zip(moves, values, strict=True) if v >= best - 1e-12]), (
            SearchStats(nodes=1, engine=self.criterion)
        )


class Feed:
    """Observateur qui transmet chaque coup au modèle."""

    def __init__(self, model: OpponentModel) -> None:
        self.model = model

    def on_start(self, match: Match) -> None:
        pass

    def on_event(self, match: Match, e: Any) -> None:
        self.model.observe(e.player, e.direction, e.origin, e.target, e.eliminated, match.state)

    def on_end(self, match: Match, result: Any) -> None:
        pass


def _play(kind: str, seed: int, model_window: int = 20) -> OpponentModel:
    cfg = make_config(
        12, 12, 2, seed=seed, first_player=seed % 2, placement={"mode": "regular", "rotate": True}
    )
    a0 = make_agent("random", 0, cfg) if kind == "random" else Synthetic(0, cfg, kind)
    model = OpponentModel(cfg, 1, window=model_window)
    Match(cfg, agents=[a0, make_agent("greedy", 1, cfg)], observers=[Feed(model)]).run()
    return model


def test_alignment_and_criteria() -> None:
    assert alignment([1.0, 3.0, 2.0], 1) == 1.0 and alignment([1.0, 3.0, 2.0], 2) == 0.5
    assert alignment([2.0, 2.0], 0) is None and alignment([5.0], 0) is None
    assert alignment(None, 0) is None
    cfg = make_config(8, 8, 3, teams=["A", "A", "B"])
    st = GameState(cfg)
    heur = CompositeHeuristic(8, 8, 3)
    before = st.key, st.ply, st.grid.tobytes()
    values = criteria(st, 0, st.legal_moves(), heur, partners=(1,))
    assert (st.key, st.ply, st.grid.tobytes()) == before
    assert set(values) == set(BEHAVIOURS)
    assert all(v is not None and len(v) == len(st.legal_moves()) for v in values.values())
    assert criteria(st, 0, st.legal_moves(), heur)["cooperative"] is None
    with pytest.raises(ValueError):
        criteria(st, 1, st.legal_moves(), heur)
    assert partners_of(cfg, 0) == (1,) and partners_of(cfg, 2) == ()


def test_probabilities_are_valid_and_reset_clears() -> None:
    model = _play("territorial", 1)
    probs = model.probabilities(0)
    assert set(probs) == set(PROFILES)
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)
    hist = model.history(0)
    assert 0 < len(hist) <= 20 and all(len(f.alignment) == len(BEHAVIOURS) for f in hist)
    assert model.confidence(0) == max(probs.values())
    model.reset()
    assert model.history(0) == ()
    empty = model.probabilities(0)
    assert empty["random_like"] == max(empty.values())  # sans données : profil indifférent


def test_identical_observations_give_identical_probabilities() -> None:
    a = _play("survival", 2)
    b = _play("survival", 2)
    assert a.probabilities(0) == b.probabilities(0)
    assert a.history(0) == b.history(0)


def test_invalid_players_and_unobserved_moves() -> None:
    cfg = make_config(8, 8, 3, teams=["A", "A", "B"])
    model = OpponentModel(cfg, 0)
    assert model.opponents == (2,)
    st = GameState(cfg)
    origin = st.head(0)
    st.apply(st.legal_moves()[0])
    with pytest.raises(ValueError):
        model.observe(7, 0, origin, (0, 0), False, st)
    with pytest.raises(ValueError):
        model.probabilities(-1)
    # Le coéquipier de l'observateur n'est pas modélisé.
    assert model.observe(0, 0, origin, st.head(0), False, st) is None
    for bad in ({"window": 0}, {"temperature": 0.0}, {"prior": -1.0}):
        with pytest.raises(ValueError):
            OpponentModel(cfg, 0, **bad)


def test_synthetic_opponents_are_classified_above_chance() -> None:
    """Test binomial unilatéral : précision du profil le plus probable contre 1/5."""
    truth = {
        "aggressive": "aggressive",
        "territorial": "territorial",
        "survival": "survival",
        "random": "random_like",
    }
    trials = 0
    correct = 0
    for kind, expected in truth.items():
        for seed in range(3):
            trials += 1
            correct += _play(kind, seed).most_likely(0) == expected
    p0 = 1 / len(PROFILES)
    p_value = sum(
        math.comb(trials, k) * p0**k * (1 - p0) ** (trials - k) for k in range(correct, trials + 1)
    )
    assert p_value < 1e-4, (correct, trials)
    assert correct / trials >= 0.75


def test_state_is_never_modified_by_observation() -> None:
    cfg = make_config(10, 10, 2, seed=4, placement={"mode": "regular", "rotate": True})
    st = GameState(cfg)
    model = OpponentModel(cfg, 1)
    rng = random.Random(0)
    for _ in range(8):
        if st.current == 0:
            origin = st.head(0)
            move = rng.choice(st.legal_moves())
            st.apply(move)
            snap = (st.grid.tobytes(), st.key, st.ply, len(st._hist), st.current)
            assert model.observe(0, move, origin, st.head(0), False, st) is not None
            assert (st.grid.tobytes(), st.key, st.ply, len(st._hist), st.current) == snap
        else:
            st.apply(rng.choice(st.legal_moves()))
