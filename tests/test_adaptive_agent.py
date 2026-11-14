"""Tests de l'agent adaptatif."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_config
from tron.agents import (
    AGENTS,
    AdaptiveAgent,
    DecisionContext,
    DiplomacyIntent,
    DiplomacySnapshot,
    adapted_weights,
    base_weights,
    make_agent,
)
from tron.agents.adaptive import TARGET_WEIGHTS
from tron.core import ConfigError, GameConfig, GameState
from tron.eval import COMPONENTS, DEFAULT_WEIGHTS
from tron.experiments import Plan, run_plan
from tron.experiments.store import ResultStore
from tron.sim import Match

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _duel(
    agent: str = "adaptive", params: dict | None = None, opponent: str = "greedy", seed: int = 3
):
    return make_config(
        10,
        10,
        2,
        agents=[agent, opponent],
        depth=[2, 1],
        params=[params or {}, {}],
        seed=seed,
        placement={"mode": "regular", "rotate": True},
    )


def test_registry_and_validation() -> None:
    assert "adaptive" in AGENTS
    cfg = _duel()
    agent = make_agent("adaptive", 0, cfg, depth=2)
    assert isinstance(agent, AdaptiveAgent) and agent.describe().startswith("adaptive[social")
    for bad in (
        {"adaptation_rate": 1.5},
        {"adaptation_rate": -0.1},
        {"adaptation_rate": "fort"},
        {"observation_window": 0},
        {"observation_window": 2.5},
        {"fallback_agent": "mcts"},
        {"min_confidence": 2},
        {"cooperation_threshold": -1},
        {"inconnu": 3},
    ):
        with pytest.raises(ConfigError):
            make_agent("adaptive", 0, cfg, **bad)


def test_zero_rate_reproduces_the_fallback_agent_exactly() -> None:
    for fallback, params in (
        ("social", {"profile": "team"}),
        ("maxn", {}),
        ("paranoid", {}),
        ("greedy", {}),
    ):
        ref = Match(_duel(fallback, params))
        ref.run()
        adaptive_params = {"adaptation_rate": 0.0, "fallback_agent": fallback}
        if fallback == "social":
            adaptive_params["fallback_profile"] = "team"
        ada = Match(_duel("adaptive", adaptive_params))
        ada.run()
        assert [(e.player, e.direction) for e in ada.events] == [
            (e.player, e.direction) for e in ref.events
        ], fallback
        agent = ada.agents[0]
        assert isinstance(agent, AdaptiveAgent)
        assert agent.weights_used == agent.base_weights


def test_adapted_weights_are_bounded_and_normalised() -> None:
    base = dict(DEFAULT_WEIGHTS)
    probs = [
        {
            "aggressive": 0.9,
            "territorial": 0.05,
            "survival": 0.0,
            "cooperative": 0.0,
            "random_like": 0.05,
        }
    ]
    w = adapted_weights(base, probs, rate=0.25, min_confidence=0.35)
    assert sum(w.values()) == pytest.approx(1.0)
    for c in COMPONENTS:
        lo = min(base[c], *(t[c] for t in TARGET_WEIGHTS.values()))
        hi = max(base[c], *(t[c] for t in TARGET_WEIGHTS.values()))
        assert lo - 1e-12 <= w[c] <= hi + 1e-12
        spread = max(abs(t[c] - base[c]) for t in TARGET_WEIGHTS.values())
        assert abs(w[c] - base[c]) <= 0.25 * spread + 1e-12  # ajustement borné par λ
    assert w["safety"] > base["safety"]  # plus de marge contre un adversaire agressif
    assert adapted_weights(base, probs, rate=0.0, min_confidence=0.35) == base
    unsure = [
        {k: 0.2 for k in ("aggressive", "territorial", "survival", "cooperative", "random_like")}
    ]
    assert adapted_weights(base, unsure, rate=1.0, min_confidence=0.35) == base
    assert adapted_weights(base, [], rate=1.0, min_confidence=0.0) == base


def test_agent_adapts_and_resets_between_games() -> None:
    cfg = _duel(params={"adaptation_rate": 0.5}, opponent="greedy")
    match = Match(cfg)
    match.run()
    agent = match.agents[0]
    assert isinstance(agent, AdaptiveAgent)
    assert agent.model.most_likely(1) == "territorial"
    assert agent.weights_used != agent.base_weights
    assert agent.weights_used["bottleneck"] > agent.base_weights["bottleneck"]
    second = Match(cfg, agents=match.agents)  # Match réinitialise les agents
    assert agent.model.history(1) == () and agent.weights_used == agent.base_weights
    assert not agent.adaptive_heuristic.adapted
    second.run()
    assert agent.model.history(1)  # le modèle réapprend sur la nouvelle partie


def test_base_weights_follow_the_configured_heuristic() -> None:
    voronoi = _duel()
    assert base_weights(voronoi) == {c: float(c == "territory") for c in COMPONENTS}
    comp = GameConfig.from_dict(
        {**voronoi.to_dict(), "heuristic": {"name": "composite", "safety": 0.5, "territory": 0.5}}
    )
    w = base_weights(comp)
    assert sum(w.values()) == pytest.approx(1.0) and w["safety"] > w["mobility"]


def test_cooperative_opponent_gets_an_alliance_proposal() -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 10,
            "diplomacy": {"enabled": True},
            "players": [
                {"name": "a", "team": "A", "agent": "adaptive", "depth": 1},
                {"name": "b", "team": "B", "agent": "greedy"},
                {"name": "c", "team": "C", "agent": "greedy"},
            ],
        }
    )
    agent = make_agent("adaptive", 0, cfg, depth=1)
    assert isinstance(agent, AdaptiveAgent)
    st = GameState(cfg)
    n = 3
    zeros = tuple(tuple(0 for _ in range(n)) for _ in range(n))
    snap = DiplomacySnapshot(
        0, (), (), tuple(tuple(1.0 for _ in range(n)) for _ in range(n)), zeros
    )

    def fake(q: int) -> dict[str, float]:
        coop = 0.8 if q == 2 else 0.1
        rest = (1 - coop) / 4
        return {
            "aggressive": rest,
            "territorial": rest,
            "survival": rest,
            "cooperative": coop,
            "random_like": rest,
        }

    agent.model.probabilities = fake  # type: ignore[method-assign]
    dec = agent.choose_move(st, 0, DecisionContext(0, snap))
    assert dec.diplomacy == DiplomacyIntent("propose", 2)
    agent.diplomatic = False
    assert agent.choose_move(st, 0, DecisionContext(0, snap)).diplomacy is None


def test_example_config_and_process_order_independence(tmp_path: Path) -> None:
    cfg = GameConfig.from_yaml(CONFIGS / "games" / "adaptive_example.yaml")
    assert cfg.players[0].agent == "adaptive" and cfg.players[0].params["adaptation_rate"] == 0.25
    plan = Plan.from_dict(
        {
            "name": "adapt",
            "scenario": "solo_vs_coalition",
            "games": 2,
            "seed": 9,
            "base": {"solo_agent": "adaptive", "adaptation_rate": 0.5, "coalition_mode": "greedy"},
            "factors": {"k": [1, 2], "grid": [8]},
        }
    )
    assert plan.conditions()[0].config["players"][0]["adaptation_rate"] == 0.5
    run_plan(plan, tmp_path / "seq", workers=1, progress=False)
    run_plan(plan, tmp_path / "par", workers=2, progress=False)
    keys = ("condition", "game", "seed", "winning_team", "n_moves", "rank_p0", "nodes_p0")

    def rows(path: Path) -> list[tuple[str, ...]]:
        return sorted(tuple(r[k] for k in keys) for r in ResultStore(path / "results.csv").rows())

    assert rows(tmp_path / "seq") == rows(tmp_path / "par")
