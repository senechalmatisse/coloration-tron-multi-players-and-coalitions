"""Environnement d'apprentissage, modèle linéaire et entraînement (P10)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tron.agents import make_agent
from tron.cli.main import main
from tron.core import ConfigError, GameConfig, GameState
from tron.learning import (
    FEATURE_NAMES,
    FeatureExtractor,
    IllegalActionError,
    LinearModel,
    ModelError,
    ObservationSpec,
    TronEnv,
    load_model,
)
from tron.learning.features import channel_names, relative_order
from tron.learning.train import TrainingSpec, evaluate_model, run_training, train
from tron.sim import Match

ROOT = Path(__file__).resolve().parents[1]
DUEL = ROOT / "configs" / "games" / "learning_duel_10x10.yaml"
_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _duel(grid: int = 7, **extra: Any) -> GameConfig:
    return GameConfig.from_dict(
        {
            "name": "tiny_duel",
            "grid": grid,
            "seed": 2,
            "placement": {"mode": "regular", "rotate": True},
            "players": [
                {"name": "learner", "team": "A", "agent": "greedy"},
                {"name": "opp", "team": "B", "agent": "greedy"},
            ],
            **extra,
        }
    )


def _first_legal(obs: dict[str, np.ndarray]) -> int:
    return int(np.flatnonzero(obs["legal"])[0])


def _episode(env: TronEnv, seed: int, policy: Any = _first_legal) -> list[Any]:
    trace: list[Any] = []
    obs = env.reset(seed=seed)
    trace.append({k: v.tolist() for k, v in obs.items()})
    while not env.done:
        obs, reward, term, trunc, info = env.step(policy(obs))
        trace.append(({k: v.tolist() for k, v in obs.items()}, reward, term, trunc, info))
    return trace


# ------------------------------------------------------------ observations
def test_observation_planes_are_relative_and_consistent() -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 6,
            "placement": {"positions": [[1, 1], [4, 4], [4, 1]]},
            "players": [
                {"name": "a", "team": "A", "profile": "team"},
                {"name": "b", "team": "B"},
                {"name": "c", "team": "A"},
            ],
        }
    )
    st = GameState(cfg)
    spec = ObservationSpec(distances=True, territory=True, preferences=True)
    ext = FeatureExtractor(cfg)
    obs = __import__("tron.learning.features", fromlist=["observe"]).observe(
        st, 1, spec, territory=ext.territory(st), preferences=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    names = channel_names(3, distances=True)
    assert obs["planes"].shape == (len(names), 8, 8) and names[3] == "walls_0"
    assert relative_order(3, 1) == [1, 2, 0]
    planes = dict(zip(names, obs["planes"], strict=True))
    assert planes["head_0"][5, 5] == 1.0  # tete de b (joueur 1) = joueur relatif 0
    assert planes["head_2"][2, 2] == 1.0  # tete de a (joueur 0) = joueur relatif 2
    assert planes["border"].sum() == 4 * 7 and planes["free"].sum() == 36 - 3
    assert planes["distance"][5, 5] == 1.0 and planes["distance"][5, 4] == 0.5
    assert obs["same_team"].tolist() == [1.0, 0.0, 0.0]  # b est seul dans B
    assert int(obs["current"]) == 2 and not obs["legal"].any()  # c'est a qui joue
    assert obs["territory"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="territoire"):
        __import__("tron.learning.features", fromlist=["observe"]).observe(st, 0, spec)


def test_reset_and_step_are_reproducible_with_a_seed() -> None:
    cfg = _duel()
    spec = ObservationSpec(territory=True)
    a = _episode(TronEnv(cfg, 0, shaping="territory", observation=spec), 11)
    b = _episode(TronEnv(cfg, 0, shaping="territory", observation=spec), 11)
    assert a == b and len(a) > 3
    c = _episode(TronEnv(cfg, 0, shaping="territory", observation=spec), 12)
    assert c != a
    env = TronEnv(cfg, 0)
    env.reset(seed=11)
    first = env.state.to_snapshot()
    env.reset(seed=11)
    assert env.state.to_snapshot() == first  # reset reconstruit les adversaires


# ---------------------------------------------------------- actions illégales
def _blocked_action(env: TronEnv) -> int:
    legal = env.legal_actions()
    return next(a for a in range(4) if a not in legal)


def test_illegal_action_rules() -> None:
    cfg = _duel()
    lose = TronEnv(cfg, 0, illegal="lose")
    lose.reset(seed=3)
    while len(lose.legal_actions()) == 4 or not lose.legal_actions():
        lose.step(lose.legal_actions()[0])
    _obs, reward, term, _trunc, info = lose.step(_blocked_action(lose))
    assert info["illegal"] and not info["learner_alive"] and term and reward == -1.0

    strict = TronEnv(cfg, 0, illegal="raise")
    strict.reset(seed=3)
    while len(strict.legal_actions()) == 4:
        strict.step(strict.legal_actions()[0])
    before = strict.state.to_snapshot()
    with pytest.raises(IllegalActionError):
        strict.step(_blocked_action(strict))
    assert strict.state.to_snapshot() == before  # rien n'a été joué

    picks = []
    for _ in range(2):
        conv = TronEnv(cfg, 0, illegal="random")
        conv.reset(seed=3)
        while len(conv.legal_actions()) == 4:
            conv.step(conv.legal_actions()[0])
        legal = conv.legal_actions()
        _o, _r, _t, _u, info = conv.step(_blocked_action(conv))
        assert info["illegal"] and info["played"] in legal and info["learner_alive"]
        picks.append(info["played"])
    assert picks[0] == picks[1]  # conversion déterministe pour une graine donnée

    for bad in (4, -1, True, 1.0, "N"):
        with pytest.raises(ValueError):
            strict.step(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        TronEnv(cfg, 0, illegal="ignore")


def test_no_legal_move_is_an_automatic_pass() -> None:
    cfg = _duel(grid=5)
    seen = False
    for seed in range(30):
        env = TronEnv(cfg, 0)
        obs = env.reset(seed=seed)
        while not env.done:
            assert obs["legal"].any()  # jamais de décision sans coup légal
            obs, _r, _t, _u, info = env.step(_first_legal(obs))
        if info["auto_pass"]:
            seen = True
            assert not info["learner_alive"] and info["winner"] == "B"
    assert seen


# -------------------------------------------------------------- récompenses
def test_terminal_reward_matches_official_result_and_draws() -> None:
    cfg = _duel()
    for seed in range(6):
        env = TronEnv(cfg, 0)
        obs = env.reset(seed=seed)
        reward = 0.0
        while not env.done:
            obs, reward, term, _u, info = env.step(_first_legal(obs))
        assert term
        expected = 1.0 if info["winner"] == "A" else -1.0
        assert reward == expected and info["end_reason"] == "elimination"
    whole = GameConfig.from_dict(
        {
            "grid": 6,
            "players": [{"name": "a", "team": "A"}, {"name": "b", "team": "B"}],
            "rules": {"mode": "zone_control", "zones": [[0, 0, 6, 6]], "max_rounds": 3},
        }
    )
    env = TronEnv(whole, 0)
    obs = env.reset(seed=1)
    while not env.done:
        obs, reward, term, _u, info = env.step(_first_legal(obs))
    assert term and info["winner"] == "" and reward == 0.0  # zone toujours disputée


def test_potential_shaping_telescopes_to_a_constant() -> None:
    cfg = _duel()
    env = TronEnv(cfg, 0, shaping="territory", shaping_scale=0.3)
    obs = env.reset(seed=5)
    phi0 = env.info["territory"]
    total = 0.0
    while not env.done:
        obs, reward, _term, _u, info = env.step(_first_legal(obs))
        total += reward
    final = 1.0 if info["winner"] == "A" else -1.0
    assert total == pytest.approx(final - 0.3 * phi0, abs=1e-9)


def test_learner_eliminated_while_team_plays_on_and_truncation() -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 8,
            "seed": 4,
            "players": [
                {"name": "l", "team": "A"},
                {"name": "x", "team": "B", "agent": "greedy"},
                {"name": "m", "team": "A", "agent": "greedy"},
            ],
        }
    )
    env = TronEnv(cfg, 0, illegal="lose")
    env.reset(seed=4)
    info = env.info
    while info["learner_alive"]:  # avance jusqu'à pouvoir se jeter dans un mur
        legal = env.legal_actions()
        action = _blocked_action(env) if len(legal) < 4 else legal[0]
        _obs, reward, term, _u, info = env.step(action)
    assert term and not info["learner_alive"]
    official = env.state.winning_team()
    assert reward == (1.0 if official == 0 else -1.0)  # partie jouée jusqu'au bout
    short = TronEnv(_duel(), 0, max_steps=2)
    obs = short.reset(seed=1)
    for _ in range(2):
        obs, _r, term, trunc, _i = short.step(_first_legal(obs))
    assert trunc and not term and short.done
    with pytest.raises(RuntimeError):
        short.step(0)


def test_environment_does_not_import_the_gui() -> None:
    code = (
        "import sys, tron.learning.env, tron.learning.train;"
        "print(any(m.startswith(('tron.viz', 'arcade')) for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env=_ENV
    )
    assert out.stdout.strip() == "False"


# -------------------------------------------------------------------- modèle
def test_model_export_reload_and_errors(tmp_path: Path) -> None:
    model = LinearModel(np.linspace(-1, 1, len(FEATURE_NAMES)), {"name": "m"})
    path = model.save(tmp_path / "m" / "model.json")
    back = load_model(path)
    assert np.array_equal(back.weights, model.weights) and back.metadata == {"name": "m"}
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in (
        ("format", "tron-model/0"),
        ("feature_set", "autre"),
        ("features", ["bias"]),
        ("weights", [0.0] * 3),
        ("weights", [float("nan")] * len(FEATURE_NAMES)),
        ("weights", [True] * len(FEATURE_NAMES)),
    ):
        broken = tmp_path / f"{key}.json"
        text = json.dumps({**data, key: value}).replace("NaN", "NaN")
        broken.write_text(text, encoding="utf-8")
        with pytest.raises(ModelError):
            load_model(broken)
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ModelError, match="JSON"):
        load_model(tmp_path / "bad.json")
    with pytest.raises(ModelError, match="introuvable"):
        load_model(tmp_path / "absent.json")
    # Rechargé et joué sans le code d'entraînement ni l'environnement.
    code = (
        "import sys, tron.agents;"
        "from tron.core import GameConfig; from tron.sim import Match;"
        f"cfg = GameConfig.from_dict({{'grid': 7, 'players': [{{'name': 'a', 'team': 'A',"
        f" 'agent': 'learned', 'model': {str(path)!r}}}, {{'name': 'b', 'team': 'B'}}]}});"
        "Match(cfg).run();"
        "print('tron.learning.train' in sys.modules, 'tron.learning.env' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env=_ENV
    )
    assert out.stdout.strip() == "False False"


def test_learned_agent_in_configs(tmp_path: Path) -> None:
    path = LinearModel(np.eye(len(FEATURE_NAMES))[1]).save(tmp_path / "model.json")
    cfg = GameConfig.from_dict(
        {
            "grid": 8,
            "players": [
                {"name": "a", "team": "A", "agent": "learned", "model": str(path)},
                {"name": "b", "team": "B", "agent": "greedy"},
            ],
        }
    )
    result = Match(cfg).run()
    assert result.n_moves > 0
    agent = make_agent("learned", 0, cfg, model=str(path))
    decision = agent.choose_move(GameState(cfg))
    assert decision.details is not None and decision.details.note == "politique linéaire apprise"
    for bad in ({"epsilon": 2.0}, {"epsilon": True}, {"model": str(tmp_path / "x.json")}):
        params = {"model": str(path), **bad}
        with pytest.raises(ConfigError):
            make_agent("learned", 0, cfg, **params)


# ------------------------------------------------------------- entraînement
def _tiny_spec(**over: Any) -> TrainingSpec:
    data: dict[str, Any] = {
        "name": "tiny",
        "game": _duel().to_dict(),
        "seed": 3,
        "episodes": 6,
        "generations": 2,
        "learning": {"alpha": 0.05, "epsilon": {"start": 0.3, "end": 0.0}},
        "env": {"shaping": "territory", "max_steps": 40},
        "opponents": ["greedy", {"agent": "random"}],
        "self_play": {"enabled": True, "share": 0.5},
        "evaluation": {"games": 2, "references": ["greedy", "paranoid"], "depth": 1},
        **over,
    }
    return TrainingSpec.from_dict(data)


def test_training_is_deterministic_and_seeds_are_disjoint() -> None:
    spec = _tiny_spec()
    a = train(spec)
    b = train(spec)
    assert np.array_equal(a.model.weights, b.model.weights) and a.episodes == b.episodes
    assert np.isfinite(a.model.weights).all() and len(a.episodes) == 12
    assert {r["generation"] for r in a.episodes} == {0, 1}
    assert any("learned" in r["opponents"] for r in a.episodes[6:])  # auto-jeu
    assert a.model.metadata["episodes"] == 12 and a.model.metadata["self_play"]
    rows = evaluate_model(
        a.model, spec.game, references=("greedy",), games=2, depth=1, exclude_seeds=a.train_seeds
    )
    assert [r["learner"] for r in rows] == ["learned", "greedy"]
    assert all(r["wins"] + r["draws"] + r["losses"] == 2 for r in rows)
    from tron.learning.train import eval_seeds

    with pytest.raises(ValueError, match="disjointes"):
        evaluate_model(a.model, spec.game, games=2, exclude_seeds=eval_seeds(0, 2))


def test_run_training_writes_model_and_reports(tmp_path: Path) -> None:
    spec = _tiny_spec(generations=1)
    result = run_training(spec, tmp_path / "out")
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["evaluation.csv", "evaluation.md", "model.json", "training.csv"]
    assert len(result.evaluation) == 4 and len(result.eval_seeds) == 2
    report = (tmp_path / "out" / "evaluation.md").read_text(encoding="utf-8")
    assert "disjoints (vérifié)" in report and "| tiny_duel | paranoid |" in report
    assert np.array_equal(load_model(tmp_path / "out" / "model.json").weights, result.model.weights)


@pytest.mark.parametrize(
    "over",
    [
        {"unknown": 1},
        {"episodes": 0},
        {"learning": {"alpha": -1}},
        {"learning": {"gamma": 1.5}},
        {"learning": {"epsilon": {"start": 2}}},
        {"env": {"shaping": "score"}},
        {"env": {"max_steps": 0}},
        {"opponents": [{"depth": 2}]},
        {"opponents": ["inconnu"]},
        {"opponents": "greedy"},
        {"self_play": {"enabled": "oui"}},
        {"evaluation": {"references": ["inconnu"]}},
        {"evaluation": {"games": -1}},
        {"player": 5},
    ],
)
def test_training_spec_validation(over: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        _tiny_spec(**over)
    with pytest.raises(ConfigError, match="game"):
        TrainingSpec.from_dict({"name": "x"})


def test_shipped_training_config_loads() -> None:
    spec = TrainingSpec.from_yaml(ROOT / "configs" / "training" / "linear_duel.yaml")
    assert spec.game.name == "learning_duel" and spec.total_episodes == 300
    assert [cfg.name for _, cfg in spec.eval_configs] == ["mcts_duel"]


def test_cli_train_and_evaluate_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml

    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "cli",
                "game": str(DUEL),
                "episodes": 3,
                "evaluation": {"games": 1, "references": ["greedy"], "depth": 1},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "model"
    assert main(["train", str(config), "--out", str(out), "--episodes", "2", "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "2 épisodes" in printed and "| learning_duel | greedy |" in printed
    code = main(
        [
            "evaluate-model",
            str(out / "model.json"),
            str(DUEL),
            "--games",
            "1",
            "--references",
            "greedy,random",
            "--depth",
            "1",
            "--out",
            str(tmp_path / "ev"),
        ]
    )
    assert code == 0 and (tmp_path / "ev" / "evaluation.csv").exists()
    assert "| learning_duel | random |" in capsys.readouterr().out
    assert main(["evaluate-model", str(tmp_path / "absent.json"), str(DUEL)]) == 2


# ------------------------------------------------------- adaptateur Gymnasium
def test_gymnasium_adapter_with_a_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    class Space:
        def __init__(self, *args: Any, **kw: Any) -> None:
            self.args = args

    class Env:
        def reset(self, *, seed: int | None = None, options: Any = None) -> None:
            self.np_random_seed = seed

    gym = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")
    for name in ("Box", "Discrete", "MultiBinary", "Dict"):
        setattr(spaces, name, type(name, (Space,), {}))
    gym.Env = Env  # type: ignore[attr-defined]
    gym.spaces = spaces  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces)
    from tron.learning.gym import make_gym_env

    env = make_gym_env(_duel(), 0, observation=ObservationSpec(territory=True))
    assert set(env.observation_space.args[0]) == {
        "planes",
        "alive",
        "same_team",
        "current",
        "legal",
        "territory",
    }
    obs, info = env.reset(seed=2)
    assert info["seed"] == 2 and obs["legal"].any()
    obs, reward, _term, trunc, info = env.step(np.int64(_first_legal(obs)))
    assert isinstance(reward, float) and not trunc
