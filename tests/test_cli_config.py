"""Outils de configuration et documentation exécutable (P15)."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
import yaml

from tron.agents import AGENTS
from tron.cli.config_tools import TEMPLATES, check, detect_kind, normalized
from tron.cli.main import build_parser, main
from tron.core import GameConfig
from tron.eval import HEURISTICS
from tron.experiments.plan import Plan
from tron.learning.train import TrainingSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((ROOT / "configs").rglob("*.yaml"))
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_shipped_yaml_loads_and_validates(path: Path) -> None:
    checked = check(path)
    expected = {"games": "game", "plans": "plan", "training": "training"}[path.parent.name]
    assert checked.kind == expected
    loader = {
        "game": GameConfig.from_yaml,
        "plan": Plan.from_yaml,
        "training": TrainingSpec.from_yaml,
    }
    assert loader[expected](path) is not None


def test_cli_validate_reports_each_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = ROOT / "configs" / "games" / "duel_20x20.yaml"
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "grid: 9\nplayers:\n  - {name: a, agent: paranoid, depth: 0}\n  - {name: b}\n",
        encoding="utf-8",
    )
    broken = tmp_path / "broken.yaml"
    broken.write_text("grid: [1, 2\n", encoding="utf-8")
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    assert main(["config", "validate", str(good)]) == 0
    assert "valide : " in capsys.readouterr().out
    assert main(["config", "validate", str(good), str(bad), str(broken), str(listing)]) == 1
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert sum(ln.startswith("invalide : ") for ln in lines) == 3
    assert "profondeur" in out and "YAML invalide" in out
    assert "dictionnaire" in out
    assert main(["config", "validate", str(tmp_path / "absent.yaml")]) == 1
    assert "introuvable" in capsys.readouterr().out
    assert main(["config", "validate", "--kind", "plan", str(good)]) == 1  # type imposé


@pytest.mark.parametrize("kind", sorted(TEMPLATES))
def test_templates_are_valid_and_never_overwrite(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{kind}.yaml"
    assert main(["config", "template", kind, "--out", str(out)]) == 0
    expected = {"campaign": "plan", "training": "training"}.get(kind, "game")
    assert check(out).kind == expected
    assert detect_kind(yaml.safe_load(out.read_text(encoding="utf-8"))) == expected
    assert main(["config", "template", kind, "--out", str(out)]) == 2  # existe déjà
    assert main(["config", "template", kind, "--out", str(out), "--force"]) == 0
    capsys.readouterr()
    assert main(["config", "template", kind]) == 0
    assert capsys.readouterr().out == TEMPLATES[kind]
    assert TEMPLATES[kind].isascii()  # convention des YAML du dépôt : sans accents


def test_show_prints_a_normalized_equivalent_config(capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("games/zone_control_16x16.yaml", "plans/pilote.yaml"):
        path = ROOT / "configs" / name
        assert main(["config", "show", str(path)]) == 0
        data = yaml.safe_load(capsys.readouterr().out)
        if name.startswith("games"):
            assert GameConfig.from_dict(data) == GameConfig.from_yaml(path)
        else:
            assert Plan.from_dict(data).to_dict() == Plan.from_yaml(path).to_dict()
    training = ROOT / "configs" / "training" / "linear_duel.yaml"
    assert main(["config", "show", "--format", "json", str(training)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert TrainingSpec.from_dict(data).total_episodes == 300
    assert normalized(check(training))["evaluation"]["games"] == 20


def test_agent_and_heuristic_catalogs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agents", "list", "--format", "json"]) == 0
    agents = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert set(agents) == set(AGENTS)
    assert "simulations" in {p["name"] for p in agents["mcts"]["parameters"]}
    assert {"depth", "seed", "heuristic"} <= set(agents["mcts"]["common"])
    assert main(["agents", "list"]) == 0
    assert "paramètres communs" in capsys.readouterr().out
    assert main(["heuristics", "list", "--format", "json"]) == 0
    heurs = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert set(heurs) == set(HEURISTICS)
    assert {"territory", "safety"} <= {p["name"] for p in heurs["composite"]["parameters"]}
    assert main(["heuristics", "list"]) == 0
    assert "voronoi" in capsys.readouterr().out


def _documented_commands() -> list[tuple[str, str]]:
    commands = []
    for doc in DOCS:
        in_block = False
        for line in doc.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if not in_block:
                continue
            for part in re.split(r"&&|;|\|", line.split(" # ")[0]):
                text = part.strip()
                if text.startswith("uv run "):
                    text = text[len("uv run ") :]
                if re.match(r"^(NUMBA_DISABLE_JIT=1 )?tron( |$)", text) and "<" not in text:
                    commands.append((doc.name, text.replace("NUMBA_DISABLE_JIT=1 ", "")))
    return commands


def test_documented_commands_parse() -> None:
    """Chaque commande ``tron`` des blocs de code du README et de docs/ est acceptée par la CLI."""
    commands = _documented_commands()
    assert len(commands) > 40
    parser = build_parser()
    for doc, text in commands:
        args = shlex.split(text)[1:]
        if args in (["--version"], []):
            continue
        try:
            parser.parse_args(args)
        except SystemExit as exc:  # pragma: no cover - message de diagnostic
            raise AssertionError(f"{doc} : commande refusée : {text}") from exc
