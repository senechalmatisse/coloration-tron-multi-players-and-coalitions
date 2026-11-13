"""Extensions externes par entry points (P12)."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import shutil
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from tron.agents import AGENTS, GreedyAgent, make_agent, register_agent
from tron.cli.main import main
from tron.core import ConfigError, GameConfig, GameState
from tron.eval import HEURISTICS, make_heuristic
from tron.plugins import (
    AGENT_GROUP,
    DISABLE_ENV,
    HEURISTIC_GROUP,
    PluginWarning,
    load_plugins,
    plugin_report,
)
from tron.sim import Match

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "plugin_exemple"

FAKE_AGENTS = """
from tron.agents import Agent, SearchStats, register_agent

LOADS = 0
LOADS += 1


class FakeAgent(Agent):
    def _decide(self, state, moves):
        return moves[0], SearchStats(nodes=1, engine=self.name)


def helper():
    return 1


@register_agent("greedy")  # tente de remplacer un agent intégré
class Evil(Agent):
    def _decide(self, state, moves):
        return moves[-1], SearchStats(nodes=1, engine=self.name)
"""


def _dist(root: Path, name: str, entry_points: str) -> None:
    info = root / f"{name.replace('-', '_')}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(entry_points, encoding="utf-8")


def _reload_all() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PluginWarning)
        load_plugins(AGENT_GROUP, force=True)
        load_plugins(HEURISTIC_GROUP, force=True)


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Paquets d'extension « installés » : dossiers .dist-info sur sys.path."""
    pkg = tmp_path / "fake_tron_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "agents.py").write_text(FAKE_AGENTS, encoding="utf-8")
    shutil.copytree(EXAMPLE / "src" / "tron_plugin_exemple", tmp_path / "tron_plugin_exemple")
    _dist(
        tmp_path,
        "fake-tron-plugin",
        "[tron.agents]\n"
        "fake_agent = fake_tron_plugin.agents:FakeAgent\n"
        "broken = fake_tron_plugin.missing:Nope\n"
        "greedy = fake_tron_plugin.agents:FakeAgent\n"
        "not_an_agent = fake_tron_plugin.agents:helper\n"
        "dup = fake_tron_plugin.agents:FakeAgent\n"
        "evil = fake_tron_plugin.agents:Evil\n",
    )
    _dist(
        tmp_path,
        "zz-other-plugin",
        "[tron.agents]\ndup = tron_plugin_exemple:PrudentAgent\n",
    )
    _dist(
        tmp_path,
        "tron-plugin-exemple",
        "[tron.agents]\nprudent = tron_plugin_exemple:PrudentAgent\n"
        "[tron.heuristics]\ncentre = tron_plugin_exemple:CenterHeuristic\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    monkeypatch.undo()
    for name in [
        m for m in sys.modules if m.startswith(("fake_tron_plugin", "tron_plugin_exemple"))
    ]:
        del sys.modules[name]
    importlib.invalidate_caches()
    _reload_all()


def test_installed_plugins_are_loaded_with_clear_statuses(installed: Path) -> None:
    with pytest.warns(PluginWarning) as caught:
        infos = load_plugins(AGENT_GROUP, force=True)
    status = {(i.name, i.distribution): i.status for i in infos}
    assert status[("fake_agent", "fake-tron-plugin 0.1")] == "loaded"
    assert status[("prudent", "tron-plugin-exemple 0.1")] == "loaded"
    assert status[("broken", "fake-tron-plugin 0.1")] == "error"
    assert status[("greedy", "fake-tron-plugin 0.1")] == "conflict"  # nom intégré
    assert status[("not_an_agent", "fake-tron-plugin 0.1")] == "invalid"
    assert status[("evil", "fake-tron-plugin 0.1")] == "conflict"  # redéfinit greedy
    assert status[("dup", "fake-tron-plugin 0.1")] == "loaded"  # premier dans l'ordre
    assert status[("dup", "zz-other-plugin 0.1")] == "conflict"
    assert AGENTS["greedy"] is GreedyAgent and "evil" not in AGENTS
    broken = next(i for i in infos if i.name == "broken")
    assert "ModuleNotFoundError" in broken.detail
    assert len(caught) == sum(not i.loaded for i in infos)
    cfg = GameConfig.from_dict(
        {
            "grid": 9,
            "players": [
                {"name": "a", "team": "A", "agent": "fake_agent"},
                {"name": "b", "team": "B", "agent": "prudent", "bonus_centre": 0.2},
            ],
        }
    )
    assert Match(cfg).run().n_moves > 0
    assert make_agent("fake_agent", 0, cfg).describe().startswith("fake_agent")
    with pytest.raises(ConfigError, match="bonus_centre"):
        make_agent("prudent", 1, cfg, bonus_centre=-1)
    with pytest.raises(ConfigError, match="paramètres invalides"):
        make_agent("fake_agent", 0, cfg, inconnu=3)


def test_loading_is_idempotent(installed: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PluginWarning)
        first = load_plugins(AGENT_GROUP, force=True)
    again = load_plugins(AGENT_GROUP)
    assert again is first  # rien n'est réimporté
    assert sys.modules["fake_tron_plugin.agents"].LOADS == 1
    importlib.reload(sys.modules["tron.agents"])  # réimport du paquet : pas de doublon
    assert load_plugins(AGENT_GROUP) is first and AGENTS["fake_agent"].name == "fake_agent"


def test_heuristic_plugins_follow_the_shares_contract(installed: Path) -> None:
    _reload_all()
    infos = load_plugins(HEURISTIC_GROUP)
    assert [(i.name, i.status) for i in infos] == [("centre", "loaded")]
    cfg = GameConfig.from_dict(
        {"grid": 9, "players": [{"name": "a", "team": "A"}, {"name": "b", "team": "B"}]}
    )
    heur = make_heuristic("centre", 9, 9, 2)
    shares = heur.shares(GameState(cfg))
    assert sum(shares) == pytest.approx(1.0) and all(s > 0 for s in shares)
    assert "centre" in HEURISTICS
    game = GameConfig.from_yaml(EXAMPLE / "partie.yaml")
    assert Match(game).run().n_moves > 0


def test_cli_lists_plugins_and_disable_switch(
    installed: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _reload_all()
    assert main(["plugins", "list"]) == 1  # des extensions ont échoué
    out = capsys.readouterr().out
    assert "fake_agent" in out and "fake-tron-plugin 0.1" in out and "error" in out
    assert "fake_tron_plugin.agents:FakeAgent" in out and "centre" in out
    assert main(["plugins", "list", "--all"]) == 1
    assert "tron (intégré)" in capsys.readouterr().out
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert load_plugins(AGENT_GROUP, force=True) == [] and "fake_agent" not in AGENTS
    assert load_plugins(HEURISTIC_GROUP, force=True) == []
    assert main(["plugins", "list"]) == 0
    assert "aucune extension" in capsys.readouterr().out
    assert plugin_report() == []


def test_module_paths_in_configs_are_never_imported(installed: Path) -> None:
    cfg = GameConfig.from_dict(
        {
            "grid": 9,
            "players": [
                {"name": "a", "team": "A", "agent": "fake_tron_plugin.agents:FakeAgent"},
                {"name": "b", "team": "B"},
            ],
        }
    )
    sys.modules.pop("fake_tron_plugin.agents", None)
    with pytest.raises(ConfigError, match=r"entry point tron\.agents"):
        Match(cfg)
    assert "fake_tron_plugin.agents" not in sys.modules  # rien n'a été importé
    with pytest.raises(ValueError, match=r"entry point tron\.heuristics"):
        make_heuristic("pkg.mod:Heur", 9, 9, 2)


def test_historical_registry_still_works() -> None:
    @register_agent("test_registry_agent")
    class Local(GreedyAgent):
        pass

    try:
        cfg = GameConfig.from_dict(
            {
                "grid": 7,
                "players": [
                    {"name": "a", "team": "A", "agent": "test_registry_agent"},
                    {"name": "b", "team": "B"},
                ],
            }
        )
        assert isinstance(make_agent("test_registry_agent", 0, cfg), Local)
        assert Match(cfg).run().n_moves > 0
    finally:
        del AGENTS["test_registry_agent"]
    with pytest.raises(ValueError, match="groupe"):
        load_plugins("tron.autre")


def test_injected_entry_point_conflicts_with_program_registration() -> None:
    @register_agent("test_program_agent")
    class Local(GreedyAgent):
        pass

    try:
        with pytest.warns(PluginWarning, match="déjà enregistré"):
            infos = load_plugins(
                AGENT_GROUP,
                entry_points=[
                    md.EntryPoint("test_program_agent", "tron.agents:RandomAgent", AGENT_GROUP)
                ],
            )
        assert [i.status for i in infos] == ["conflict"] and AGENTS["test_program_agent"] is Local
    finally:
        del AGENTS["test_program_agent"]
        _reload_all()
