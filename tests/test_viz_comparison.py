"""Comparaison synchronisée de deux rejeux (P17), testée sans fenêtre graphique."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tron.cli.main import main
from tron.core import GameConfig
from tron.sim import GameLog, Match, Recorder
from tron.viz.comparison import ComparisonSession, describe, first_divergence, pairing, timeline


def _config(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "mini",
        "grid": 7,
        "seed": 5,
        "placement": {"positions": [[1, 1], [5, 5], [5, 1]]},
        "players": [
            {"name": "a", "team": "A", "agent": "greedy"},
            {"name": "b", "team": "B", "agent": "greedy"},
            {"name": "c", "team": "A", "agent": "greedy"},
        ],
    }
    base.update(over)
    return base


def _log(path: Path, **over: Any) -> Path:
    Match(GameConfig.from_dict(_config(**over)), observers=[Recorder(path)]).run()
    return path


def _legacy(src: Path, dst: Path) -> Path:
    """Journal ``tron-log/1`` : en-tête sans diplomatie, coups sans détails ni événements."""
    lines = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines()]
    out = []
    for rec in lines:
        if rec["type"] == "header":
            rec = {k: v for k, v in rec.items() if k != "diplomacy"}
            rec["format"] = "tron-log/1"
        elif rec["type"] == "move":
            rec = {k: v for k, v in rec.items() if k not in ("details", "diplomacy", "zone")}
        out.append(json.dumps(rec))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


@pytest.fixture
def logs(tmp_path: Path) -> dict[str, Path]:
    greedy_b = _config()["players"]
    random_b = [dict(p) for p in greedy_b]
    random_b[1]["agent"] = "random"
    return {
        "a": _log(tmp_path / "a.jsonl"),
        "same": _log(tmp_path / "same.jsonl"),
        "agents": _log(tmp_path / "agents.jsonl", players=random_b),
        "seed": _log(tmp_path / "seed.jsonl", seed=6),
        "first": _log(tmp_path / "first.jsonl", first_player=1),
        "place": _log(tmp_path / "place.jsonl", placement={"positions": [[1, 2], [5, 5], [5, 1]]}),
        "grid": _log(tmp_path / "grid.jsonl", grid=8),
    }


def test_logs_load_independently_and_pairing_is_checked(logs: dict[str, Path]) -> None:
    a = GameLog.load(logs["a"])
    assert pairing(a, GameLog.load(logs["same"])).paired
    assert pairing(a, GameLog.load(logs["agents"])).paired  # mêmes départ et sièges
    for key, reason in (
        ("seed", "graines"),
        ("first", "premiers joueurs"),
        ("place", "placements"),
        ("grid", "grilles"),
    ):
        check = pairing(a, GameLog.load(logs[key]))
        assert not check.paired and reason in check.warning
    teams = GameLog.load(logs["a"])
    other = GameLog.load(logs["agents"])
    other.config = GameConfig.from_dict(
        {
            **other.config.to_dict(),
            "players": [{**p, "team": t} for p, t in zip(_config()["players"], "ABB", strict=True)],
        }
    )
    assert "équipes" in pairing(teams, other).warning


def test_synchronised_cursor_freezes_the_shorter_game(logs: dict[str, Path]) -> None:
    a, b = GameLog.load(logs["a"]), GameLog.load(logs["agents"])
    session = ComparisonSession(a, b)
    lengths = (len(a.moves), len(b.moves))
    assert session.max_ply == max(lengths) and session.ply == 0
    short = 0 if lengths[0] < lengths[1] else 1
    session.end()
    assert session.frozen(short) == (lengths[0] != lengths[1])
    assert session.sources[short].index == lengths[short]
    assert session.sources[short].state.is_terminal()
    view = session.view(short)
    assert view.frozen == session.frozen(short) and view.shown_ply == lengths[short]
    assert session.seek(10_000) == session.max_ply and session.seek(-3) == 0
    snapshots = []
    for _ in range(2):  # déterministe : mêmes curseurs, mêmes états
        session.start()
        trace = []
        while session.forward():
            trace.append([src.state.to_snapshot() for src in session.sources])
        snapshots.append(trace)
        assert not session.forward() and session.ply == session.max_ply
    assert snapshots[0] == snapshots[1]
    assert session.backward() and session.ply == session.max_ply - 1


def test_views_rows_divergence_and_events(logs: dict[str, Path]) -> None:
    a, same, agents = (GameLog.load(logs[k]) for k in ("a", "same", "agents"))
    assert first_divergence(a, same) is None
    div = first_divergence(a, agents)
    assert div is not None and div <= min(len(a.moves), len(agents.moves))
    cut = GameLog.load(logs["a"])
    cut.moves = cut.moves[:5]
    assert first_divergence(a, cut) == 5
    session = ComparisonSession(a, agents, labels=("glouton", "aléatoire"))
    session.seek(6)
    view = session.view(0)
    assert sum(t.decisions for t in view.totals) == 6
    assert sum(view.shares) == pytest.approx(1.0) and set(view.team_shares) == {"A", "B"}
    assert view.winner in ("A", "B") and len(view.ranks) == 3
    rows = dict((label, (va, vb)) for label, va, vb in session.rows())
    assert rows["graine · premier joueur"] == ("5 · 0", "5 · 0")
    assert rows["première divergence"][0] == f"demi-coup {div}"
    assert "glouton" in session.text() and "ATTENTION" not in session.text()
    data = describe(session)
    assert data["paired"] and data["ply"] == 6 and data["divergence"] == div
    events = timeline(a)
    assert [e.kind for e in events].count("elimination") == sum(m["elim"] for m in a.moves)


def test_event_timelines_include_zones_obstacles_and_diplomacy(tmp_path: Path) -> None:
    zone = _log(
        tmp_path / "zone.jsonl",
        grid=9,
        placement={"positions": [[1, 1], [7, 7], [7, 1]]},
        rules={"mode": "zone_control", "zones": [[0, 0, 3, 3], [6, 6, 3, 3]], "max_rounds": 6},
    )
    kinds = {e.kind for e in timeline(GameLog.load(zone))}
    assert "zone" in kinds
    obst = _log(tmp_path / "obst.jsonl", rules={"obstacles": {"every": 1, "count": 2}})
    assert "obstacles" in {e.kind for e in timeline(GameLog.load(obst))}
    log = GameLog.load(zone)
    proposal = {"ply": 0, "player": 0, "target": 2, "kind": "proposed"}
    log.moves[0] = {**log.moves[0], "diplomacy": [proposal]}
    texts = [e.text for e in timeline(log) if e.kind == "diplomacy"]
    assert texts == ["proposed a → c"]


def test_legacy_v1_logs_are_compared(logs: dict[str, Path], tmp_path: Path) -> None:
    legacy = GameLog.load(_legacy(logs["a"], tmp_path / "v1.jsonl"))
    assert legacy.format == "tron-log/1"
    session = ComparisonSession(legacy, GameLog.load(logs["same"]))
    assert session.pairing.paired and session.divergence is None
    session.end()
    assert session.view(0).winner == session.view(1).winner


def test_cli_compare_text_json_and_exports(
    logs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import imageio.v3 as iio

    assert main(["compare", str(logs["a"]), str(logs["agents"]), "--no-gui"]) == 0
    out = capsys.readouterr()
    assert "Comparaison au demi-coup" in out.out and "attention" not in out.err
    assert main(["compare", str(logs["a"]), str(logs["seed"]), "--no-gui", "--ply", "3"]) == 0
    out = capsys.readouterr()
    assert "ATTENTION : comparaison non appariée" in out.out and "graines" in out.err
    code = main(
        [
            "compare",
            str(logs["a"]),
            str(logs["grid"]),
            "--no-gui",
            "--format",
            "json",
            "--png",
            str(tmp_path / "c.png"),
            "--gif",
            str(tmp_path / "c.gif"),
            "--cell",
            "6",
            "--theme",  # dimensions du rendu classique (le néon ajoute un liseré)
            "classic",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out.split("\n", 2)[2])
    assert not data["paired"] and any("grilles" in r for r in data["reasons"])
    img = iio.imread(tmp_path / "c.png")
    assert img.shape == (8 * 6, 7 * 6 + 12 + 8 * 6, 3)  # hauteurs complétées, plateaux côte à côte
    assert (tmp_path / "c.gif").stat().st_size > 0


# ------------------------------------------------- fenêtre (bouchon d'Arcade)
class _Stub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.text = args[0] if args else ""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def draw(self) -> None:
        pass


class _SpriteList(list):  # type: ignore[type-arg]
    def draw(self) -> None:
        pass


class _Window:
    def __init__(self, width: int, height: int, title: str, update_rate: float = 1 / 60) -> None:
        self.width, self.height, self.closed = width, height, False

    def clear(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_compare_window_with_arcade_stub(
    logs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    stub = types.ModuleType("arcade")
    stub.Window = _Window  # type: ignore[attr-defined]
    stub.SpriteList = _SpriteList  # type: ignore[attr-defined]
    stub.SpriteSolidColor = _Stub  # type: ignore[attr-defined]
    stub.Text = _Stub  # type: ignore[attr-defined]
    stub.draw_lrbt_rectangle_filled = lambda *a, **k: calls.append("rect")  # type: ignore[attr-defined]
    stub.draw_lrbt_rectangle_outline = lambda *a, **k: calls.append("outline")  # type: ignore[attr-defined]
    stub.draw_line = lambda *a, **k: calls.append("line")  # type: ignore[attr-defined]
    stub.run = lambda: calls.append("run")  # type: ignore[attr-defined]
    stub.key = types.SimpleNamespace(  # type: ignore[attr-defined]
        SPACE=1, RIGHT=2, LEFT=3, UP=4, DOWN=5, V=6, HOME=8, END=9, ESCAPE=11, Q=12
    )
    monkeypatch.setitem(sys.modules, "arcade", stub)
    arcade_app = importlib.reload(importlib.import_module("tron.viz.arcade_app"))
    app = importlib.reload(importlib.import_module("tron.viz.comparison_app"))
    try:
        session = ComparisonSession(GameLog.load(logs["a"]), GameLog.load(logs["seed"]))
        win = app.CompareWindow(session, cell=8)
        assert len(win.boards[0].cells) == 49 and len(win.boards[1].cells) == 49
        win.on_draw()
        assert "outline" in calls and "line" in calls
        k = stub.key
        win.on_key_press(k.RIGHT, 0)
        assert session.ply == 1 and win.paused
        win.on_key_press(k.LEFT, 0)
        assert session.ply == 0
        win.on_key_press(k.SPACE, 0)
        win.on_update(0.5)  # 10 demi-coups par seconde
        assert session.ply == 5
        for symbol in (k.UP, k.DOWN, k.V, k.END, k.HOME, k.END):
            win.on_key_press(symbol, 0)
            win.on_draw()
        assert session.ply == session.max_ply
        win.on_key_press(k.ESCAPE, 0)
        assert win.closed
        app.run_compare(session, cell=8)
        assert calls.count("run") == 1
    finally:
        monkeypatch.delitem(sys.modules, "arcade")
        importlib.reload(arcade_app)
        importlib.reload(app)
