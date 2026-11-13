"""Suite de mesures de performance (P13) : composants mesurés séparément.

Chaque mesure tourne sur le même jeu de positions déterministe
(:func:`tron.benchmarks.positions.position_set`), avec des nombres de
répétitions fixes : deux exécutions de même paramètres font exactement le même
travail. Les noyaux JIT sont préchauffés avant tout chronométrage. Chaque
mesure porte aussi une **somme de contrôle** déterministe (nœuds, coups choisis,
comptes de Voronoi…) : une optimisation qui changerait le résultat d'un calcul
change sa somme de contrôle, ce que :func:`compare` signale.

Format de sortie ``tron-bench/1`` (JSON) : environnement, paramètres, empreinte
des positions, liste des mesures.
"""

from __future__ import annotations

import cProfile
import csv
import hashlib
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tron
from tron._jit import backend_name
from tron.agents.base import make_agent
from tron.benchmarks.positions import (
    PHASES,
    POSITION_SET_VERSION,
    BenchPosition,
    fingerprint,
    position_set,
)
from tron.core.state import GameState
from tron.eval.composite import CompositeHeuristic
from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.match import Match, ensure_warmup

BENCH_FORMAT = "tron-bench/1"
_ENVIRONMENT_COLUMNS = frozenset(
    {"duration", "code_version", "git_hash", "python", "numpy", "backend"}
)
"""Colonnes de résultat qui dépendent de la machine ou de la version : hors somme de contrôle."""
MEASURES = (
    "legal_moves",
    "apply_undo",
    "voronoi",
    "composite",
    "search",
    "mcts",
    "results_write",
    "render",
)


@dataclass(frozen=True)
class SuiteParams:
    """Paramètres d'une exécution de la suite (tous enregistrés dans le résultat)."""

    grid: int = 20
    players: int = 4
    per_phase: int = 5
    seed: int = 0
    agent: str = "paranoid"
    depth: int = 3
    simulations: int = 200
    repeats: int = 200
    """Répétitions par position des mesures élémentaires (coups, Voronoi, composite)."""
    rows: int = 2000
    """Lignes écrites pour la mesure d'écriture des résultats."""
    quick: bool = False

    @classmethod
    def quick_params(cls, **kw: Any) -> SuiteParams:
        """Paramètres réduits (tests, essais rapides)."""
        base: dict[str, Any] = {
            "grid": 10,
            "players": 2,
            "per_phase": 2,
            "depth": 2,
            "simulations": 30,
            "repeats": 10,
            "rows": 100,
            "quick": True,
        }
        base.update(kw)
        return cls(**base)


@dataclass
class Measure:
    name: str
    phase: str
    unit: str
    ops: int
    seconds: float
    checksum: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def per_op_us(self) -> float:
        return 1e6 * self.seconds / self.ops if self.ops else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "unit": self.unit,
            "ops": self.ops,
            "seconds": round(self.seconds, 6),
            "per_op_us": round(self.per_op_us, 3),
            "per_second": round(self.ops / self.seconds, 1) if self.seconds > 0 else None,
            "checksum": self.checksum,
            **self.extra,
        }


def _digest(values: Any) -> str:
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def environment() -> dict[str, Any]:
    return {
        "code_version": tron.__version__,
        "backend": backend_name(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _by_phase(positions: Sequence[BenchPosition]) -> dict[str, list[GameState]]:
    out: dict[str, list[GameState]] = {phase: [] for phase in PHASES}
    for pos in positions:
        out[pos.phase].append(pos.state())
    return out


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    value = fn()
    return time.perf_counter() - t0, value


def _elementary(
    name: str, phase: str, states: list[GameState], repeats: int, params: SuiteParams
) -> Measure:
    n = params.players
    if name == "legal_moves":

        def run() -> Any:
            return [sum(len(st.legal_moves()) for _ in range(repeats)) for st in states]

        ops = repeats * len(states)
        unit = "appel"
    elif name == "apply_undo":

        def run() -> Any:
            total = 0
            for st in states:
                moves = st.legal_moves()
                key = st.key
                for _ in range(repeats):
                    for m in moves:
                        st.apply(m)
                        try:
                            total += st.n_moves
                        finally:
                            st.undo()
                assert st.key == key
            return total

        ops = repeats * sum(len(st.legal_moves()) for st in states)
        unit = "apply+undo"
    elif name == "voronoi":
        heur = VoronoiHeuristic(params.grid, params.grid, n)

        def run() -> Any:
            return [heur.counts(st)[0] for st in states for _ in range(repeats)][:: repeats or 1]

        ops = repeats * len(states)
        unit = "évaluation"
    else:  # composite
        comp = CompositeHeuristic(params.grid, params.grid, n)

        def run() -> Any:
            return [
                [round(v, 9) for v in comp.features(st).territory]
                for st in states
                for _ in range(repeats)
            ][:: repeats or 1]

        ops = repeats * len(states)
        unit = "caractéristiques"
    run()  # préchauffage (JIT, caches)
    seconds, value = _timed(run)
    return Measure(name, phase, unit, ops, seconds, _digest(value))


def _decisions(
    name: str, phase: str, positions: list[BenchPosition], params: SuiteParams
) -> Measure:
    moves: list[int] = []
    nodes = 0
    seconds = 0.0
    for pos in positions:
        st = pos.state()
        key = st.key
        if name == "search":
            agent = make_agent(params.agent, st.current, pos.config, depth=params.depth)
        else:
            agent = make_agent("mcts", st.current, pos.config, simulations=params.simulations)
        t0 = time.perf_counter()
        decision = agent.choose_move(st)
        seconds += time.perf_counter() - t0
        assert st.key == key, "l'agent n'a pas restauré l'état"
        moves.append(decision.move)
        nodes += decision.stats.nodes
    ops = len(positions) if name == "search" else len(positions) * params.simulations
    extra: dict[str, Any] = {
        "nodes": nodes,
        "nodes_per_second": round(nodes / seconds, 1) if seconds else None,
    }
    if name == "search":
        extra["agent"] = f"{params.agent}(d={params.depth})"
    else:
        extra["agent"] = f"mcts({params.simulations})"
    unit = "décision" if name == "search" else "simulation"
    return Measure(name, phase, unit, ops, seconds, _digest([moves, nodes]), extra)


def _results_write(params: SuiteParams, sample: BenchPosition) -> Measure:
    cfg = sample.config.with_seed(params.seed)
    result = Match(cfg, metrics=False).run()
    row = {"game": 0, **result.to_row()}
    stable = {
        k: v
        for k, v in row.items()
        if k not in _ENVIRONMENT_COLUMNS and not str(k).startswith("time_p")
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.csv"

        def run() -> int:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
                for g in range(params.rows):
                    writer.writerow({**row, "game": g})
                    fh.flush()
            return path.stat().st_size

        run()
        seconds, _size = _timed(run)
    return Measure("results_write", "all", "ligne", params.rows, seconds, _digest(stable))


def _render(states: list[GameState], params: SuiteParams) -> Measure:
    from tron.viz.palette import make_palette  # noqa: PLC0415 - NumPy seulement, sans Arcade
    from tron.viz.render import render_frame  # noqa: PLC0415

    palette = make_palette(states[0].config.team_index)
    reps = max(1, params.repeats // 20)

    def run() -> Any:
        return [int(render_frame(st, palette, cell=8).sum()) for st in states for _ in range(reps)]

    run()
    seconds, value = _timed(run)
    return Measure("render", "all", "image", reps * len(states), seconds, _digest(value[::reps]))


def run_suite(params: SuiteParams | None = None) -> dict[str, Any]:
    """Exécute toutes les mesures ; renvoie le résultat au format ``tron-bench/1``."""
    params = params or SuiteParams()
    ensure_warmup()
    positions = position_set(params.grid, params.players, params.per_phase, params.seed)
    states = _by_phase(positions)
    measures: list[Measure] = []
    for phase in PHASES:
        group = states[phase]
        phase_positions = [p for p in positions if p.phase == phase]
        for name in ("legal_moves", "apply_undo", "voronoi", "composite"):
            measures.append(_elementary(name, phase, group, params.repeats, params))
        for name in ("search", "mcts"):
            measures.append(_decisions(name, phase, phase_positions, params))
    measures.append(_results_write(params, positions[0]))
    measures.append(_render([pos.state() for pos in positions], params))
    return {
        "format": BENCH_FORMAT,
        "environment": environment(),
        "params": params.__dict__,
        "positions": {
            "version": POSITION_SET_VERSION,
            "count": len(positions),
            "phases": {phase: len(states[phase]) for phase in PHASES},
            "fingerprint": fingerprint(positions),
        },
        "measures": [m.to_dict() for m in measures],
    }


def compare(current: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Mesure par mesure : temps de référence, temps courant, rapport, somme de contrôle.

    ``same_result`` est faux si la somme de contrôle diffère (résultat de calcul
    changé) ; ``comparable`` est faux si les paramètres ou les positions diffèrent.
    """
    for doc in (current, reference):
        if doc.get("format") != BENCH_FORMAT:
            raise ValueError(f"format de mesures inconnu : {doc.get('format')!r}")
    comparable = (
        current["params"] == reference["params"]
        and current["positions"]["fingerprint"] == reference["positions"]["fingerprint"]
    )
    ref = {(m["name"], m["phase"]): m for m in reference["measures"]}
    rows = []
    for m in current["measures"]:
        r = ref.get((m["name"], m["phase"]))
        if r is None:
            continue
        rows.append(
            {
                "name": m["name"],
                "phase": m["phase"],
                "reference_us": r["per_op_us"],
                "current_us": m["per_op_us"],
                "ratio": round(m["per_op_us"] / r["per_op_us"], 3) if r["per_op_us"] else None,
                "same_result": m["checksum"] == r["checksum"],
                "comparable": comparable,
                "reference_backend": reference["environment"]["backend"],
                "current_backend": current["environment"]["backend"],
            }
        )
    return rows


def suite_text(result: dict[str, Any]) -> str:
    """Tableau texte d'un résultat ``tron-bench/1``."""
    env = result["environment"]
    p = result["params"]
    lines = [
        f"tron {env['code_version']} · noyaux : {env['backend']} · Python {env['python']} · "
        f"grille {p['grid']}x{p['grid']} · {p['players']} joueurs · "
        f"{result['positions']['count']} positions ({result['positions']['fingerprint'][:12]})",
        f"{'mesure':<14}{'phase':<13}{'µs/op':>12}{'op/s':>14}  {'unité':<17}contrôle",
    ]
    for m in result["measures"]:
        rate = f"{m['per_second']:,.0f}" if m["per_second"] else "-"
        lines.append(
            f"{m['name']:<14}{m['phase']:<13}{m['per_op_us']:>12,.1f}{rate:>14}  "
            f"{m['unit']:<17}{m['checksum']}"
        )
    return "\n".join(lines)


def comparison_text(rows: Sequence[dict[str, Any]]) -> str:
    lines = [f"{'mesure':<14}{'phase':<13}{'réf. µs':>12}{'µs':>12}{'rapport':>9}  résultat"]
    for r in rows:
        same = "identique" if r["same_result"] else "DIFFÉRENT"
        ratio = f"{r['ratio']:.2f}" if r["ratio"] is not None else "-"
        lines.append(
            f"{r['name']:<14}{r['phase']:<13}{r['reference_us']:>12,.1f}"
            f"{r['current_us']:>12,.1f}{ratio:>9}  {same}"
        )
    if rows and not rows[0]["comparable"]:
        lines.append("attention : paramètres ou positions différents de la référence")
    return "\n".join(lines)


# ------------------------------------------------------------------ profilage
def profile(
    fn: Callable[[], Any], *, sort: str = "cumulative", limit: int = 25
) -> tuple[str, list[dict[str, Any]], pstats.Stats]:
    """Profil ``cProfile`` de ``fn`` : texte ``pstats``, lignes JSON et statistiques."""
    prof = cProfile.Profile()
    prof.enable()
    try:
        fn()
    finally:
        prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats(sort)
    stats.print_stats(limit)
    rows = []
    raw: dict[Any, Any] = stats.stats  # type: ignore[attr-defined]
    keyed = sorted(
        raw.items(), key=lambda kv: kv[1][3 if sort == "cumulative" else 2], reverse=True
    )
    for (filename, line, func), (_cc, ncalls, tottime, cumtime, _callers) in keyed[:limit]:
        rows.append(
            {
                "function": func,
                "file": filename,
                "line": line,
                "ncalls": ncalls,
                "tottime": round(tottime, 6),
                "cumtime": round(cumtime, 6),
            }
        )
    return buf.getvalue(), rows, stats


# --------------------------------------------------------------- campagnes
_LAUNCHER = (
    "import json, resource, subprocess, sys\n"
    "proc = subprocess.run(sys.argv[1:], capture_output=True, text=True)\n"
    "usage = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "print(json.dumps({'code': proc.returncode, 'stderr': proc.stderr[-2000:],"
    " 'cpu': usage.ru_utime + usage.ru_stime, 'maxrss_kb': usage.ru_maxrss}))\n"
)
"""Processus intermédiaire neuf : ses compteurs ``RUSAGE_CHILDREN`` ne couvrent qu'une campagne."""


def bench_campaign(
    plan: str | Path, workers: Sequence[int] = (1, 2), limit: int = 8
) -> list[dict[str, Any]]:
    """Temps, CPU et mémoire d'une campagne ``tron run`` pour chaque nombre de processus.

    Chaque campagne (sans reprise, ``limit`` parties) tourne sous un processus
    intermédiaire neuf qui mesure, par ``getrusage``, le temps CPU de tous ses
    descendants (processus de calcul compris) et la mémoire résidente maximale
    du plus gros d'entre eux (Linux, macOS).
    """
    rows = []
    env = {**os.environ, "PYTHONPATH": str(Path(tron.__file__).parents[1])}
    for w in workers:
        if w < 1:
            raise ValueError(f"nombre de processus invalide : {w}")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "tron",
                "run",
                str(plan),
                "--out",
                tmp,
                "--workers",
                str(w),
                "--limit",
                str(limit),
                "--no-resume",
            ]
            t0 = time.perf_counter()
            out = subprocess.run(
                [sys.executable, "-c", _LAUNCHER, *cmd],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            wall = time.perf_counter() - t0
            usage = json.loads(out.stdout.strip().splitlines()[-1])
            if usage["code"] != 0:
                raise RuntimeError(f"campagne en échec ({w} processus) : {usage['stderr']}")
            with (Path(tmp) / "results.csv").open(encoding="utf-8") as fh:
                games = sum(1 for _ in csv.DictReader(fh))
        rss_kb = usage["maxrss_kb"] if sys.platform != "darwin" else usage["maxrss_kb"] / 1024
        rows.append(
            {
                "workers": w,
                "games": games,
                "wall_s": round(wall, 3),
                "cpu_s": round(usage["cpu"], 3),
                "games_per_s": round(games / wall, 3) if wall > 0 else None,
                "max_rss_mb": round(rss_kb / 1024, 1),
            }
        )
    return rows


def write_json(data: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


__all__ = [
    "BENCH_FORMAT",
    "MEASURES",
    "Measure",
    "SuiteParams",
    "bench_campaign",
    "compare",
    "comparison_text",
    "environment",
    "profile",
    "run_suite",
    "suite_text",
    "write_json",
]
