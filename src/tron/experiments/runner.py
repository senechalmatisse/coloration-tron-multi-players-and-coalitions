"""Exécution parallèle d'un plan d'expériences (F5.4).

Chaque partie est une fonction pure de sa configuration et de sa graine : le
résultat ne dépend pas de l'ordonnancement entre processus. Les lignes sont
écrites au fil de l'eau dans ``results.csv`` ; une relance reprend là où
l'exécution s'était arrêtée. Les noyaux compilés sont préchauffés dans chaque
processus avant la première partie (temps de compilation JIT exclus).
"""

from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tron.experiments.plan import Plan, Task
from tron.experiments.store import ResultStore
from tron.sim.match import Match, MatchObserver, ensure_warmup
from tron.sim.recorder import Recorder
from tron.sim.result import METRICS_VERSION, environment_info


@dataclass(frozen=True, slots=True)
class RunSummary:
    total: int
    skipped: int
    played: int
    elapsed: float
    results: Path


def _cell(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, list | dict | tuple) else value


def run_task(task: Task, log_dir: str | None = None) -> dict[str, Any]:
    """Joue une partie et renvoie sa ligne de résultats (fonction de processus)."""
    ensure_warmup()
    observers: list[MatchObserver] = []
    if log_dir is not None:
        name = f"c{task.cond_index:04d}_g{task.game:04d}.jsonl"
        observers.append(Recorder(Path(log_dir) / name))
    result = Match(task.game_config(), observers=observers, metrics=task.metrics).run()
    row: dict[str, Any] = {
        "plan": task.plan,
        "condition": task.condition,
        "cond_index": task.cond_index,
        "game": task.game,
    }
    row.update({f"f_{k}": _cell(v) for k, v in task.factors.items()})
    row.update(result.to_row())
    return row


def _progress(total: int) -> Callable[[int], None]:
    try:
        from tqdm import tqdm  # noqa: PLC0415 - dépendance optionnelle
    except ImportError:
        start = time.perf_counter()
        state = {"done": 0, "last": 0.0}

        def report(k: int) -> None:
            state["done"] += k
            now = time.perf_counter()
            if now - state["last"] > 5 or state["done"] == total:
                state["last"] = now
                rate = state["done"] / max(now - start, 1e-9)
                eta = (total - state["done"]) / rate if rate > 0 else float("inf")
                print(
                    f"  {state['done']}/{total} parties, {rate:.2f}/s, reste ~{eta:.0f} s",
                    file=sys.stderr,
                    flush=True,
                )

        return report
    bar = tqdm(total=total, unit="partie", smoothing=0.05)

    def update(k: int) -> None:
        bar.update(k)
        if bar.n >= total:
            bar.close()

    return update


def run_plan(
    plan: Plan,
    out_dir: str | Path,
    workers: int | None = None,
    resume: bool = True,
    limit: int | None = None,
    progress: bool = True,
    logs: bool | None = None,
) -> RunSummary:
    """Exécute le plan ; écrit ``results.csv`` et ``manifest.json`` dans ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = ResultStore(out / "results.csv")
    tasks = plan.tasks()
    done = store.completed() if resume else set()
    if not resume and store.path.exists():
        store.path.unlink()
    pending = [t for t in tasks if t.key not in done]
    already = len(tasks) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    log_dir = str(out / "logs") if (plan.logs if logs is None else logs) else None
    manifest = {
        "plan": plan.to_dict(),
        "environment": environment_info(),
        "metrics_version": METRICS_VERSION if plan.metrics else None,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "tasks_total": len(tasks),
        "tasks_already_done": already,
        "tasks_this_run": len(pending),
        "workers": workers or os.cpu_count(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    start = time.perf_counter()
    report = _progress(len(pending)) if progress and pending else (lambda k: None)
    try:
        for row in _execute(pending, workers, log_dir):
            store.append(row)
            report(1)
    finally:
        store.close()
    return RunSummary(
        total=len(tasks),
        skipped=already,
        played=len(pending),
        elapsed=time.perf_counter() - start,
        results=store.path,
    )


def _execute(
    tasks: list[Task], workers: int | None, log_dir: str | None
) -> Iterable[dict[str, Any]]:
    n_workers = workers if workers is not None else (os.cpu_count() or 1)
    if n_workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield run_task(task, log_dir)
        return
    ensure_warmup()  # remplit le cache de compilation avant de lancer les processus
    with ProcessPoolExecutor(max_workers=n_workers, initializer=ensure_warmup) as pool:
        queue = iter(tasks)
        running: set[Future[dict[str, Any]]] = set()
        try:
            for task in itertools.islice(queue, 4 * n_workers):
                running.add(pool.submit(run_task, task, log_dir))
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    yield fut.result()
                    nxt = next(queue, None)
                    if nxt is not None:
                        running.add(pool.submit(run_task, nxt, log_dir))
        except BaseException:
            for fut in running:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def estimate(
    plan: Plan, samples: int = 10, workers: int | None = None, seed: int = 0
) -> dict[str, float]:
    """Estime le budget de calcul en jouant ``samples`` parties tirées au hasard (§7.3)."""
    tasks = plan.tasks()
    rng = random.Random(seed)
    chosen = rng.sample(tasks, min(samples, len(tasks)))
    ensure_warmup()
    durations = []
    for task in chosen:
        t0 = time.perf_counter()
        run_task(task)
        durations.append(time.perf_counter() - t0)
    mean = sum(durations) / len(durations)
    n_workers = workers if workers is not None else (os.cpu_count() or 1)
    return {
        "tasks": float(len(tasks)),
        "sampled": float(len(chosen)),
        "mean_game_s": mean,
        "max_game_s": max(durations),
        "cpu_hours": mean * len(tasks) / 3600,
        "wall_hours": mean * len(tasks) / 3600 / n_workers,
    }
