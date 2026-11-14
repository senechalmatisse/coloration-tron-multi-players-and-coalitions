"""Comparaison contrôlée des options de recherche.

Pour chaque position d'un jeu déterministe, chaque variante (une option à la
fois) est jouée par un agent neuf et comparée à la recherche de référence
``plain`` (sans tri ni mémoire) : meilleur coup, coups ex aequo, valeurs racine
exactes, nœuds, coupes et temps. Une variante exacte doit reproduire les
décisions et les valeurs de la référence ; seuls les nœuds et le temps changent.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from tron.agents.base import make_agent
from tron.agents.search import SearchResult
from tron.agents.searchers import SearchAgent
from tron.benchmarks.positions import BenchPosition

PLAIN: dict[str, Any] = {"ordering": "static", "iterative_deepening": False, "tt": False}
"""Recherche de référence : alpha-bêta ou MaxN exacts, tri N, E, S, O, sans mémoire."""
VARIANTS: dict[str, dict[str, Any]] = {
    "plain": PLAIN,
    "history": {**PLAIN, "ordering": "history"},
    "mobility": {**PLAIN, "ordering": "mobility"},
    "territory": {**PLAIN, "ordering": "territory"},
    "tt": {**PLAIN, "tt": True},
    "iterative": {**PLAIN, "iterative_deepening": True},
    "iterative+tt": {**PLAIN, "iterative_deepening": True, "tt": True},
    "history+it+tt": {"ordering": "history", "iterative_deepening": True, "tt": True},
    "mobility+it+tt": {"ordering": "mobility", "iterative_deepening": True, "tt": True},
    "aspiration": {
        "ordering": "history",
        "iterative_deepening": True,
        "tt": True,
        "aspiration": 0.05,
    },
    "reduction": {**PLAIN, "reduction": 1},
}
"""Variantes mesurées, une option à la fois par rapport à ``plain`` (``history+it+tt`` :
options par défaut de Paranoid). ``reduction`` est inexacte par construction."""
EXACT_VARIANTS = tuple(v for v in VARIANTS if v != "reduction")


def _search(agent: SearchAgent, pos: BenchPosition, depth: int) -> tuple[SearchResult, float]:
    st = pos.state()
    key = st.key
    t0 = time.perf_counter()
    result = agent.run_search(st, depth)
    elapsed = time.perf_counter() - t0
    if st.key != key:  # pragma: no cover - garde-fou
        raise RuntimeError("la recherche n'a pas restauré l'état")
    return result, elapsed


def same_decision(a: SearchResult, b: SearchResult, tol: float = 1e-9) -> dict[str, bool]:
    """Comparaison de deux recherches : meilleur coup, ex aequo, valeurs exactes communes."""
    common = set(a.exact) & set(b.exact)
    values = all(abs(a.root_values[m] - b.root_values[m]) <= tol for m in common)
    return {
        "best": a.best == b.best,
        "ties": a.ties == b.ties,
        "value": abs(a.value - b.value) <= tol,
        "exact_values": values,
    }


def compare_variants(
    positions: Sequence[BenchPosition],
    *,
    agent: str = "paranoid",
    depth: int = 4,
    variants: Sequence[str] = tuple(VARIANTS),
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Une ligne par variante : totaux et accord avec la base sur toutes les positions."""
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"variantes inconnues : {sorted(unknown)} (connues : {list(VARIANTS)})")
    base_params = dict(params or {})
    bases: list[SearchResult] = []
    rows: list[dict[str, Any]] = []
    for name in ["plain", *[v for v in variants if v != "plain"]]:
        nodes = prunes = evals = researches = 0
        seconds = 0.0
        agree = {"best": 0, "ties": 0, "value": 0, "exact_values": 0}
        for i, pos in enumerate(positions):
            a = make_agent(agent, pos.state().current, pos.config, **base_params, **VARIANTS[name])
            assert isinstance(a, SearchAgent)
            result, elapsed = _search(a, pos, depth)
            if name == "plain":
                bases.append(result)
            same = same_decision(result, bases[i])
            for k, ok in same.items():
                agree[k] += ok
            nodes += result.nodes
            prunes += result.prunes
            evals += result.order_evals
            researches += result.researches
            seconds += elapsed
        n = len(positions)
        rows.append(
            {
                "variant": name,
                "agent": agent,
                "depth": depth,
                "positions": n,
                "nodes": nodes,
                "prunes": prunes,
                "order_evals": evals,
                "researches": researches,
                "seconds": round(seconds, 4),
                **{f"same_{k}": v for k, v in agree.items()},
                "exact": name in EXACT_VARIANTS,
            }
        )
    base_row = rows[0]
    for row in rows:
        row["nodes_ratio"] = (
            round(row["nodes"] / base_row["nodes"], 3) if base_row["nodes"] else None
        )
        row["time_ratio"] = (
            round(row["seconds"] / base_row["seconds"], 3) if base_row["seconds"] else None
        )
    return rows


def variants_text(rows: Sequence[Mapping[str, Any]]) -> str:
    head = rows[0]
    lines = [
        f"{head['agent']} profondeur {head['depth']} · {head['positions']} positions",
        f"{'variante':<14}{'nœuds':>11}{'rapport':>9}{'temps s':>9}{'rapport':>9}"
        f"{'coupes':>9}{'tri':>8}{'re-rech.':>9}  accord (meilleur, ex aequo, valeurs)",
    ]
    for r in rows:
        n = r["positions"]
        lines.append(
            f"{r['variant']:<14}{r['nodes']:>11,}{r['nodes_ratio']:>9.2f}{r['seconds']:>9.2f}"
            f"{r['time_ratio']:>9.2f}{r['prunes']:>9,}{r['order_evals']:>8,}{r['researches']:>9}"
            f"  {r['same_best']}/{n}, {r['same_ties']}/{n}, {r['same_exact_values']}/{n}"
            + ("" if r["exact"] else "  (inexacte)")
        )
    return "\n".join(lines)


__all__ = ["EXACT_VARIANTS", "VARIANTS", "compare_variants", "same_decision", "variants_text"]
