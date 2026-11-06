"""Analyse complète d'un fichier de résultats : tableaux, tests, figures, régression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tron.analysis import figures
from tron.analysis.regression import default_formula, fit_logit
from tron.analysis.stats import factor_columns, load_results, paired_comparisons, summarize


@dataclass
class AnalysisOutput:
    summary: Path
    figures: list[Path] = field(default_factory=list)
    paired: Path | None = None
    regression: Path | None = None
    messages: list[str] = field(default_factory=list)


def analyze(
    results: str | Path, out_dir: str | Path, formula: str | None = None, annotate: bool = False
) -> AnalysisOutput:
    """Produit ``summary.csv``, ``paired.csv``, ``regression.txt`` et les figures PNG."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(str(results))
    summary = summarize(df)
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    result = AnalysisOutput(summary=summary_path)
    factors = factor_columns(df)

    # Comparaisons appariées des profondeurs consécutives (H1).
    depth_factor = next((f for f in ("f_solo_depth", "f_depth_t0") if f in factors), None)
    if depth_factor is not None:
        paired = paired_comparisons(df, depth_factor)
        if not paired.empty:
            result.paired = out / "paired.csv"
            paired.to_csv(result.paired, index=False)

    # Figures de l'étude A : une série de figures par niveau des autres facteurs.
    if {"f_solo_depth", "f_k"} <= set(factors):
        result.figures += _study_a_figures(summary, factors, out, annotate)
    elif {"f_depth_t0", "f_composition"} <= set(factors):
        result.figures += _study_b_figures(summary, factors, out)

    # Régression logistique (F7.3).
    if formula is not None or "f_solo_depth" in factors:
        try:
            reg = fit_logit(df, formula or default_formula(df))
        except (ValueError, KeyError, ArithmeticError) as exc:
            result.messages.append(f"régression non ajustée : {exc}")
        else:
            result.regression = out / "regression.txt"
            result.regression.write_text(reg.to_text() + "\n", encoding="utf-8")
            reg.table.to_csv(out / "regression.csv", index=False)
    return result


def _groups(
    summary: pd.DataFrame, factors: list[str], used: set[str]
) -> list[tuple[str, pd.DataFrame]]:
    rest = [f for f in factors if f not in used and summary[f].nunique() > 1]
    if not rest:
        return [("", summary)]
    out = []
    for values, sub in summary.groupby(rest, sort=True):
        vals = values if isinstance(values, tuple) else (values,)
        tag = "_".join(f"{f[2:]}-{v}" for f, v in zip(rest, vals, strict=True))
        out.append((tag, sub))
    return out


def _study_a_figures(
    summary: pd.DataFrame, factors: list[str], out: Path, annotate: bool
) -> list[Path]:
    paths = []
    used = {"f_solo_depth", "f_k", "f_grid"}
    for tag, sub in _groups(summary, factors, used):
        suffix = f"_{tag}" if tag else ""
        paths.append(figures.winrate_vs_depth(sub, out / f"winrate_solo_depth{suffix}.png"))
        if "nodes_per_decision" in sub:
            paths.append(figures.cost_vs_depth(sub, out / f"cost_solo_depth{suffix}.png"))
    if "f_coalition_depth" in factors:
        for tag, sub in _groups(summary, factors, used | {"f_coalition_depth"}):
            suffix = f"_{tag}" if tag else ""
            paths.append(
                figures.heatmaps(sub, out / f"heatmap_depths{suffix}.png", annotate=annotate)
            )
    return paths


def _study_b_figures(summary: pd.DataFrame, factors: list[str], out: Path) -> list[Path]:
    paths = []
    used = {"f_depth_t0", "f_composition", "f_grid"}
    for tag, sub in _groups(summary, factors, used):
        suffix = f"_{tag}" if tag else ""
        paths.append(
            figures.winrate_vs_depth(
                sub,
                out / f"winrate_team_depth{suffix}.png",
                x="f_depth_t0",
                series="f_composition",
                facet="f_grid",
                title="Taux de victoire de l'équipe T0",
            )
        )
    return paths
