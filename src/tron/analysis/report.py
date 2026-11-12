"""Analyse complète d'un fichier de résultats : tableaux, tests, figures, régression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tron.analysis import figures
from tron.analysis.cooperation import (
    condition_summary,
    cooperation_summary,
    diplomacy_summary,
    has_diplomacy,
    has_metrics,
)
from tron.analysis.regression import default_formula, fit_logit, new_terms, prepare
from tron.analysis.stats import factor_columns, load_results, paired_comparisons, summarize

COOPERATION_AXES = (
    "f_k",
    "f_coalition_depth",
    "f_diplomacy_enabled",
    "f_alliance_weight",
    "f_proposal_ttl_plies",
    "f_grid",
    "f_composition",
)
"""Facteurs candidats (dans cet ordre) pour la figure de l'indice de coopération."""

REFERENCE_LEVELS: dict[str, object] = {
    "f_heuristic_name": "voronoi",
    "f_solo_heuristic": "voronoi",
    "f_solo_agent": "paranoid",
    "f_ablate": "none",
    "f_diplomacy_enabled": False,
}
"""Facteurs comparés à un niveau de référence (McNemar apparié, Holm par facteur)."""

DOSE_FACTORS = (
    "f_simulations",
    "f_adaptation_rate",
    "f_alliance_weight",
    "f_proposal_ttl_plies",
    "f_w_territory",
    "f_w_mobility",
    "f_w_connectivity",
    "f_w_safety",
    "f_w_bottleneck",
)
"""Facteurs numériques comparés de niveau en niveau, avec une courbe de taux de victoire."""


@dataclass
class AnalysisOutput:
    summary: Path
    figures: list[Path] = field(default_factory=list)
    paired: Path | None = None
    regression: Path | None = None
    cooperation: Path | None = None
    cooperation_conditions: Path | None = None
    diplomacy: Path | None = None
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

    # Comparaisons appariées : profondeurs consécutives (H1), puis facteurs P06.
    tables = []
    depth_factor = next((f for f in ("f_solo_depth", "f_depth_t0") if f in factors), None)
    if depth_factor is not None:
        tables.append(paired_comparisons(df, depth_factor))
    for factor, ref in REFERENCE_LEVELS.items():
        if factor in factors and df[factor].nunique() > 1:
            tables.append(paired_comparisons(df, factor, reference=ref))
    for factor in DOSE_FACTORS:
        if factor in factors and df[factor].nunique() > 1:
            tables.append(paired_comparisons(df, factor))
    tables = [t for t in tables if not t.empty]
    if tables:
        result.paired = out / "paired.csv"
        pd.concat(tables, ignore_index=True).to_csv(result.paired, index=False)

    # Figures de l'étude A : une série de figures par niveau des autres facteurs.
    if {"f_solo_depth", "f_k"} <= set(factors):
        result.figures += _study_a_figures(summary, factors, out, annotate)
    elif {"f_depth_t0", "f_composition"} <= set(factors):
        result.figures += _study_b_figures(summary, factors, out)

    # Figures des campagnes P06 : niveaux d'un facteur catégoriel, courbes de dose.
    result.figures += _campaign_figures(df, summary, factors, out)

    # Régression logistique (F7.3), étendue aux facteurs P06 qui varient.
    if formula is not None or "f_solo_depth" in factors or new_terms(prepare(df)):
        try:
            reg = fit_logit(df, formula or default_formula(df))
        except (ValueError, KeyError, ArithmeticError) as exc:
            result.messages.append(f"régression non ajustée : {exc}")
        else:
            result.regression = out / "regression.txt"
            result.regression.write_text(reg.to_text() + "\n", encoding="utf-8")
            reg.table.to_csv(out / "regression.csv", index=False)

    # Métriques de coopération (P02) : absentes d'un CSV antérieur, sans effet sur le reste.
    if has_metrics(df):
        _cooperation_outputs(df, factors, out, result)
    if has_diplomacy(df):
        dipl = diplomacy_summary(df)
        if not dipl.empty:
            result.diplomacy = out / "diplomacy.csv"
            dipl.to_csv(result.diplomacy, index=False)
    return result


def _campaign_figures(
    df: pd.DataFrame, summary: pd.DataFrame, factors: list[str], out: Path
) -> list[Path]:
    paths = []
    for factor, ref in REFERENCE_LEVELS.items():
        if factor in factors and df[factor].nunique() > 1:
            pooled = summarize(df, by=[factor])
            paths.append(
                figures.level_comparison(
                    pooled, out / f"winrate_{factor[2:]}.png", factor, reference=ref
                )
            )
    for factor in DOSE_FACTORS:
        if factor in factors and df[factor].nunique() > 1:
            found: str | None = next(
                (
                    f
                    for f in ("f_grid", "f_k", "f_solo_agent", "f_coalition_mode")
                    if f in factors and f != factor and summary[f].nunique() > 1
                ),
                None,
            )
            data = summary if found is not None else summary.assign(_serie="toutes")
            series = found if found is not None else "_serie"
            aggs = {
                c: (c, "mean") for c in ("win_rate", "ci_low", "ci_high", "chance") if c in data
            }
            pooled = data.groupby([factor, series], as_index=False).agg(**aggs)
            paths.append(
                figures.winrate_vs_depth(
                    pooled,
                    out / f"winrate_{factor[2:]}.png",
                    x=factor,
                    series=series,
                    facet="_aucune",
                    title=f"Taux de victoire : {figures.label(factor)}",
                    logx=factor == "f_simulations",
                )
            )
    return paths


def _cooperation_outputs(
    df: pd.DataFrame, factors: list[str], out: Path, result: AnalysisOutput
) -> None:
    coop = cooperation_summary(df)
    if coop.empty:
        return
    result.cooperation = out / "cooperation.csv"
    coop.to_csv(result.cooperation, index=False)
    cond = condition_summary(df)
    if not cond.empty:
        result.cooperation_conditions = out / "cooperation_conditions.csv"
        cond.to_csv(result.cooperation_conditions, index=False)
    axis = next((f for f in COOPERATION_AXES if f in factors and coop[f].nunique() > 1), None)
    if axis is None:
        return
    multi = coop[coop["team_size"] >= 2]
    if not multi.empty and multi["cooperation_proxy_mean"].notna().any():
        others = [f for f in factors if f != axis and multi[f].nunique() > 1]
        data = multi.groupby([axis, "team"], as_index=False)["cooperation_proxy_mean"].mean()
        result.figures.append(
            figures.metric_vs_factor(data, out / f"cooperation_proxy_{axis[2:]}.png", x=axis)
        )
        if others:
            result.messages.append(
                "figure de coopération moyennée sur : " + ", ".join(f[2:] for f in others)
            )
    data = coop.groupby([axis, "team"], as_index=False)["team_survival_plies_mean"].mean()
    result.figures.append(
        figures.metric_vs_factor(
            data, out / f"team_survival_{axis[2:]}.png", metric="team_survival_plies_mean", x=axis
        )
    )


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
