"""Figures de la campagne (F7.2), régénérées à partir des CSV.

* taux de victoire du joueur suivi en fonction de sa profondeur : une courbe par
  taille de coalition, un graphique par taille de grille, barres d'erreur de
  Wilson, niveau de hasard en pointillés ;
* cartes de chaleur profondeur du solitaire × profondeur de la coalition (une
  rampe séquentielle bleue, échelle commune) ;
* coût de calcul (nœuds par décision, échelle logarithmique).

Charte : surface claire, marques fines (lignes de 2 px, marqueurs de 8 px cerclés),
grille en filet discret, légende systématique dès deux séries,
couleur attachée à la valeur du facteur et jamais à son rang.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tron.style import (
    BASELINE_LIGHT,
    CATEGORICAL_LIGHT,
    GRID_LIGHT,
    INK_MUTED,
    INK_PRIMARY_LIGHT,
    INK_SECONDARY_LIGHT,
    SEQUENTIAL_BLUE,
    SURFACE_LIGHT,
)

DPI = 150
PX = 72 / DPI  # 1 pixel en points

LABELS = {
    "f_solo_depth": "Profondeur du solitaire (plies)",
    "f_coalition_depth": "Profondeur de la coalition (plies)",
    "f_k": "Taille de la coalition k",
    "f_grid": "Grille",
    "f_depth_t0": "Profondeur de l'équipe T0 (plies)",
    "f_depth_t1": "Profondeur de l'équipe T1 (plies)",
    "f_composition": "Composition",
    "f_mode": "Mode",
    "f_coalition_mode": "Mode de la coalition",
    "f_solo_agent": "Algorithme du solitaire",
    "f_simulations": "Simulations MCTS par décision",
    "f_heuristic_name": "Heuristique",
    "f_solo_heuristic": "Heuristique du solitaire",
    "f_ablate": "Composante retirée (ablation)",
    "f_diplomacy_enabled": "Diplomatie active",
    "f_alliance_weight": "Poids d'alliance",
    "f_proposal_ttl_plies": "Durée de vie d'une proposition (demi-coups)",
    "f_adaptation_rate": "Taux d'adaptation λ",
    "f_w_territory": "Poids du territoire",
    "f_w_mobility": "Poids de la mobilité",
    "f_w_connectivity": "Poids des sorties",
    "f_w_safety": "Poids de la marge",
    "f_w_bottleneck": "Poids des passages",
}


def _plt() -> Any:
    import matplotlib  # noqa: PLC0415 - import tardif (mode sans affichage)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE_LIGHT,
            "axes.facecolor": SURFACE_LIGHT,
            "savefig.facecolor": SURFACE_LIGHT,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK_PRIMARY_LIGHT,
            "axes.labelcolor": INK_SECONDARY_LIGHT,
            "axes.edgecolor": BASELINE_LIGHT,
            "axes.linewidth": PX,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY_LIGHT,
            "ytick.labelcolor": INK_SECONDARY_LIGHT,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY_LIGHT,
        }
    )
    return plt


def label(col: str) -> str:
    return LABELS.get(col, col.removeprefix("f_"))


def level_text(col: str, value: Any) -> str:
    if col == "f_grid":
        side = str(value).strip("[]() ").split(",")
        return f"{side[0]}×{side[-1]}" if len(side) == 2 else f"{side[0]}×{side[0]}"
    if col == "f_k":
        return f"k = {value}"
    return f"{label(col)} = {value}"


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID_LIGHT, linewidth=PX, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)


def _header(
    fig: Any,
    title: str,
    handles: Sequence[Any] = (),
    labels: Sequence[str] = (),
    width_in: float = 4.0,
) -> None:
    """Titre aligné à gauche puis légende sur une ligne en dessous (jamais superposés)."""
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold")
    top = 0.90
    if handles:
        per_row = max(1, min(len(handles), int(width_in / 1.15)))
        rows = math.ceil(len(handles) / per_row)
        fig.legend(
            handles,
            labels,
            loc="upper left",
            ncols=per_row,
            bbox_to_anchor=(0.0, 0.93),
            handlelength=1.6,
            columnspacing=1.2,
        )
        top = 0.90 - 0.07 * rows
    fig.tight_layout(rect=(0, 0, 1, top))


def _colors(levels: Sequence[Any]) -> dict[Any, str]:
    """Couleur fixe par niveau (ordre trié), jamais recyclée au-delà de huit."""
    if len(levels) > len(CATEGORICAL_LIGHT):
        raise ValueError("plus de huit séries : regroupez-les ou découpez la figure")
    return {lv: CATEGORICAL_LIGHT[i] for i, lv in enumerate(levels)}


def winrate_vs_depth(
    summary: pd.DataFrame,
    out: str | Path,
    x: str = "f_solo_depth",
    series: str = "f_k",
    facet: str = "f_grid",
    title: str = "Taux de victoire du solitaire",
    logx: bool = False,
) -> Path:
    """Taux de victoire selon la profondeur : une courbe par niveau de ``series``.

    Un panneau par niveau de ``facet``. ``logx`` : abscisse logarithmique (base 2),
    utile pour un nombre de simulations.
    """
    plt = _plt()
    facets = sorted(summary[facet].unique(), key=_sort_key) if facet in summary else [None]
    levels = sorted(summary[series].unique(), key=_sort_key)
    colors = _colors(levels)
    fig, axes = plt.subplots(
        1, len(facets), figsize=(3.6 * len(facets) + 1.0, 3.9), sharey=True, squeeze=False
    )
    for ax, fv in zip(axes[0], facets, strict=True):
        sub = summary if fv is None else summary[summary[facet] == fv]
        for lv in levels:
            s = sub[sub[series] == lv].sort_values(x)
            if s.empty:
                continue
            c = colors[lv]
            yerr = [
                (s["win_rate"] - s["ci_low"]).clip(lower=0),
                (s["ci_high"] - s["win_rate"]).clip(lower=0),
            ]
            ax.errorbar(
                s[x],
                s["win_rate"],
                yerr=yerr,
                color=c,
                linewidth=2 * PX * 1.5,
                elinewidth=PX * 1.5,
                capsize=0,
                marker="o",
                markersize=8 * PX * 1.4,
                markeredgecolor=SURFACE_LIGHT,
                markeredgewidth=2 * PX,
                label=level_text(series, lv),
                zorder=3,
            )
            if "chance" in s and s["chance"].notna().any():
                ax.axhline(
                    float(s["chance"].mean()),
                    color=c,
                    linewidth=PX * 1.2,
                    linestyle=(0, (3, 3)),
                    alpha=0.7,
                    zorder=2,
                )
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{100 * v:.0f} %")
        ax.set_xlabel(label(x))
        if logx:
            ax.set_xscale("log", base=2)
        ticks = sorted(summary[x].unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        if fv is not None:
            ax.set_title(
                f"Grille {level_text(facet, fv)}" if facet == "f_grid" else level_text(facet, fv),
                loc="left",
            )
        _style(ax)
    axes[0][0].set_ylabel("Taux de victoire (IC de Wilson 95 %)")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    _header(
        fig,
        title + " (pointillés : niveau du hasard)",
        handles,
        labels_,
        width_in=fig.get_size_inches()[0],
    )
    return _save(fig, out)


def heatmaps(
    summary: pd.DataFrame,
    out: str | Path,
    x: str = "f_solo_depth",
    y: str = "f_coalition_depth",
    row: str = "f_k",
    col: str = "f_grid",
    value: str = "win_rate",
    annotate: bool = False,
) -> Path:
    """Cartes de chaleur ``x`` × ``y`` (une par couple ``row`` × ``col``), échelle 0-100 %."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    cmap = LinearSegmentedColormap.from_list("bleu", list(SEQUENTIAL_BLUE))
    rows = sorted(summary[row].unique(), key=_sort_key) if row in summary else [None]
    cols = sorted(summary[col].unique(), key=_sort_key) if col in summary else [None]
    xs = sorted(summary[x].unique(), key=_sort_key)
    ys = sorted(summary[y].unique(), key=_sort_key)
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(2.8 * len(cols) + 1.6, 1.9 * len(rows) + 1.0),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = axes[i][j]
            sub = summary
            if rv is not None:
                sub = sub[sub[row] == rv]
            if cv is not None:
                sub = sub[sub[col] == cv]
            grid = sub.pivot_table(index=y, columns=x, values=value).reindex(index=ys, columns=xs)
            image = ax.imshow(
                grid.to_numpy(), cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto"
            )
            if annotate:
                for (yi, xi), v in pd.DataFrame(grid.to_numpy()).stack().items():
                    ax.text(
                        xi,
                        yi,
                        f"{100 * v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#ffffff" if v > 0.55 else INK_PRIMARY_LIGHT,
                    )
            ax.set_xticks(range(len(xs)), [str(v) for v in xs])
            ax.set_yticks(range(len(ys)), [str(v) for v in ys])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0 and cv is not None:
                ax.set_title(f"Grille {level_text(col, cv)}", loc="left")
            if j == 0 and rv is not None:
                ax.set_ylabel(level_text(row, rv), color=INK_PRIMARY_LIGHT)
    if image is not None:
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.04, pad=0.03, shrink=0.8)
        bar.outline.set_visible(False)
        bar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{100 * v:.0f} %")
        bar.set_label("Taux de victoire du solitaire", color=INK_SECONDARY_LIGHT)
    fig.supxlabel(label(x), fontsize=9, color=INK_SECONDARY_LIGHT)
    fig.supylabel(label(y), fontsize=9, color=INK_SECONDARY_LIGHT)
    fig.suptitle(
        "Taux de victoire : profondeur du solitaire × profondeur de la coalition",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    return _save(fig, out)


def cost_vs_depth(
    summary: pd.DataFrame,
    out: str | Path,
    x: str = "f_solo_depth",
    series: str = "f_k",
    facet: str = "f_grid",
) -> Path:
    """Nœuds explorés par décision du joueur suivi (échelle logarithmique)."""
    plt = _plt()
    facets = sorted(summary[facet].unique(), key=_sort_key) if facet in summary else [None]
    levels = sorted(summary[series].unique(), key=_sort_key)
    colors = _colors(levels)
    fig, axes = plt.subplots(
        1, len(facets), figsize=(3.6 * len(facets) + 1.0, 3.7), sharey=True, squeeze=False
    )
    for ax, fv in zip(axes[0], facets, strict=True):
        sub = summary if fv is None else summary[summary[facet] == fv]
        for lv in levels:
            s = sub[sub[series] == lv].groupby(x, as_index=False)["nodes_per_decision"].mean()
            if s.empty:
                continue
            ax.plot(
                s[x],
                s["nodes_per_decision"],
                color=colors[lv],
                linewidth=2 * PX * 1.5,
                marker="o",
                markersize=8 * PX * 1.4,
                markeredgecolor=SURFACE_LIGHT,
                markeredgewidth=2 * PX,
                label=level_text(series, lv),
            )
        ax.set_yscale("log")
        ax.set_xlabel(label(x))
        if fv is not None:
            ax.set_title(f"Grille {level_text(facet, fv)}", loc="left")
        _style(ax)
    axes[0][0].set_ylabel("Nœuds par décision")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    _header(fig, "Coût de calcul du solitaire", handles, labels_, width_in=fig.get_size_inches()[0])
    return _save(fig, out)


METRIC_LABELS = {
    "cooperation_proxy_mean": "Indice de coopération (proxy, 0 à 1)",
    "intra_team_distance_mean_mean": "Distance moyenne entre coéquipiers (cases)",
    "team_survival_plies_mean": "Survie de l'équipe (demi-coups)",
    "territory_share_mean_mean": "Part de territoire de l'équipe",
    "collective_vs_individual_gap_mean": "Écart collectif / individuel (0 à 1)",
    "win_rate": "Taux de victoire",
}


def metric_vs_factor(
    summary: pd.DataFrame,
    out: str | Path,
    metric: str = "cooperation_proxy_mean",
    x: str = "f_k",
    series: str = "team",
    title: str | None = None,
) -> Path:
    """Moyenne d'une métrique selon un facteur, une courbe par niveau de ``series``.

    Les niveaux de ``x`` sont placés à intervalles réguliers dans leur ordre
    naturel (numérique si possible), ce qui accepte aussi les grilles ``[l, h]``.
    """
    plt = _plt()
    data = summary[summary[metric].notna()]
    xs = sorted(data[x].unique(), key=_sort_key)
    pos = {v: i for i, v in enumerate(xs)}
    levels = sorted(data[series].unique(), key=_sort_key)
    colors = _colors(levels)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for lv in levels:
        s = data[data[series] == lv].groupby(x, as_index=False)[metric].mean()
        s = s.assign(_pos=s[x].map(pos)).sort_values("_pos")
        ax.plot(
            s["_pos"],
            s[metric],
            color=colors[lv],
            linewidth=2 * PX * 1.5,
            marker="o",
            markersize=8 * PX * 1.4,
            markeredgecolor=SURFACE_LIGHT,
            markeredgewidth=2 * PX,
            label=str(lv) if series == "team" else level_text(series, lv),
        )
    ax.set_xticks(range(len(xs)), [level_text(x, v).split(" = ")[-1] for v in xs])
    ax.set_xlabel(label(x))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    _style(ax)
    handles, labels_ = ax.get_legend_handles_labels()
    _header(
        fig,
        title or f"{METRIC_LABELS.get(metric, metric)} selon {label(x).lower()}",
        handles,
        labels_,
        width_in=fig.get_size_inches()[0],
    )
    return _save(fig, out)


def level_comparison(
    pooled: pd.DataFrame,
    out: str | Path,
    factor: str,
    reference: Any = None,
    title: str | None = None,
) -> Path:
    """Taux de victoire (IC de Wilson) par niveau d'un facteur, niveau de référence en évidence.

    ``pooled`` : une ligne par niveau (``summarize(df, by=[factor])``). La ligne
    pointillée marque le taux du niveau de référence.
    """
    plt = _plt()
    levels = sorted(pooled[factor].unique(), key=_sort_key)
    refs = [lv for lv in levels if reference is not None and str(lv) == str(reference)]
    levels = refs + [lv for lv in levels if lv not in refs]  # référence en premier
    fig, ax = plt.subplots(figsize=(max(4.2, 0.95 * len(levels) + 1.6), 3.7))
    ref_rate = None
    for i, lv in enumerate(levels):
        row = pooled[pooled[factor] == lv].iloc[0]
        is_ref = reference is not None and str(lv) == str(reference)
        color = CATEGORICAL_LIGHT[1] if is_ref else CATEGORICAL_LIGHT[0]
        if is_ref:
            ref_rate = float(row["win_rate"])
        ax.errorbar(
            [i],
            [row["win_rate"]],
            yerr=[
                [max(0.0, row["win_rate"] - row["ci_low"])],
                [max(0.0, row["ci_high"] - row["win_rate"])],
            ],
            color=color,
            elinewidth=PX * 1.5,
            capsize=0,
            marker="o",
            markersize=9 * PX * 1.4,
            markeredgecolor=SURFACE_LIGHT,
            markeredgewidth=2 * PX,
            zorder=3,
        )
    if ref_rate is not None:
        ax.axhline(ref_rate, color=CATEGORICAL_LIGHT[1], linewidth=PX * 1.2, linestyle=(0, (3, 3)))
    ax.set_xticks(range(len(levels)), [str(v) for v in levels])
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{100 * v:.0f} %")
    ax.set_xlabel(label(factor))
    ax.set_ylabel("Taux de victoire (IC de Wilson 95 %)")
    _style(ax)
    note = " (pointillés : niveau de référence)" if ref_rate is not None else ""
    _header(
        fig,
        (title or f"Taux de victoire : {label(factor)}") + note,
        width_in=fig.get_size_inches()[0],
    )
    return _save(fig, out)


def _sort_key(value: Any) -> tuple[int, float, str]:
    if isinstance(value, int | float) and not (isinstance(value, float) and math.isnan(value)):
        return (0, float(value), "")
    text = str(value)
    try:
        return (0, float(text.strip("[]() ").split(",")[0]), text)
    except ValueError:
        return (1, 0.0, text)


def _save(fig: Any, out: str | Path) -> Path:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
