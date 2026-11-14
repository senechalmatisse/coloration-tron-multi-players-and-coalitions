"""Indicateurs et tests statistiques.

* intervalle de score de Wilson à 95 % : centre (p̂ + z²/2n)/(1 + z²/n), demi-largeur
  z/(1 + z²/n)·√(p̂(1 − p̂)/n + z²/4n²) ;
* agrégats par condition : taux de victoire de l'équipe suivie (celle du joueur 0 :
  le solitaire de l'étude A, l'équipe T0 de l'étude B), niveau de hasard
  |équipe| / n, rang de survie, durée, coût de calcul ;
* comparaisons appariées (mêmes graines, mêmes sièges) par le test exact de
  McNemar, avec correction de Holm.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Intervalle de Wilson pour ``successes`` succès sur ``n`` essais."""
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, min(p, center - half)), min(1.0, max(p, center + half))


def factor_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes de facteurs d'un fichier de résultats (préfixe ``f_``)."""
    return [c for c in df.columns if c.startswith("f_")]


def load_results(path: str) -> pd.DataFrame:
    """Lit ``results.csv`` et ajoute les colonnes dérivées utiles à l'analyse."""
    df = pd.read_csv(path)
    return enrich(df)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team_cols = [c for c in df.columns if c.startswith("team_p")]
    if team_cols and "focus_team" in df.columns:
        focus = df["focus_team"].astype(str)
        size = sum((df[c].astype(str) == focus).astype(int) for c in team_cols)
        df["focus_size"] = size
        df["chance"] = size / df["n_players"]
    if {"nodes_p0", "decisions_p0"} <= set(df.columns):
        dec = df["decisions_p0"].replace(0, np.nan)
        df["focus_nodes_per_decision"] = df["nodes_p0"] / dec
        df["focus_time_per_decision"] = df["time_p0"] / dec
        if "leaves_p0" in df.columns:  # colonnes ajoutées par P04 : absentes des anciens CSV
            df["focus_leaves_per_decision"] = pd.to_numeric(df["leaves_p0"], errors="coerce") / dec
        if "meandepth_p0" in df.columns:
            df["focus_mean_depth"] = pd.to_numeric(df["meandepth_p0"], errors="coerce")
    if "f_solo_depth" in df.columns and "n_players" in df.columns:
        df["solo_depth_rounds"] = df["f_solo_depth"] / df["n_players"]
    return df


def summarize(df: pd.DataFrame, by: Sequence[str] | None = None) -> pd.DataFrame:
    """Une ligne par condition : effectif, victoires, taux et IC de Wilson, rang, coût."""
    keys = list(by) if by is not None else factor_columns(df)
    if not keys:
        df = df.assign(_all=1)
        keys = ["_all"]
    rows: list[dict[str, Any]] = []
    for values, g in df.groupby(keys, sort=True, dropna=False):
        vals = values if isinstance(values, tuple) else (values,)
        n = len(g)
        wins = int(g["focus_win"].sum())
        lo, hi = wilson_interval(wins, n)
        row: dict[str, Any] = dict(zip(keys, vals, strict=True))
        row.update(
            n=n,
            wins=wins,
            win_rate=wins / n,
            ci_low=lo,
            ci_high=hi,
            chance=float(g["chance"].mean()) if "chance" in g else math.nan,
            mean_rank=float(g["rank_p0"].mean()) if "rank_p0" in g else math.nan,
            mean_moves=float(g["n_moves"].mean()),
            mean_duration=float(g["duration"].mean()),
        )
        if "focus_nodes_per_decision" in g:
            row["nodes_per_decision"] = float(g["focus_nodes_per_decision"].mean())
            row["time_per_decision"] = float(g["focus_time_per_decision"].mean())
        if "focus_leaves_per_decision" in g:
            row["leaves_per_decision"] = float(g["focus_leaves_per_decision"].mean())
        if "focus_mean_depth" in g:
            row["mean_depth"] = float(g["focus_mean_depth"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.drop(columns=["_all"], errors="ignore")


def holm(pvalues: Sequence[float]) -> list[float]:
    """Correction de Holm-Bonferroni (p-valeurs ajustées, ordre d'origine)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def mcnemar_exact(b: int, c: int) -> float:
    """p-valeur bilatérale exacte du test de McNemar (paires discordantes b et c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilcoxon_p(differences: Sequence[float]) -> float:
    """p-valeur bilatérale du test des rangs signés de Wilcoxon (1 si toutes nulles)."""
    diffs = [d for d in differences if not math.isnan(d) and d != 0.0]
    if len(diffs) < 2:
        return 1.0
    from scipy.stats import wilcoxon  # noqa: PLC0415 - import ponctuel

    try:
        return float(wilcoxon(diffs).pvalue)
    except ValueError:  # pragma: no cover - cas dégénéré
        return 1.0


def _effort(joined: pd.DataFrame, col: str) -> tuple[float, float]:
    lo, hi = joined.get(f"{col}_low"), joined.get(f"{col}_high")
    if lo is None or hi is None:
        return math.nan, math.nan
    d = (pd.to_numeric(hi, errors="coerce") - pd.to_numeric(lo, errors="coerce")).dropna()
    if d.empty:
        return math.nan, math.nan
    return float(d.mean()), wilcoxon_p(d.tolist())


def paired_comparisons(
    df: pd.DataFrame,
    factor: str,
    pair_keys: Sequence[str] = ("seed", "first_player"),
    reference: Any = None,
) -> pd.DataFrame:
    """Compare les niveaux de ``factor`` sur les mêmes parties (McNemar, Holm).

    Sans ``reference`` : niveaux consécutifs (comportement historique). Avec
    ``reference`` : chaque niveau contre ce niveau de référence (``low``). Les
    parties sont appariées par les autres facteurs, la graine et le premier
    joueur. Colonnes : b (victoire avec ``high`` seulement), c (avec ``low``
    seulement), p-valeur exacte de McNemar et correction de Holm (famille : ce
    facteur). Si les coûts sont présents, écarts moyens appariés (``high`` − ``low``)
    du temps et des nœuds par décision du joueur suivi, avec la p-valeur du test
    de Wilcoxon.
    """
    others = [c for c in factor_columns(df) if c != factor]
    levels = sorted(df[factor].dropna().unique(), key=_level_key)
    if reference is None:
        pairs = list(itertools.pairwise(levels))
    else:
        ref = next((lv for lv in levels if str(lv) == str(reference)), None)
        if ref is None:
            return pd.DataFrame()
        pairs = [(ref, lv) for lv in levels if lv is not ref and str(lv) != str(ref)]
    rows = []
    keys = [*others, *pair_keys]
    effort = [c for c in ("focus_time_per_decision", "focus_nodes_per_decision") if c in df]
    cols = ["focus_win", *effort]
    for lo, hi in pairs:
        a = df[df[factor] == lo].set_index(keys)[cols]
        b = df[df[factor] == hi].set_index(keys)[cols]
        joined = a.join(b, how="inner", lsuffix="_low", rsuffix="_high")
        n_b = int(((joined["focus_win_high"] == 1) & (joined["focus_win_low"] == 0)).sum())
        n_c = int(((joined["focus_win_high"] == 0) & (joined["focus_win_low"] == 1)).sum())
        row: dict[str, Any] = {
            "factor": factor,
            "low": lo,
            "high": hi,
            "pairs": len(joined),
            "high_only": n_b,
            "low_only": n_c,
            "diff": (n_b - n_c) / len(joined) if len(joined) else math.nan,
            "p_value": mcnemar_exact(n_b, n_c),
        }
        if effort:
            row["time_diff"], row["time_p_value"] = _effort(joined, "focus_time_per_decision")
            row["nodes_diff"], row["nodes_p_value"] = _effort(joined, "focus_nodes_per_decision")
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].tolist())
    return out


def _level_key(value: Any) -> tuple[int, float, str]:
    if isinstance(value, bool | np.bool_):
        return (0, float(value), "")
    if isinstance(value, int | float | np.integer | np.floating) and not math.isnan(value):
        return (0, float(value), "")
    return (1, 0.0, str(value))
