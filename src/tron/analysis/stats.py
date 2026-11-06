"""Indicateurs et tests statistiques (F7.1, §7.3).

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


def paired_comparisons(
    df: pd.DataFrame, factor: str, pair_keys: Sequence[str] = ("seed", "first_player")
) -> pd.DataFrame:
    """Compare deux à deux les niveaux consécutifs de ``factor`` sur les mêmes parties.

    Les parties sont appariées par les autres facteurs, la graine et le premier
    joueur. Renvoie b (gain avec le niveau haut seulement), c (avec le niveau bas
    seulement), la p-valeur exacte de McNemar et sa correction de Holm.
    """
    others = [c for c in factor_columns(df) if c != factor]
    levels = sorted(df[factor].dropna().unique())
    rows = []
    keys = [*others, *pair_keys]
    for lo, hi in itertools.pairwise(levels):
        a = df[df[factor] == lo].set_index(keys)["focus_win"]
        b = df[df[factor] == hi].set_index(keys)["focus_win"]
        joined = pd.concat([a.rename("low"), b.rename("high")], axis=1, join="inner")
        n_b = int(((joined["high"] == 1) & (joined["low"] == 0)).sum())
        n_c = int(((joined["high"] == 0) & (joined["low"] == 1)).sum())
        rows.append(
            {
                "factor": factor,
                "low": lo,
                "high": hi,
                "pairs": len(joined),
                "high_only": n_b,
                "low_only": n_c,
                "diff": (n_b - n_c) / len(joined) if len(joined) else math.nan,
                "p_value": mcnemar_exact(n_b, n_c),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].tolist())
    return out
