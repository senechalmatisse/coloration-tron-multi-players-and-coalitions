"""Agrégation des métriques de coopération (P02) à partir des fichiers de résultats.

Fonctions pures sur des ``DataFrame`` pandas, sans dépendance au moteur, à Numba
ni à l'affichage. Les définitions des métriques sont dans :mod:`tron.sim.metrics`.

* :func:`team_table` : format long, une ligne par (partie, équipe) ;
* :func:`cooperation_summary` : agrégat par condition expérimentale et par équipe ;
* :func:`condition_summary` : agrégat par condition, équipes d'au moins deux joueurs ;
* :func:`has_metrics` : faux pour un CSV produit avant l'introduction des métriques
  (les tables sont alors vides et l'analyse historique reste inchangée).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from tron.analysis.stats import factor_columns, wilson_interval
from tron.schema import MAX_PLAYERS, MAX_TEAMS, TEAM_METRIC_COLUMNS

METRIC_FIELDS: tuple[str, ...] = tuple(field for _, field in TEAM_METRIC_COLUMNS)
"""Métriques par équipe, dans l'ordre du schéma CSV."""

_CONTEXT = (
    "plan",
    "condition",
    "game",
    "seed",
    "first_player",
    "winning_team",
    "n_players",
    "width",
    "height",
)


def has_metrics(df: pd.DataFrame) -> bool:
    """Vrai si le fichier contient des métriques de coopération renseignées."""
    if "metrics_version" not in df.columns:
        return False
    return bool(pd.to_numeric(df["metrics_version"], errors="coerce").notna().any())


def team_table(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (partie, équipe) : facteurs, équipe, taille, victoire et métriques.

    Colonnes ajoutées : ``team``, ``team_index`` (0 = équipe du joueur 0),
    ``focus`` (équipe 0), ``team_size``, ``won`` et une colonne par métrique de
    :data:`METRIC_FIELDS` (NaN si non calculable).
    """
    if not has_metrics(df):
        return pd.DataFrame()
    keep = [*factor_columns(df), *(c for c in _CONTEXT if c in df.columns)]
    player_cols = [f"team_p{i}" for i in range(MAX_PLAYERS) if f"team_p{i}" in df.columns]
    parts = []
    for t in range(MAX_TEAMS):
        name_col = f"team_t{t}"
        if name_col not in df.columns:
            break
        names = df[name_col]
        mask = names.notna() & (names.astype(str) != "") & (names.astype(str) != "nan")
        if not mask.any():
            continue
        sub = df.loc[mask, keep].copy()
        team = names[mask].astype(str)
        sub["team"] = team.to_numpy()
        sub["team_index"] = t
        sub["focus"] = t == 0
        size = np.zeros(int(mask.sum()), dtype=int)
        for col in player_cols:
            size += (df.loc[mask, col].astype(str).to_numpy() == team.to_numpy()).astype(int)
        sub["team_size"] = size
        if "winning_team" in df.columns:
            won = df.loc[mask, "winning_team"].astype(str).to_numpy() == team.to_numpy()
            sub["won"] = won.astype(int)
        for prefix, field in TEAM_METRIC_COLUMNS:
            col = f"{prefix}_t{t}"
            values = df.loc[mask, col] if col in df.columns else pd.Series(np.nan, index=sub.index)
            sub[field] = pd.to_numeric(values, errors="coerce").to_numpy()
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _aggregate(g: pd.DataFrame) -> dict[str, Any]:
    n = len(g)
    row: dict[str, Any] = {"n": n}
    if "won" in g:
        wins = int(g["won"].sum())
        lo, hi = wilson_interval(wins, n)
        row.update(win_rate=wins / n, ci_low=lo, ci_high=hi)
    for field in METRIC_FIELDS:
        col = g[field]
        row[f"{field}_mean"] = float(col.mean()) if col.notna().any() else np.nan
    row["cooperation_proxy_n"] = int(g["cooperation_proxy"].notna().sum())
    row["ally_collisions_total"] = int(g["ally_collisions"].fillna(0).sum())
    return row


def cooperation_summary(df: pd.DataFrame, by: Sequence[str] | None = None) -> pd.DataFrame:
    """Agrégat par condition (facteurs ``f_*`` par défaut) et par équipe.

    Une ligne par (condition, équipe) : effectif, taille d'équipe, taux de
    victoire et intervalle de Wilson, moyenne de chaque métrique sur les parties
    où elle est calculable, nombre total de collisions entre alliés.
    """
    table = team_table(df)
    if table.empty:
        return table
    keys = [*(list(by) if by is not None else factor_columns(df)), "team"]
    rows = []
    for values, g in table.groupby(keys, sort=True, dropna=False):
        vals = values if isinstance(values, tuple) else (values,)
        row: dict[str, Any] = dict(zip(keys, vals, strict=True))
        row["team_size"] = int(g["team_size"].max())
        row["focus"] = bool(g["focus"].any())
        row.update(_aggregate(g))
        rows.append(row)
    return pd.DataFrame(rows)


def condition_summary(df: pd.DataFrame, by: Sequence[str] | None = None) -> pd.DataFrame:
    """Agrégat par condition, sur les équipes d'au moins deux joueurs (coalitions)."""
    table = team_table(df)
    if table.empty:
        return table
    table = table[table["team_size"] >= 2]
    if table.empty:
        return pd.DataFrame()
    keys = list(by) if by is not None else factor_columns(df)
    if not keys:
        table = table.assign(_all=1)
        keys = ["_all"]
    rows = []
    for values, g in table.groupby(keys, sort=True, dropna=False):
        vals = values if isinstance(values, tuple) else (values,)
        row: dict[str, Any] = dict(zip(keys, vals, strict=True))
        row.update(_aggregate(g))
        rows.append(row)
    return pd.DataFrame(rows).drop(columns=["_all"], errors="ignore")


__all__ = [
    "METRIC_FIELDS",
    "condition_summary",
    "cooperation_summary",
    "has_metrics",
    "team_table",
]
