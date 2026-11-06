"""Régression logistique de la victoire (F7.3).

Avec statsmodels, le modèle est ajusté par ``smf.logit`` avec des erreurs
standard groupées par graine (parties appariées). Sans statsmodels, un
ajustement IRLS en NumPy fournit les mêmes coefficients et une variance robuste
« sandwich » groupée. Le repli comprend les termes ``a``, ``log2(a)``, ``a:b`` et
``a*b`` (= a + b + a:b).

Les facteurs sont désignés sans leur préfixe ``f_`` (``solo_depth``, ``k``…).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log2 = np.log2  # accessible aux formules statsmodels (patsy évalue dans ce module)

DEFAULT_FORMULA = (
    "focus_win ~ log2(solo_depth) * k + log2(solo_depth) * grid + log2(coalition_depth)"
)


def default_formula(df: pd.DataFrame) -> str:
    """Formule par défaut restreinte aux facteurs qui varient effectivement."""
    data = prepare(df)

    def varies(col: str) -> bool:
        return col in data.columns and data[col].nunique() > 1

    terms = []
    if varies("solo_depth"):
        terms.append("log2(solo_depth)")
    for col in ("k", "grid"):
        if varies(col):
            terms.append(col)
            if varies("solo_depth"):
                terms.append(f"log2(solo_depth):{col}")
    if varies("coalition_depth"):
        terms.append("log2(coalition_depth)")
    if not terms:
        raise ValueError("aucun facteur ne varie : pas de régression possible")
    return "focus_win ~ " + " + ".join(terms)


@dataclass
class RegressionResult:
    table: pd.DataFrame
    n: int
    clusters: int
    backend: str
    formula: str
    converged: bool

    def to_text(self) -> str:
        lines = [
            f"Régression logistique ({self.backend}) : {self.formula}",
            f"{self.n} parties, {self.clusters} groupes (graines), convergence : {self.converged}",
            "",
            self.table.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Copie avec les facteurs sans préfixe ``f_`` (grille réduite à son côté)."""
    data = df.rename(columns={c: c[2:] for c in df.columns if c.startswith("f_")}).copy()
    if "grid" in data.columns:
        data["grid"] = data["grid"].map(_grid_side)
    return data


def _grid_side(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip("[]() ")
    return float(text.split(",")[0])


def fit_logit(
    df: pd.DataFrame,
    formula: str = DEFAULT_FORMULA,
    cluster: str = "seed",
    force_numpy: bool = False,
) -> RegressionResult:
    """Ajuste le modèle logistique ``formula`` sur les résultats ``df``."""
    data = prepare(df)
    if not force_numpy:
        try:
            import statsmodels.formula.api as smf  # noqa: PLC0415 - dépendance optionnelle
        except ImportError:
            pass
        else:
            groups = pd.factorize(data[cluster])[0]
            model = smf.logit(formula, data=data).fit(
                disp=False, cov_type="cluster", cov_kwds={"groups": groups}
            )
            ci = model.conf_int()
            table = pd.DataFrame(
                {
                    "term": model.params.index,
                    "coef": model.params.values,
                    "std_err": model.bse.values,
                    "z": model.tvalues.values,
                    "p_value": model.pvalues.values,
                    "ci_low": ci[0].values,
                    "ci_high": ci[1].values,
                }
            )
            return RegressionResult(
                table,
                int(model.nobs),
                len(set(groups)),
                "statsmodels",
                formula,
                bool(model.mle_retvals["converged"]),
            )
    return _fit_numpy(data, formula, cluster)


# ------------------------------------------------------------------ repli
def _parse(formula: str) -> tuple[str, list[str]]:
    left, right = (s.strip() for s in formula.split("~"))
    terms: list[str] = []
    for chunk in right.split("+"):
        chunk = chunk.strip()
        if "*" in chunk:
            parts = [p.strip() for p in chunk.split("*")]
            if len(parts) != 2:
                raise ValueError("le repli NumPy ne gère que des produits a * b")
            new = [parts[0], parts[1], f"{parts[0]}:{parts[1]}"]
        else:
            new = [chunk]
        for t in new:
            if t not in terms:
                terms.append(t)
    return left, terms


def _column(data: pd.DataFrame, name: str) -> np.ndarray:
    name = name.strip()
    if name.startswith("log2(") and name.endswith(")"):
        return np.log2(data[name[5:-1]].to_numpy(dtype=float))
    if name.startswith("C("):
        raise ValueError("variables catégorielles C(...) : installez statsmodels")
    return data[name].to_numpy(dtype=float)


def _fit_numpy(data: pd.DataFrame, formula: str, cluster: str) -> RegressionResult:
    target, terms = _parse(formula)
    y = data[target].to_numpy(dtype=float)
    cols = [np.ones(len(data))]
    for term in terms:
        prod = np.ones(len(data))
        for factor in term.split(":"):
            prod = prod * _column(data, factor)
        cols.append(prod)
    x = np.column_stack(cols)
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(100):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        hess = x.T @ (x * w[:, None])
        step = np.linalg.solve(hess, x.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    bread = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    groups = pd.factorize(data[cluster])[0]
    scores = x * (y - p)[:, None]
    meat = np.zeros((x.shape[1], x.shape[1]))
    n_groups = int(groups.max()) + 1
    for g in range(n_groups):
        s = scores[groups == g].sum(axis=0)
        meat += np.outer(s, s)
    n = len(y)
    k = x.shape[1]
    correction = (n_groups / (n_groups - 1)) * ((n - 1) / (n - k)) if n_groups > 1 else 1.0
    cov = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = [math.erfc(abs(v) / math.sqrt(2)) for v in z]
    table = pd.DataFrame(
        {
            "term": ["Intercept", *terms],
            "coef": beta,
            "std_err": se,
            "z": z,
            "p_value": pvals,
            "ci_low": beta - 1.959963984540054 * se,
            "ci_high": beta + 1.959963984540054 * se,
        }
    )
    return RegressionResult(table, n, n_groups, "numpy-irls", formula, converged)
