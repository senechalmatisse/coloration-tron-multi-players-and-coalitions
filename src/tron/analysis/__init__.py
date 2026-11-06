"""Analyse statistique et figures à partir des fichiers CSV de résultats."""

from tron.analysis.regression import DEFAULT_FORMULA, RegressionResult, default_formula, fit_logit
from tron.analysis.report import AnalysisOutput, analyze
from tron.analysis.stats import (
    enrich,
    factor_columns,
    holm,
    load_results,
    mcnemar_exact,
    paired_comparisons,
    summarize,
    wilson_interval,
)

__all__ = [
    "DEFAULT_FORMULA",
    "AnalysisOutput",
    "RegressionResult",
    "analyze",
    "default_formula",
    "enrich",
    "factor_columns",
    "fit_logit",
    "holm",
    "load_results",
    "mcnemar_exact",
    "paired_comparisons",
    "summarize",
    "wilson_interval",
]
