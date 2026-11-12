"""Analyse statistique et figures à partir des fichiers CSV de résultats."""

from tron.analysis.cooperation import (
    METRIC_FIELDS,
    condition_summary,
    cooperation_summary,
    has_metrics,
    team_table,
)
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
    "METRIC_FIELDS",
    "AnalysisOutput",
    "RegressionResult",
    "analyze",
    "condition_summary",
    "cooperation_summary",
    "default_formula",
    "enrich",
    "factor_columns",
    "fit_logit",
    "has_metrics",
    "holm",
    "load_results",
    "mcnemar_exact",
    "paired_comparisons",
    "summarize",
    "team_table",
    "wilson_interval",
]
