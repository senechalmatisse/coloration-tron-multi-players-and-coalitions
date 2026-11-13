"""Apprentissage par renforcement (P10) : environnement, caractéristiques, modèle, entraînement.

NumPy suffit : aucun cadre d'apprentissage n'est requis. Les sous-modules sont
chargés à la demande, pour que l'agent ``learned`` (qui n'a besoin que du modèle
et des caractéristiques) n'importe ni l'environnement ni l'entraînement.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "FEATURE_NAMES": "features",
    "FEATURE_SET": "features",
    "FeatureExtractor": "features",
    "ObservationSpec": "features",
    "observe": "features",
    "LinearModel": "model",
    "ModelError": "model",
    "load_model": "model",
    "IllegalActionError": "env",
    "TronEnv": "env",
    "TrainingSpec": "train",
    "evaluate_model": "train",
    "train": "train",
}

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SET",
    "FeatureExtractor",
    "IllegalActionError",
    "LinearModel",
    "ModelError",
    "ObservationSpec",
    "TrainingSpec",
    "TronEnv",
    "evaluate_model",
    "load_model",
    "observe",
    "train",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'tron.learning' has no attribute {name!r}")
    return getattr(importlib.import_module(f"tron.learning.{module}"), name)
