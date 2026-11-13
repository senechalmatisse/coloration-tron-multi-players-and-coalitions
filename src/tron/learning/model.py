"""Modèle linéaire de référence (P10) : valeur d'un état après coup, V = w · φ.

Le modèle s'exporte en JSON (format ``tron-model/1``) : poids, jeu de
caractéristiques, métadonnées d'entraînement. Il se recharge et joue avec NumPy
seul, sans aucune dépendance d'entraînement.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import tron
from tron.core.state import GameState
from tron.learning.features import FEATURE_NAMES, FEATURE_SET, FeatureExtractor

MODEL_FORMAT = "tron-model/1"
MODEL_KIND = "linear-afterstate"


class ModelError(ValueError):
    """Modèle illisible, incomplet ou incompatible."""


@dataclass(slots=True)
class LinearModel:
    """Poids d'une fonction de valeur linéaire sur les caractéristiques ``afterstate-v1``."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64).copy()
        if w.shape != (len(FEATURE_NAMES),):
            raise ModelError(f"{w.size} poids pour {len(FEATURE_NAMES)} caractéristiques")
        if not np.isfinite(w).all():
            raise ModelError("poids non finis")
        self.weights = w

    def value(self, phi: np.ndarray) -> float:
        return float(self.weights @ phi)

    def action_values(
        self,
        state: GameState,
        player: int,
        extractor: FeatureExtractor,
        moves: Sequence[int],
    ) -> dict[int, float]:
        """Valeur de l'état après chacun des coups ``moves`` de ``player``."""
        return {
            m: self.value(phi) for m, phi in extractor.action_features(state, player, moves).items()
        }

    def copy(self) -> LinearModel:
        return LinearModel(self.weights.copy(), dict(self.metadata))

    # -------------------------------------------------------- sérialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "kind": MODEL_KIND,
            "feature_set": FEATURE_SET,
            "features": list(FEATURE_NAMES),
            "weights": [float(w) for w in self.weights],
            "code_version": tron.__version__,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearModel:
        if not isinstance(data, Mapping):
            raise ModelError("modèle invalide : objet JSON attendu")
        if data.get("format") != MODEL_FORMAT:
            raise ModelError(f"format de modèle {data.get('format')!r} (attendu : {MODEL_FORMAT})")
        if data.get("kind") != MODEL_KIND or data.get("feature_set") != FEATURE_SET:
            raise ModelError(
                f"modèle {data.get('kind')!r} / {data.get('feature_set')!r} incompatible "
                f"(attendu : {MODEL_KIND} / {FEATURE_SET})"
            )
        if tuple(data.get("features", ())) != FEATURE_NAMES:
            raise ModelError("liste des caractéristiques incompatible avec ce code")
        raw = data.get("weights")
        if not isinstance(raw, list) or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in raw
        ):
            raise ModelError("poids absents ou invalides")
        meta = data.get("metadata") or {}
        if not isinstance(meta, Mapping):
            raise ModelError("métadonnées invalides")
        return cls(np.asarray(raw, dtype=np.float64), dict(meta))

    def save(self, path: str | Path) -> Path:
        """Écrit le modèle en JSON (écriture atomique)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, out)
        return out


def load_model(source: str | Path | Mapping[str, Any] | LinearModel) -> LinearModel:
    """Charge un modèle depuis un fichier JSON, un dictionnaire ou un modèle existant."""
    if isinstance(source, LinearModel):
        return source
    if isinstance(source, Mapping):
        return LinearModel.from_dict(source)
    path = Path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelError(f"modèle introuvable : {path}") from None
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} : JSON invalide ({exc.msg})") from exc
    try:
        return LinearModel.from_dict(data)
    except ModelError as exc:
        raise ModelError(f"{path} : {exc}") from exc


__all__ = ["MODEL_FORMAT", "MODEL_KIND", "LinearModel", "ModelError", "load_model"]
