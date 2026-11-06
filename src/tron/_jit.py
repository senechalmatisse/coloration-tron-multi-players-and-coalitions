"""Compilation JIT optionnelle des noyaux de calcul.

Les noyaux (parcours en largeur de Voronoi) sont écrits dans un style procédural
compatible Numba : tableaux NumPy, entiers, boucles simples. Si Numba est absent
ou si ``NUMBA_DISABLE_JIT=1``, ils s'exécutent en Python pur, avec le même
résultat mais plus lentement.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - dépend de l'environnement
    import numba as _numba
except ImportError:  # pragma: no cover
    _numba = None

HAVE_NUMBA: bool = _numba is not None
JIT_ENABLED: bool = HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT", "0") in ("", "0")


def njit[F: Callable[..., Any]](func: F) -> F:
    """Applique ``numba.njit(cache=True)`` si Numba est disponible, l'identité sinon."""
    if _numba is None:
        return func
    return _numba.njit(cache=True, nogil=True)(func)


def backend_name() -> str:
    """Nom du moteur d'exécution des noyaux (pour les journaux et le benchmark)."""
    if JIT_ENABLED:
        return f"numba {_numba.__version__}"
    return "python"
