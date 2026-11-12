"""Compilation JIT optionnelle des noyaux de calcul.

Les noyaux (parcours en largeur de Voronoi) sont écrits dans un style procédural
compatible Numba : tableaux NumPy, entiers, boucles simples. Si Numba est absent
ou si ``NUMBA_DISABLE_JIT=1``, ils s'exécutent en Python pur, avec le même
résultat mais plus lentement.
"""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Callable
from typing import Any, cast

try:  # pragma: no cover - dépend de l'environnement
    import numba as _numba
except ImportError:  # pragma: no cover
    _numba = None

try:  # pragma: no cover - dépend de l'environnement
    from numba.core.errors import NumbaError as _NumbaError
except ImportError:  # pragma: no cover
    _NumbaError = Exception

HAVE_NUMBA: bool = _numba is not None
JIT_ENABLED: bool = HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT", "0") in ("", "0")


def njit[F: Callable[..., Any]](func: F) -> F:
    """Applique ``numba.njit(cache=True)`` si Numba est disponible, l'identité sinon."""
    if _numba is None:
        return func
    return _numba.njit(cache=True, nogil=True)(func)


def safe_njit[F: Callable[..., Any]](func: F) -> F:
    """Comme :func:`njit`, avec repli en Python pur si Numba refuse de compiler le noyau.

    La compilation a lieu au premier appel. Si elle échoue (erreur de typage
    Numba), un ``RuntimeWarning`` est émis et le noyau s'exécute ensuite en
    Python pur : même résultat, plus lentement. Les erreurs d'exécution
    ordinaires ne sont pas masquées (le repli les relance).
    """
    if not JIT_ENABLED or _numba is None:
        return func
    compiled = _numba.njit(cache=True, nogil=True)(func)  # pragma: no cover - avec Numba
    impl: list[Callable[..., Any]] = [compiled]  # pragma: no cover
    checked = [False]  # pragma: no cover

    @functools.wraps(func)
    def wrapper(*args: Any) -> Any:  # pragma: no cover - avec Numba
        if checked[0]:
            return impl[0](*args)
        try:
            out = impl[0](*args)
        except _NumbaError as exc:
            warnings.warn(
                f"noyau {func.__name__} non compilé par Numba ({type(exc).__name__}) : "
                "exécution en Python pur",
                RuntimeWarning,
                stacklevel=2,
            )
            impl[0] = func
            out = func(*args)
        checked[0] = True
        return out

    return cast(F, wrapper)  # pragma: no cover


def backend_name() -> str:
    """Nom du moteur d'exécution des noyaux (pour les journaux et le benchmark)."""
    if JIT_ENABLED:
        return f"numba {_numba.__version__}"
    return "python"
