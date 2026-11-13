"""Tron multi-joueur et coalitions.

Couches (chacune ne dépend que de celles du dessous) :
``cli`` → ``viz``, ``experiments``, ``analysis`` → ``sim`` → ``agents`` → ``eval`` → ``core``.

L'API publique et stable est :mod:`tron.api` (``from tron.api import run_game``) ;
ses noms sont aussi accessibles depuis ``tron`` (``tron.run_game``), chargés à la
demande pour que ``import tron`` reste léger.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    api = importlib.import_module("tron.api")  # chargement à la demande
    if name in api.__all__:
        return getattr(api, name)
    raise AttributeError(f"module 'tron' has no attribute {name!r}")
