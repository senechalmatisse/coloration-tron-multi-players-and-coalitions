"""Interface des heuristiques et registre (extensibilité, §4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from tron.core.config import HeuristicSpec
from tron.core.state import GameState


class Heuristic(ABC):
    """Évaluation d'un état non terminal par un vecteur de parts h (F2.2).

    ``shares`` renvoie h(i) ≥ 0 pour chaque joueur, nul pour les joueurs
    éliminés, avec Σ h(i) = 1 sur les joueurs vivants.
    """

    name: ClassVar[str] = "?"

    def __init__(self, width: int, height: int, n_players: int) -> None:
        self.width = width
        self.height = height
        self.n_players = n_players

    @abstractmethod
    def shares(self, state: GameState) -> list[float]:
        """Vecteur h de l'état (une composante par joueur)."""


HEURISTICS: dict[str, type[Heuristic]] = {}


def register_heuristic(name: str) -> Callable[[type[Heuristic]], type[Heuristic]]:
    """Décorateur d'enregistrement d'une heuristique sous ``name``."""

    def deco(cls: type[Heuristic]) -> type[Heuristic]:
        cls.name = name
        HEURISTICS[name] = cls
        return cls

    return deco


def make_heuristic(
    spec: HeuristicSpec | str, width: int, height: int, n_players: int, **overrides: Any
) -> Heuristic:
    """Instancie l'heuristique décrite par ``spec`` (nom et paramètres)."""
    if isinstance(spec, str):
        spec = HeuristicSpec(name=spec)
    try:
        cls = HEURISTICS[spec.name]
    except KeyError:
        raise ValueError(
            f"heuristique inconnue : {spec.name} (connues : {sorted(HEURISTICS)})"
        ) from None
    params = {**spec.params, **overrides}
    try:
        return cls(width, height, n_players, **params)
    except TypeError as exc:
        raise ValueError(f"paramètres invalides pour l'heuristique {spec.name} : {exc}") from exc


@register_heuristic("uniform")
class UniformHeuristic(Heuristic):
    """Parts égales entre joueurs vivants (référence pour les tests de recherche)."""

    def shares(self, state: GameState) -> list[float]:
        alive = state.alive_players()
        share = 1.0 / len(alive) if alive else 0.0
        return [share if state.is_alive(p) else 0.0 for p in range(state.n)]
