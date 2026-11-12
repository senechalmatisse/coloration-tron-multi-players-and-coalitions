"""Règles de partie et modes de jeu (P09).

Les règles sont décrites par la configuration (clé YAML ``rules``), jamais par
des conditions codées dans les agents. Le mode historique (``classic``) reste le
mode par défaut : une partie classique se déroule exactement comme avant.

Séparation des responsabilités :

* **déplacements** : inchangés dans tous les modes (``GameState.legal_moves`` et
  ``apply``) ;
* **conditions de victoire** : portée de la victoire (équipe ou joueur) et
  conditions propres au mode ;
* **objectifs secondaires** : décrits par le mode (survie individuelle…).

Modes :

* ``classic`` : la partie s'arrête quand il ne reste qu'une équipe en vie ;
* ``survival`` : chaque joueur est classé selon sa durée de survie. Avec
  ``victory_scope: team`` la règle de fin reste celle des équipes ; avec
  ``victory_scope: individual`` la partie continue tant que deux joueurs vivent
  (coéquipiers compris) et le dernier survivant gagne. Les résultats donnent
  explicitement le vainqueur individuel et l'ordre de survie.

Les équipes de victoire (:attr:`GameConfig.victory_team_index`) sont celles qui
décident de la fin et du vainqueur ; les équipes de la configuration
(``team_index``) restent celles des préférences sociales des agents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MODES = ("classic", "survival")
VICTORY_SCOPES = ("team", "individual")


class RulesError(ValueError):
    """Règles de partie invalides."""


@dataclass(frozen=True, slots=True)
class RulesSpec:
    """Règles d'une partie : mode et portée de la victoire."""

    mode: str = "classic"
    victory_scope: str = "team"

    @property
    def individual(self) -> bool:
        """Vrai si la victoire est individuelle (mode survie, portée ``individual``)."""
        return self.mode == "survival" and self.victory_scope == "individual"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise RulesError(f"mode de jeu inconnu : {self.mode} (attendu : {MODES})")
        if self.victory_scope not in VICTORY_SCOPES:
            raise RulesError(
                f"portée de victoire inconnue : {self.victory_scope} (attendu : {VICTORY_SCOPES})"
            )
        if self.mode != "survival" and self.victory_scope != "team":
            raise RulesError("victory_scope: individual n'existe qu'en mode survival")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "victory_scope": self.victory_scope}


_RULES_KEYS = frozenset({"mode", "victory_scope"})


def parse_rules(raw: Any) -> RulesSpec:
    """Lit la clé ``rules`` d'une configuration (absente : mode classique)."""
    if raw is None:
        return RulesSpec()
    if isinstance(raw, str):
        return RulesSpec(mode=raw)
    if not isinstance(raw, Mapping):
        raise RulesError(f"règles invalides : {raw!r}")
    unknown = set(raw) - _RULES_KEYS
    if unknown:
        raise RulesError(f"clés de règles inconnues : {sorted(unknown)}")
    return RulesSpec(
        mode=str(raw.get("mode", "classic")),
        victory_scope=str(raw.get("victory_scope", "team")),
    )


__all__ = ["MODES", "VICTORY_SCOPES", "RulesError", "RulesSpec", "parse_rules"]
