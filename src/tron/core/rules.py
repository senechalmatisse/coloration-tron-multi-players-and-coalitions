"""Règles de partie et modes de jeu (P09).

Les règles sont décrites par la configuration (clé YAML ``rules``), jamais par
des conditions codées dans les agents. Le mode historique (``classic``) reste le
mode par défaut : une partie classique se déroule exactement comme avant.

Séparation des responsabilités :

* **déplacements** : inchangés dans tous les modes (``GameState.legal_moves`` et
  ``apply``) ;
* **conditions de victoire** : portée de la victoire (équipe ou joueur) et
  conditions propres au mode (score cible, durée) ;
* **objectifs secondaires** : décrits par le mode (survie, points de zone).

Modes :

* ``classic`` : la partie s'arrête quand il ne reste qu'une équipe en vie ;
* ``survival`` : chaque joueur est classé selon sa durée de survie. Avec
  ``victory_scope: team`` la règle de fin reste celle des équipes ; avec
  ``victory_scope: individual`` la partie continue tant que deux joueurs vivent
  (coéquipiers compris) et le dernier survivant gagne ;
* ``zone_control`` : des zones rectangulaires rapportent des points. Règle V1
  (``zone_rule: head``) : à la fin de chaque tour complet, une zone rapporte un
  point à l'équipe de victoire qui y a, seule, au moins une tête vivante (zone
  vide ou disputée : aucun point). Fin de partie, par ordre de priorité :
  (1) une équipe atteint ``target_score`` — le plus haut score gagne ; (2) une
  seule équipe en vie — elle gagne ; (3) ``max_rounds`` tours complets joués —
  le plus haut score gagne. À égalité de score en (1) ou (3), la seule équipe
  encore en vie parmi les meneuses gagne s'il n'en reste qu'une ; sinon, égalité.

Un **tour complet** se termine quand le joueur suivant à jouer revient, dans
l'ordre cyclique, au premier joueur de la partie ou avant lui : les tours sont
comptés depuis ``first_player``, ce qui garde la rotation des sièges équitable.

Les équipes de victoire (``GameConfig.victory_team_index``) décident de la fin et
du vainqueur ; les équipes de la configuration (``team_index``) restent celles
des préférences sociales des agents.

Les données propres aux règles (tours complets, scores) font partie de l'état :
``GameState`` les tient à jour dans ``apply``/``undo``, les copie, les inclut dans
la clé de Zobrist et dans ses instantanés. Le moteur :class:`RulesEngine` est sans
état propre ; il n'existe que pour les modes qui en ont besoin (sinon, aucun coût).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from tron.core.state import GameState

MODES = ("classic", "survival", "zone_control")
VICTORY_SCOPES = ("team", "individual")
ZONE_RULES = ("head",)
MAX_COUNTER = 255
"""Plafond des compteurs de tours et de scores (la clé de Zobrist les code sur un octet)."""


class RulesError(ValueError):
    """Règles de partie invalides."""


@dataclass(frozen=True, slots=True)
class ZoneSpec:
    """Zone rectangulaire : coin (x, y) en haut à gauche, largeur et hauteur en cases."""

    x: int
    y: int
    width: int = 1
    height: int = 1

    def cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.y, self.y + self.height)
            for x in range(self.x, self.x + self.width)
        ]

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class RulesSpec:
    """Règles d'une partie : mode, portée de la victoire et paramètres du mode."""

    mode: str = "classic"
    victory_scope: str = "team"
    zones: tuple[ZoneSpec, ...] = ()
    zone_rule: str = "head"
    target_score: int | None = None
    max_rounds: int | None = None

    @property
    def individual(self) -> bool:
        """Vrai si la victoire est individuelle (mode survie, portée ``individual``)."""
        return self.mode == "survival" and self.victory_scope == "individual"

    @property
    def needs_engine(self) -> bool:
        """Vrai si le mode a des données d'état propres (tours, scores)."""
        return self.mode == "zone_control"

    def validate(self, width: int | None = None, height: int | None = None) -> None:
        if self.mode not in MODES:
            raise RulesError(f"mode de jeu inconnu : {self.mode} (attendu : {MODES})")
        if self.victory_scope not in VICTORY_SCOPES:
            raise RulesError(
                f"portée de victoire inconnue : {self.victory_scope} (attendu : {VICTORY_SCOPES})"
            )
        if self.mode != "survival" and self.victory_scope != "team":
            raise RulesError("victory_scope: individual n'existe qu'en mode survival")
        if self.zone_rule not in ZONE_RULES:
            raise RulesError(f"règle de zone inconnue : {self.zone_rule} (V1 : {ZONE_RULES})")
        for name, value in (("target_score", self.target_score), ("max_rounds", self.max_rounds)):
            if value is not None and not 1 <= value <= MAX_COUNTER:
                raise RulesError(f"{name} doit être dans [1, {MAX_COUNTER}]")
        if self.mode == "zone_control":
            if not self.zones:
                raise RulesError("le mode zone_control exige au moins une zone")
            if self.target_score is None and self.max_rounds is None:
                raise RulesError("le mode zone_control exige target_score ou max_rounds")
        elif self.zones or self.target_score is not None or self.max_rounds is not None:
            raise RulesError("zones, target_score et max_rounds sont propres à zone_control")
        for z in self.zones:
            if z.width < 1 or z.height < 1:
                raise RulesError(f"zone vide : {z}")
            if width is not None and height is not None:
                inside = z.x >= 0 and z.y >= 0 and z.x + z.width <= width
                if not (inside and z.y + z.height <= height):
                    raise RulesError(f"zone hors de la grille : {z}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode, "victory_scope": self.victory_scope}
        if self.mode == "zone_control":
            data.update(
                zones=[z.to_dict() for z in self.zones],
                zone_rule=self.zone_rule,
                target_score=self.target_score,
                max_rounds=self.max_rounds,
            )
        return data


_RULES_KEYS = frozenset(
    {"mode", "victory_scope", "zones", "zone_rule", "target_score", "max_rounds"}
)


def _zone(raw: Any) -> ZoneSpec:
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"x", "y", "width", "height"}
        if unknown:
            raise RulesError(f"clés de zone inconnues : {sorted(unknown)}")
        return ZoneSpec(
            int(raw["x"]), int(raw["y"]), int(raw.get("width", 1)), int(raw.get("height", 1))
        )
    if isinstance(raw, list | tuple) and len(raw) in (2, 4):
        return ZoneSpec(*(int(v) for v in raw))
    raise RulesError(f"zone invalide : {raw!r}")


def _optional_int(raw: Any) -> int | None:
    return None if raw is None else int(raw)


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
    try:
        return RulesSpec(
            mode=str(raw.get("mode", "classic")),
            victory_scope=str(raw.get("victory_scope", "team")),
            zones=tuple(_zone(z) for z in raw.get("zones") or ()),
            zone_rule=str(raw.get("zone_rule", "head")),
            target_score=_optional_int(raw.get("target_score")),
            max_rounds=_optional_int(raw.get("max_rounds")),
        )
    except (KeyError, TypeError) as exc:
        raise RulesError(f"règles invalides : {exc}") from exc


class RulesEngine:
    """Règles des modes à état propre ; toutes les données mutables sont dans ``GameState``.

    Crochets appelés par ``GameState`` : :meth:`after_apply` (renvoie un jeton
    d'annulation), :meth:`undo`, :meth:`decided`, :meth:`winner`, :meth:`key_part`.
    """

    def __init__(self, spec: RulesSpec, first_player: int, victory_team: tuple[int, ...]) -> None:
        self.spec = spec
        self.first = first_player
        self.team = victory_team
        self.n_teams = len(set(victory_team))
        self.zones = spec.zones

    # ------------------------------------------------------------ tours
    def round_ended(self, n: int, player: int, nxt: int) -> bool:
        """Vrai si le coup de ``player`` termine un tour complet (depuis le premier joueur)."""
        f = self.first
        return (nxt - f) % n <= (player - f) % n

    def controllers(self, state: GameState) -> list[int]:
        """Équipe qui contrôle chaque zone (règle ``head``), −1 si vide ou disputée."""
        out = []
        for zone in self.zones:
            teams = {
                self.team[p]
                for p in range(state.n)
                if state.alive[p] and zone.contains(*state.head(p))
            }
            out.append(next(iter(teams)) if len(teams) == 1 else -1)
        return out

    def after_apply(self, state: GameState, player: int, nxt: int) -> tuple[int, ...] | None:
        """Fin de tour éventuelle : compte le tour et les points ; jeton = équipes marquantes."""
        if not self.round_ended(state.n, player, nxt) or state.round >= MAX_COUNTER:
            return None
        z = state._z
        state.key ^= z.rounds[state.round] ^ z.rounds[state.round + 1]
        state.round += 1
        gained = tuple(t for t in self.controllers(state) if t >= 0)
        for t in gained:
            s = state.scores[t]
            if s < MAX_COUNTER:
                state.key ^= z.score[t][s] ^ z.score[t][s + 1]
                state.scores[t] = s + 1
        return gained

    def undo(self, state: GameState, token: tuple[int, ...] | None) -> None:
        """Annule :meth:`after_apply` (la clé est restaurée par ``GameState.undo``)."""
        if token is None:
            return
        for t in reversed(token):
            if state.scores[t] > 0:
                state.scores[t] -= 1
        state.round -= 1

    def key_part(self, state: GameState) -> int:
        z = state._z
        key = z.rounds[state.round]
        for t, s in enumerate(state.scores):
            key ^= z.score[t][s]
        return key

    # --------------------------------------------------------- victoire
    def _target_reached(self, state: GameState) -> bool:
        target = self.spec.target_score
        return target is not None and max(state.scores) >= target

    def decided(self, state: GameState) -> bool:
        limit = self.spec.max_rounds
        return self._target_reached(state) or (limit is not None and state.round >= limit)

    def winner(self, state: GameState) -> int:
        """Équipe de victoire gagnante d'une partie finie, −1 en cas d'égalité."""
        if not self._target_reached(state) and state.teams_alive == 1:
            return next(self.team[p] for p in range(state.n) if state.alive[p])
        if not state.scores:
            return -1
        best = max(state.scores)
        leaders = [t for t, s in enumerate(state.scores) if s == best]
        if len(leaders) == 1:
            return leaders[0]
        if state.teams_alive == 1:  # égalité de score : la seule équipe vivante l'emporte
            alive = next(self.team[p] for p in range(state.n) if state.alive[p])
            return alive if alive in leaders else -1
        return -1


def make_engine(
    spec: RulesSpec, first_player: int, victory_team: tuple[int, ...]
) -> RulesEngine | None:
    """Moteur de règles du mode, ou ``None`` si le mode n'a pas d'état propre."""
    return RulesEngine(spec, first_player, victory_team) if spec.needs_engine else None


__all__ = [
    "MAX_COUNTER",
    "MODES",
    "VICTORY_SCOPES",
    "ZONE_RULES",
    "RulesEngine",
    "RulesError",
    "RulesSpec",
    "ZoneSpec",
    "make_engine",
    "parse_rules",
]
