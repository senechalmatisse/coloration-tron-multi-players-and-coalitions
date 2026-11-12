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

**Obstacles dynamiques** (clé ``obstacles``, combinable avec tous les modes) :
à la fin des tours complets du calendrier (``start``, puis tous les ``every``
tours), ``count`` obstacles apparaissent sur des cases libres, jamais sur une
tête ni à distance de Manhattan ≤ ``margin`` d'une tête vivante, dans la limite
de ``max`` obstacles au total. Les cases sont tirées par un hachage déterministe
de la graine de la partie, du tour et d'un compteur (:func:`tron.core.rng.hash_index`) :
la même position donne toujours les mêmes obstacles, ce que la recherche peut
anticiper. Un obstacle est une case bloquée (valeur ``OBSTACLE`` de la grille)
que tous les noyaux traitent comme un mur.

Les équipes de victoire (``GameConfig.victory_team_index``) décident de la fin et
du vainqueur ; les équipes de la configuration (``team_index``) restent celles
des préférences sociales des agents.

Les données propres aux règles (tours complets, scores, obstacles) font partie de
l'état : ``GameState`` les tient à jour dans ``apply``/``undo``, les copie, les
inclut dans la clé de Zobrist et dans ses instantanés. Le moteur
:class:`RulesEngine` est sans état de partie ; il n'existe que pour les règles qui
en ont besoin (sinon, aucun coût).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tron.core.rng import STREAM_OBSTACLES, derive_seed, hash_index

if TYPE_CHECKING:  # pragma: no cover
    from tron.core.state import GameState

FREE = -1

MODES = ("classic", "survival", "zone_control")
VICTORY_SCOPES = ("team", "individual")
ZONE_RULES = ("head",)
OBSTACLE = -3
"""Valeur d'une case occupée par un obstacle dynamique (P09.3)."""

RulesToken = tuple[tuple[int, ...], tuple[int, ...]] | None
"""Jeton d'annulation d'une fin de tour : (équipes marquantes, cases d'obstacles posées)."""


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
class ObstacleSpec:
    """Calendrier des obstacles dynamiques (P09.3).

    Un événement a lieu à la fin des tours complets ``start``, ``start + every``…
    (``start`` vaut ``every`` par défaut) ; chacun pose ``count`` obstacles, dans la
    limite de ``max_total`` au total (clé YAML ``max``), hors des cases à distance
    de Manhattan ≤ ``margin`` d'une tête vivante.
    """

    every: int
    count: int = 1
    start: int | None = None
    max_total: int | None = None
    margin: int = 1

    @property
    def first(self) -> int:
        return self.every if self.start is None else self.start

    def is_event(self, rounds: int) -> bool:
        """Vrai si la fin du tour complet ``rounds`` déclenche des obstacles."""
        first = self.first
        return rounds >= first and (rounds - first) % self.every == 0

    def validate(self) -> None:
        for name, value, low in (
            ("every", self.every, 1),
            ("count", self.count, 1),
            ("start", self.start, 1),
            ("max", self.max_total, 1),
            ("margin", self.margin, 0),
        ):
            if value is not None and value < low:
                raise RulesError(f"obstacles.{name} doit être ≥ {low} (reçu : {value})")

    def to_dict(self) -> dict[str, int]:
        data = {"every": self.every, "count": self.count, "margin": self.margin}
        if self.start is not None:
            data["start"] = self.start
        if self.max_total is not None:
            data["max"] = self.max_total
        return data


@dataclass(frozen=True, slots=True)
class RulesSpec:
    """Règles d'une partie : mode, portée de la victoire et paramètres du mode."""

    mode: str = "classic"
    victory_scope: str = "team"
    zones: tuple[ZoneSpec, ...] = ()
    zone_rule: str = "head"
    target_score: int | None = None
    max_rounds: int | None = None
    obstacles: ObstacleSpec | None = None
    """Obstacles dynamiques (P09.3), combinables avec tous les modes."""

    @property
    def individual(self) -> bool:
        """Vrai si la victoire est individuelle (mode survie, portée ``individual``)."""
        return self.mode == "survival" and self.victory_scope == "individual"

    @property
    def needs_engine(self) -> bool:
        """Vrai si les règles ont des données d'état propres (tours, scores, obstacles)."""
        return self.mode == "zone_control" or self.obstacles is not None

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
            if value is not None and value < 1:
                raise RulesError(f"{name} doit être ≥ 1 (reçu : {value})")
        if self.obstacles is not None:
            self.obstacles.validate()
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
        if self.obstacles is not None:
            data["obstacles"] = self.obstacles.to_dict()
        return data


_RULES_KEYS = frozenset(
    {"mode", "victory_scope", "zones", "zone_rule", "target_score", "max_rounds", "obstacles"}
)
_OBSTACLE_KEYS = frozenset({"every", "count", "start", "max", "margin"})


def _int(raw: Any, name: str) -> int:
    """Entier strict : refuse booléens, flottants et chaînes (pas de conversion implicite)."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RulesError(f"{name} doit être un entier (reçu : {raw!r})")
    return raw


def _optional_int(raw: Any, name: str) -> int | None:
    return None if raw is None else _int(raw, name)


def _zone(raw: Any) -> ZoneSpec:
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"x", "y", "width", "height"}
        if unknown:
            raise RulesError(f"clés de zone inconnues : {sorted(unknown)}")
        return ZoneSpec(
            _int(raw["x"], "zone.x"),
            _int(raw["y"], "zone.y"),
            _int(raw.get("width", 1), "zone.width"),
            _int(raw.get("height", 1), "zone.height"),
        )
    if isinstance(raw, list | tuple) and len(raw) in (2, 4):
        return ZoneSpec(*(_int(v, "zone") for v in raw))
    raise RulesError(f"zone invalide : {raw!r}")


def _obstacles(raw: Any) -> ObstacleSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RulesError(f"obstacles invalides : {raw!r} (attendu : every, count, start, max)")
    unknown = set(raw) - _OBSTACLE_KEYS
    if unknown:
        raise RulesError(f"clés d'obstacles inconnues : {sorted(unknown)}")
    if "every" not in raw:
        raise RulesError("obstacles.every est obligatoire")
    return ObstacleSpec(
        every=_int(raw["every"], "obstacles.every"),
        count=_int(raw.get("count", 1), "obstacles.count"),
        start=_optional_int(raw.get("start"), "obstacles.start"),
        max_total=_optional_int(raw.get("max"), "obstacles.max"),
        margin=_int(raw.get("margin", 1), "obstacles.margin"),
    )


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
            target_score=_optional_int(raw.get("target_score"), "target_score"),
            max_rounds=_optional_int(raw.get("max_rounds"), "max_rounds"),
            obstacles=_obstacles(raw.get("obstacles")),
        )
    except (KeyError, TypeError) as exc:
        raise RulesError(f"règles invalides : {exc}") from exc


class RulesEngine:
    """Règles à état propre ; toutes les données mutables de la partie sont dans ``GameState``.

    Crochets appelés par ``GameState`` : :meth:`after_apply` (renvoie un jeton
    d'annulation), :meth:`undo`, :meth:`decided`, :meth:`winner`, :meth:`key_part`.
    """

    def __init__(
        self,
        spec: RulesSpec,
        first_player: int,
        victory_team: tuple[int, ...],
        seed: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self.spec = spec
        self.first = first_player
        self.team = victory_team
        self.n_teams = len(set(victory_team))
        self.zones = spec.zones
        self.obstacles = spec.obstacles
        self.width = width
        self.height = height
        self.stride = width + 2
        self.seed = derive_seed(seed, STREAM_OBSTACLES)

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

    def after_apply(self, state: GameState, player: int, nxt: int) -> RulesToken:
        """Fin de tour éventuelle : tour, points de zone puis obstacles ; renvoie le jeton."""
        if not self.round_ended(state.n, player, nxt):
            return None
        z = state._z
        r = state.round
        state.key ^= z.round_key(r) ^ z.round_key(r + 1)
        state.round = r + 1
        gained: tuple[int, ...] = ()
        if self.zones:
            gained = tuple(t for t in self.controllers(state) if t >= 0)
            for t in gained:
                s = state.scores[t]
                state.key ^= z.score_key(t, s) ^ z.score_key(t, s + 1)
                state.scores[t] = s + 1
        placed: tuple[int, ...] = ()
        obs = self.obstacles
        if (
            obs is not None
            and obs.is_event(r + 1)
            and state.teams_alive > 1
            and not self.decided(state)
        ):
            placed = self._place(state, r + 1, obs)
        return gained, placed

    def _place(self, state: GameState, rounds: int, obs: ObstacleSpec) -> tuple[int, ...]:
        """Pose les obstacles de la fin du tour ``rounds`` (déterministe pour un état donné)."""
        want = obs.count
        if obs.max_total is not None:
            want = min(want, obs.max_total - state.n_obstacles)
        if want <= 0:
            return ()
        g = state._g
        w, stride, margin = self.width, self.stride, obs.margin
        heads = [state.head(p) for p in range(state.n) if state.alive[p]]

        def allowed(cell: int) -> bool:
            if g[cell] != FREE:
                return False
            y, x = divmod(cell, stride)
            x, y = x - 1, y - 1
            return all(abs(x - hx) + abs(y - hy) > margin for hx, hy in heads)

        total = w * self.height
        base = hash_index(self.seed, rounds)
        placed: list[int] = []
        tries = 8 * want + 16
        for i in range(tries):
            if len(placed) == want:
                break
            k = hash_index(base, i) % total
            cell = (k // w + 1) * stride + k % w + 1
            if allowed(cell):
                g[cell] = OBSTACLE
                placed.append(cell)
        if len(placed) < want:  # grille encombrée : toutes les cases admissibles, ordre haché
            rest = [(y + 1) * stride + x + 1 for y in range(self.height) for x in range(w)]
            rest = sorted((c for c in rest if allowed(c)), key=lambda c: hash_index(base, -1, c))
            for cell in rest[: want - len(placed)]:
                g[cell] = OBSTACLE
                placed.append(cell)
        obst = state._z.obst
        for cell in placed:
            state.key ^= obst[cell]
        state.n_obstacles += len(placed)
        return tuple(placed)

    def undo(self, state: GameState, token: RulesToken) -> None:
        """Annule :meth:`after_apply` (la clé est restaurée par ``GameState.undo``)."""
        if token is None:
            return
        gained, placed = token
        g = state._g
        for cell in placed:
            g[cell] = FREE
        state.n_obstacles -= len(placed)
        for t in gained:
            state.scores[t] -= 1
        state.round -= 1

    def key_part(self, state: GameState) -> int:
        """Part de la clé de Zobrist propre aux règles (tours et scores ; obstacles : grille)."""
        z = state._z
        key = z.round_key(state.round)
        for t, s in enumerate(state.scores):
            key ^= z.score_key(t, s)
        return key

    # --------------------------------------------------------- victoire
    def _target_reached(self, state: GameState) -> bool:
        target = self.spec.target_score
        return target is not None and max(state.scores) >= target

    def decided(self, state: GameState) -> bool:
        """Vrai si une condition propre au mode (score cible, durée) termine la partie."""
        if not self.zones:
            return False
        limit = self.spec.max_rounds
        return self._target_reached(state) or (limit is not None and state.round >= limit)

    def winner(self, state: GameState) -> int:
        """Équipe de victoire gagnante d'une partie finie, −1 en cas d'égalité."""
        if not self.zones or (not self._target_reached(state) and state.teams_alive == 1):
            if state.teams_alive != 1:
                return -1
            return next(self.team[p] for p in range(state.n) if state.alive[p])
        best = max(state.scores)
        leaders = [t for t, s in enumerate(state.scores) if s == best]
        if len(leaders) == 1:
            return leaders[0]
        if state.teams_alive == 1:  # égalité de score : la seule équipe vivante l'emporte
            alive = next(self.team[p] for p in range(state.n) if state.alive[p])
            return alive if alive in leaders else -1
        return -1


def make_engine(
    spec: RulesSpec,
    first_player: int,
    victory_team: tuple[int, ...],
    seed: int = 0,
    width: int = 0,
    height: int = 0,
) -> RulesEngine | None:
    """Moteur de règles, ou ``None`` si les règles n'ont pas d'état propre."""
    if not spec.needs_engine:
        return None
    return RulesEngine(spec, first_player, victory_team, seed, width, height)


__all__ = [
    "MODES",
    "OBSTACLE",
    "VICTORY_SCOPES",
    "ZONE_RULES",
    "ObstacleSpec",
    "RulesEngine",
    "RulesError",
    "RulesSpec",
    "RulesToken",
    "ZoneSpec",
    "make_engine",
    "parse_rules",
]
