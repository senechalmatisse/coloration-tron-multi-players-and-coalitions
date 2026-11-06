"""Configuration immuable d'une partie (F5.1).

Exemple de fichier YAML ::

    name: ffa4
    grid: {width: 20, height: 20}      # ou grid: 20, ou grid: [20, 20]
    seed: 42
    first_player: 0
    placement: {mode: regular, radius: 0.35, rotate: true}
    heuristic: {name: voronoi, tie: order}
    defaults: {agent: paranoid, depth: 4}
    players:
      - {name: rouge, team: A, agent: paranoid, depth: 6}
      - {name: bleu, team: B, agent: social, depth: 3, profile: team, alpha: 1.0}

Les clés d'un joueur autres que ``name``, ``team``, ``agent`` et ``depth`` sont
transmises à l'agent comme paramètres. Un joueur sans équipe forme sa propre équipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

MIN_SIZE, MAX_SIZE = 5, 100
MIN_PLAYERS, MAX_PLAYERS = 2, 8
PLACEMENT_MODES = ("regular", "random", "manual")

_PLAYER_KEYS = frozenset({"name", "team", "agent", "depth", "params"})
_CONFIG_KEYS = frozenset(
    {
        "name",
        "grid",
        "width",
        "height",
        "seed",
        "first_player",
        "placement",
        "heuristic",
        "players",
        "defaults",
    }
)


class ConfigError(ValueError):
    """Configuration de partie invalide."""


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    """Un joueur : nom, équipe, algorithme, profondeur (plies) et paramètres d'agent."""

    name: str
    team: str
    agent: str = "greedy"
    depth: int = 1
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "team": self.team,
            "agent": self.agent,
            "depth": self.depth,
        }
        if self.params:
            data["params"] = dict(self.params)
        return data


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    """Placement initial (F1.6) : régulier autour du centre, aléatoire ou manuel."""

    mode: str = "regular"
    radius: float = 0.35
    rotate: bool = False
    min_distance: int = 3
    positions: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.mode == "manual":
            return {"mode": "manual", "positions": [list(p) for p in self.positions]}
        if self.mode == "random":
            return {"mode": "random", "min_distance": self.min_distance}
        return {"mode": "regular", "radius": self.radius, "rotate": self.rotate}


@dataclass(frozen=True, slots=True)
class HeuristicSpec:
    """Heuristique d'évaluation et ses paramètres (F2.1–F2.4)."""

    name: str = "voronoi"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Configuration complète et immuable d'une partie."""

    width: int
    height: int
    players: tuple[PlayerSpec, ...]
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    heuristic: HeuristicSpec = field(default_factory=HeuristicSpec)
    seed: int = 0
    first_player: int = 0
    name: str = "partie"

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ dérivés
    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def team_names(self) -> tuple[str, ...]:
        """Noms d'équipes dans l'ordre de première apparition."""
        seen: dict[str, None] = {}
        for p in self.players:
            seen.setdefault(p.team, None)
        return tuple(seen)

    @property
    def team_index(self) -> tuple[int, ...]:
        """Indice d'équipe de chaque joueur."""
        names = self.team_names
        return tuple(names.index(p.team) for p in self.players)

    @property
    def team_sizes(self) -> tuple[int, ...]:
        idx = self.team_index
        return tuple(idx.count(t) for t in range(len(self.team_names)))

    def teammates(self, player: int) -> tuple[int, ...]:
        """Coéquipiers du joueur (lui exclu)."""
        team = self.players[player].team
        return tuple(i for i, p in enumerate(self.players) if p.team == team and i != player)

    def with_seed(self, seed: int, first_player: int | None = None) -> GameConfig:
        """Copie avec une autre graine (et éventuellement un autre premier joueur)."""
        fp = self.first_player if first_player is None else first_player
        return replace(self, seed=seed, first_player=fp)

    # ------------------------------------------------------------ validation
    def _validate(self) -> None:
        w, h = self.width, self.height
        if not (MIN_SIZE <= w <= MAX_SIZE and MIN_SIZE <= h <= MAX_SIZE):
            raise ConfigError(f"grille {w}x{h} hors de [{MIN_SIZE}, {MAX_SIZE}]")
        n = len(self.players)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:
            raise ConfigError(f"{n} joueurs : il en faut entre {MIN_PLAYERS} et {MAX_PLAYERS}")
        names = [p.name for p in self.players]
        if len(set(names)) != n:
            raise ConfigError(f"noms de joueurs non uniques : {names}")
        if len(self.team_names) < 2:
            raise ConfigError("il faut au moins deux équipes")
        for p in self.players:
            if p.depth < 1:
                raise ConfigError(f"profondeur invalide pour {p.name} : {p.depth}")
        if not 0 <= self.first_player < n:
            raise ConfigError(f"premier joueur invalide : {self.first_player}")
        if self.seed < 0:
            raise ConfigError("la graine doit être positive")
        pl = self.placement
        if pl.mode not in PLACEMENT_MODES:
            raise ConfigError(f"mode de placement inconnu : {pl.mode}")
        if pl.mode == "regular" and not 0.0 < pl.radius <= 0.5:
            raise ConfigError("radius doit être dans ]0, 0.5]")
        if pl.mode == "random" and pl.min_distance < 1:
            raise ConfigError("min_distance doit être >= 1")
        if pl.mode == "manual":
            if len(pl.positions) != n:
                raise ConfigError("placement manuel : une position par joueur")
            if len(set(pl.positions)) != n:
                raise ConfigError("placement manuel : positions en double")
            for x, y in pl.positions:
                if not (0 <= x < w and 0 <= y < h):
                    raise ConfigError(f"position hors grille : ({x}, {y})")

    # -------------------------------------------------------- sérialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grid": {"width": self.width, "height": self.height},
            "seed": self.seed,
            "first_player": self.first_player,
            "placement": self.placement.to_dict(),
            "heuristic": self.heuristic.to_dict(),
            "players": [p.to_dict() for p in self.players],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GameConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("la configuration doit être un dictionnaire")
        unknown = set(data) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"clés inconnues : {sorted(unknown)}")
        width, height = _parse_grid(data)
        defaults = dict(data.get("defaults") or {})
        raw_players = data.get("players")
        if not raw_players or not isinstance(raw_players, Sequence):
            raise ConfigError("la liste 'players' est obligatoire")
        players = tuple(_parse_player(i, raw, defaults) for i, raw in enumerate(raw_players))
        return cls(
            width=width,
            height=height,
            players=players,
            placement=_parse_placement(data.get("placement", "regular")),
            heuristic=_parse_heuristic(data.get("heuristic", "voronoi")),
            seed=int(data.get("seed", 0)),
            first_player=int(data.get("first_player", 0)),
            name=str(data.get("name", "partie")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GameConfig:
        return cls.from_dict(load_yaml(path))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


# ---------------------------------------------------------------- utilitaires
def load_yaml(path: str | Path) -> Any:
    """Lit un fichier YAML (chargeur C s'il est disponible)."""
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.load(fh, Loader=loader)


def _parse_grid(data: Mapping[str, Any]) -> tuple[int, int]:
    grid = data.get("grid")
    if grid is None:
        w, h = data.get("width"), data.get("height")
    elif isinstance(grid, int):
        w = h = grid
    elif isinstance(grid, Mapping):
        w, h = grid.get("width"), grid.get("height")
    elif isinstance(grid, Sequence) and len(grid) == 2:
        w, h = grid[0], grid[1]
    else:
        raise ConfigError(f"grille invalide : {grid!r}")
    if w is None or h is None:
        raise ConfigError("dimensions de grille manquantes")
    return int(w), int(h)


def _parse_player(i: int, raw: Any, defaults: Mapping[str, Any]) -> PlayerSpec:
    if isinstance(raw, str):
        raw = {"agent": raw}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"joueur {i} invalide : {raw!r}")
    merged: dict[str, Any] = {**defaults, **raw}
    params: dict[str, Any] = dict(defaults.get("params") or {})
    params.update(raw.get("params") or {})
    for key, value in merged.items():
        if key not in _PLAYER_KEYS:
            params[key] = value
    name = str(merged.get("name", f"J{i}"))
    return PlayerSpec(
        name=name,
        team=str(merged.get("team", name)),
        agent=str(merged.get("agent", "greedy")),
        depth=int(merged.get("depth", 1)),
        params=params,
    )


def _parse_placement(raw: Any) -> PlacementSpec:
    if isinstance(raw, str):
        return PlacementSpec(mode=raw)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"placement invalide : {raw!r}")
    positions = tuple((int(p[0]), int(p[1])) for p in raw.get("positions", ()))
    return PlacementSpec(
        mode=str(raw.get("mode", "manual" if positions else "regular")),
        radius=float(raw.get("radius", 0.35)),
        rotate=bool(raw.get("rotate", False)),
        min_distance=int(raw.get("min_distance", 3)),
        positions=positions,
    )


def _parse_heuristic(raw: Any) -> HeuristicSpec:
    if isinstance(raw, str):
        return HeuristicSpec(name=raw)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"heuristique invalide : {raw!r}")
    params = {k: v for k, v in raw.items() if k != "name"}
    return HeuristicSpec(name=str(raw.get("name", "voronoi")), params=params)
