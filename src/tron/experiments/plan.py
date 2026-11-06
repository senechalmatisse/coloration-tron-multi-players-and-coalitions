"""Plan d'expériences YAML (F5.3).

Exemple ::

    name: etude_a
    scenario: solo_vs_coalition
    games: 100                 # parties par condition
    seed: 2026                 # graine du plan : graines appariées entre conditions
    seats: cyclic              # cyclic : chaque graine jouée avec chaque premier joueur
    base: {coalition_mode: team, alpha: 1.0}
    factors:
      k: [1, 2, 3, 4, 5]
      solo_depth: [1, 2, 4, 6, 8, 10]
      coalition_depth: [1, 2, 3, 4]
      grid: [10, 20, 40]
    design: full               # full | {type: subset, n: 40, seed: 1} | {type: list, ...}
    exclude: [{k: 5, grid: 10}]
    key_conditions: {games: 200, where: [{k: 3, grid: 20}]}

Graines appariées : la graine de la partie g dépend seulement de la graine du
plan et de g (et non de la condition) ; toutes les conditions de même nombre de
joueurs partagent donc les mêmes placements initiaux. Avec ``seats: cyclic``, la
graine d'indice g // n est jouée par n parties consécutives, chacune avec un
premier joueur différent (rotation complète des sièges).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tron.core.config import ConfigError, GameConfig, load_yaml
from tron.core.rng import derive_seed
from tron.experiments.scenarios import build_config

_PLAN_KEYS = frozenset(
    {
        "name",
        "scenario",
        "games",
        "seed",
        "seats",
        "base",
        "factors",
        "design",
        "exclude",
        "key_conditions",
        "logs",
        "description",
    }
)


@dataclass(frozen=True, slots=True)
class Condition:
    """Une combinaison de niveaux des facteurs."""

    index: int
    factors: dict[str, Any]
    games: int
    config: dict[str, Any]
    n_players: int

    @property
    def key(self) -> str:
        """Identifiant canonique (JSON trié) : stable quel que soit l'ordre du plan."""
        return json.dumps(self.factors, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class Task:
    """Une partie à jouer."""

    plan: str
    condition: str
    cond_index: int
    game: int
    seed: int
    first_player: int
    factors: dict[str, Any]
    config: dict[str, Any]

    @property
    def key(self) -> tuple[str, int]:
        return self.condition, self.game

    def game_config(self) -> GameConfig:
        return GameConfig.from_dict(
            {**self.config, "seed": self.seed, "first_player": self.first_player}
        )


def _matches(factors: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    return all(factors.get(k) == v for k, v in where.items())


@dataclass
class Plan:
    """Plan d'expériences : scénario, facteurs, nombre de parties, graines."""

    name: str
    scenario: str
    factors: dict[str, list[Any]]
    games: int = 100
    seed: int = 0
    seats: str = "cyclic"
    base: dict[str, Any] = field(default_factory=dict)
    design: Any = "full"
    exclude: list[dict[str, Any]] = field(default_factory=list)
    key_conditions: dict[str, Any] = field(default_factory=dict)
    logs: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ConfigError("games doit être >= 1")
        if self.seats not in ("cyclic", "per_game"):
            raise ConfigError("seats doit valoir 'cyclic' ou 'per_game'")
        for name, levels in self.factors.items():
            if not isinstance(levels, Sequence) or isinstance(levels, str) or not levels:
                raise ConfigError(f"le facteur {name} doit avoir une liste de niveaux")

    # ---------------------------------------------------------------- chargement
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        unknown = set(data) - _PLAN_KEYS
        if unknown:
            raise ConfigError(f"clés de plan inconnues : {sorted(unknown)}")
        try:
            return cls(
                name=str(data["name"]),
                scenario=str(data["scenario"]),
                factors={
                    k: list(v) if isinstance(v, list | tuple) else v
                    for k, v in (data.get("factors") or {}).items()
                },
                games=int(data.get("games", 100)),
                seed=int(data.get("seed", 0)),
                seats=str(data.get("seats", "cyclic")),
                base=dict(data.get("base") or {}),
                design=data.get("design", "full"),
                exclude=list(data.get("exclude") or []),
                key_conditions=dict(data.get("key_conditions") or {}),
                logs=bool(data.get("logs", False)),
                description=str(data.get("description", "")),
            )
        except KeyError as exc:
            raise ConfigError(f"clé de plan obligatoire manquante : {exc}") from None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Plan:
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "games": self.games,
            "seed": self.seed,
            "seats": self.seats,
            "base": self.base,
            "factors": self.factors,
            "design": self.design,
            "exclude": self.exclude,
            "key_conditions": self.key_conditions,
            "logs": self.logs,
            "description": self.description,
        }

    # ------------------------------------------------------------- conditions
    def _combinations(self) -> list[dict[str, Any]]:
        design = self.design
        names = list(self.factors)
        full = [
            dict(zip(names, values, strict=True))
            for values in itertools.product(*(self.factors[k] for k in names))
        ]
        if design in (None, "full"):
            combos = full
        elif isinstance(design, Mapping) and design.get("type") == "subset":
            n = int(design["n"])
            rng = np.random.default_rng(int(design.get("seed", 0)))
            idx = sorted(rng.choice(len(full), size=min(n, len(full)), replace=False).tolist())
            combos = [full[i] for i in idx]
        elif isinstance(design, Mapping) and design.get("type") == "list":
            combos = [dict(c) for c in design["conditions"]]
        else:
            raise ConfigError(f"plan d'expériences inconnu : {design!r}")
        return [c for c in combos if not any(_matches(c, ex) for ex in self.exclude)]

    def conditions(self) -> list[Condition]:
        """Conditions du plan, avec configuration de partie validée."""
        out = []
        where = self.key_conditions.get("where", [])
        key_games = int(self.key_conditions.get("games", self.games))
        for i, factors in enumerate(self._combinations()):
            params = {**self.base, **factors}
            config = build_config(self.scenario, params)
            n_players = GameConfig.from_dict(config).n_players  # validation précoce
            games = key_games if any(_matches(factors, w) for w in where) else self.games
            out.append(Condition(i, factors, games, config, n_players))
        return out

    def tasks(self) -> list[Task]:
        """Toutes les parties du plan, dans un ordre déterministe."""
        tasks = []
        for cond in self.conditions():
            n = cond.n_players
            for g in range(cond.games):
                seed_index = g // n if self.seats == "cyclic" else g
                tasks.append(
                    Task(
                        plan=self.name,
                        condition=cond.key,
                        cond_index=cond.index,
                        game=g,
                        seed=derive_seed(self.seed, seed_index),
                        first_player=g % n,
                        factors=cond.factors,
                        config=cond.config,
                    )
                )
        return tasks
