"""Scénarios : traduisent les niveaux des facteurs d'un plan en configuration de partie.

* ``solo_vs_coalition`` (étude A) : un joueur seul contre une coalition de k joueurs ;
* ``teams`` (étude B) : équipes contre équipes (``2v2``, ``1v2v3``, …) ;
* ``custom`` : configuration complète dans ``config``, facteurs = chemins pointés
  (``players.0.depth``, ``grid``…).

Chaque scénario renvoie un dictionnaire de configuration (sans graine ni premier
joueur, fixés par partie).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from tron.core.config import ConfigError

ScenarioFn = Callable[[Mapping[str, Any]], dict[str, Any]]
SCENARIOS: dict[str, ScenarioFn] = {}

_COMMON = {"heuristic", "placement", "grid", "name", "tie_break", "depth_unit"}


def register_scenario(name: str) -> Callable[[ScenarioFn], ScenarioFn]:
    def deco(fn: ScenarioFn) -> ScenarioFn:
        SCENARIOS[name] = fn
        return fn

    return deco


def build_config(scenario: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Dictionnaire de configuration de partie pour ``scenario`` et ``params``."""
    try:
        fn = SCENARIOS[scenario]
    except KeyError:
        raise ConfigError(f"scénario inconnu : {scenario} (connus : {sorted(SCENARIOS)})") from None
    return fn(params)


def _depth_field(params: Mapping[str, Any], value: int) -> dict[str, int]:
    unit = params.get("depth_unit", "plies")
    if unit == "plies":
        return {"depth": int(value)}
    if unit == "rounds":
        return {"depth": int(value), "rounds": int(value)}
    raise ConfigError(f"unité de profondeur inconnue : {unit} (plies ou rounds)")


def _grid(params: Mapping[str, Any]) -> Any:
    grid = params.get("grid", 20)
    return list(grid) if isinstance(grid, list | tuple) else int(grid)


def _base(params: Mapping[str, Any], name: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "name": params.get("name", name),
        "grid": _grid(params),
        "placement": copy.deepcopy(params.get("placement", {"mode": "regular", "rotate": True})),
        "heuristic": copy.deepcopy(params.get("heuristic", {"name": "voronoi", "tie": "order"})),
    }
    if "tie_break" in params:
        cfg["defaults"] = {"tie_break": params["tie_break"]}
    return cfg


def _member(params: Mapping[str, Any], mode: str, depth: int) -> dict[str, Any]:
    """Agent d'un membre d'équipe selon le mode de coordination."""
    alpha = float(params.get("alpha", 1.0))
    if mode == "team":
        player = {"agent": "social", "profile": "team", "alpha": alpha}
    elif mode == "team_paranoid":
        player = {"agent": "social", "profile": "team_paranoid"}
    elif mode == "maxn":
        player = {"agent": "maxn"}
    elif mode == "paranoid":
        player = {"agent": "paranoid"}
    elif mode in ("greedy", "random"):
        player = {"agent": mode}
    else:
        raise ConfigError(f"mode de coalition inconnu : {mode}")
    player.update(_depth_field(params, depth))
    return player


@register_scenario("solo_vs_coalition")
def solo_vs_coalition(params: Mapping[str, Any]) -> dict[str, Any]:
    """Étude A : joueur 0 seul (équipe ``solo``) contre k joueurs (équipe ``coalition``).

    Facteurs usuels : ``k``, ``solo_depth``, ``coalition_depth``, ``grid``,
    ``coalition_mode`` (``team`` : profil Équipe α ; ``maxn`` : individualiste),
    ``solo_agent`` (``paranoid``, ``maxn``…).
    """
    k = int(params.get("k", 1))
    if not 1 <= k <= 7:
        raise ConfigError("k doit être compris entre 1 et 7")
    cfg = _base(params, f"solo_vs_{k}")
    solo = {"name": "solo", "team": "solo", "agent": params.get("solo_agent", "paranoid")}
    solo.update(_depth_field(params, int(params.get("solo_depth", 2))))
    solo.update(params.get("solo_params", {}))
    mode = str(params.get("coalition_mode", "team"))
    members = []
    for i in range(k):
        m = {"name": f"c{i + 1}", "team": "coalition"}
        m.update(_member(params, mode, int(params.get("coalition_depth", 2))))
        m.update(params.get("coalition_params", {}))
        members.append(m)
    cfg["players"] = [solo, *members]
    return cfg


def parse_composition(text: str) -> list[int]:
    """``"2v2"`` → [2, 2] ; ``"1v2v3"`` → [1, 2, 3]."""
    try:
        sizes = [int(x) for x in str(text).lower().replace("c", "v").split("v")]
    except ValueError:
        raise ConfigError(f"composition invalide : {text!r}") from None
    if len(sizes) < 2 or min(sizes) < 1 or sum(sizes) > 8:
        raise ConfigError(f"composition invalide : {text!r}")
    return sizes


@register_scenario("teams")
def teams(params: Mapping[str, Any]) -> dict[str, Any]:
    """Étude B : équipes T0, T1… de tailles données par ``composition``.

    Profondeur de l'équipe i : ``depth_t{i}`` sinon ``depth``. Mode de chaque
    équipe : ``mode_t{i}`` sinon ``mode`` (``team``, ``maxn``, ``team_paranoid``…).
    """
    sizes = parse_composition(params.get("composition", "2v2"))
    cfg = _base(params, "teams_" + "v".join(map(str, sizes)))
    players = []
    for t, size in enumerate(sizes):
        depth = int(params.get(f"depth_t{t}", params.get("depth", 2)))
        mode = str(params.get(f"mode_t{t}", params.get("mode", "team")))
        for i in range(size):
            p = {"name": f"t{t}_{i}", "team": f"T{t}"}
            p.update(_member(params, mode, depth))
            players.append(p)
    # Sièges entrelacés : T0, T1, …, puis les seconds membres, etc.
    by_team: list[list[dict[str, Any]]] = [[] for _ in sizes]
    for p in players:
        by_team[int(p["team"][1:])].append(p)
    order: list[dict[str, Any]] = []
    for r in range(max(sizes)):
        order.extend(team[r] for team in by_team if r < len(team))
    cfg["players"] = order
    return cfg


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Affecte ``value`` au chemin pointé ``a.b.0.c`` (listes indexées par entier)."""
    keys = path.split(".")
    node: Any = data
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


@register_scenario("custom")
def custom(params: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration complète ``config`` modifiée par des facteurs à chemin pointé."""
    if "config" not in params:
        raise ConfigError("le scénario 'custom' exige une clé 'config'")
    cfg = copy.deepcopy(dict(params["config"]))
    for key, value in params.items():
        if key == "config":
            continue
        set_path(cfg, key, copy.deepcopy(value))
    return cfg
