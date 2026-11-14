"""Outils de configuration : validation, forme normalisée, modèles, catalogues.

Commandes associées : ``tron config validate|show|template``, ``tron agents list``
et ``tron heuristics list``. Le type d'un fichier (partie, plan d'expériences,
entraînement) est reconnu à ses clés, ou imposé par ``--kind``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tron.core.config import ConfigError, GameConfig

KINDS = ("auto", "game", "plan", "training")
_PLAN_KEYS = frozenset({"scenario", "factors", "design", "key_conditions", "seats"})
_TRAINING_KEYS = frozenset({"episodes", "generations", "learning", "opponents", "self_play"})


@dataclass(frozen=True)
class Checked:
    """Fichier validé : type, objet chargé et résumé d'une ligne."""

    kind: str
    obj: Any
    summary: str


def detect_kind(data: Mapping[str, Any]) -> str:
    """Partie, plan ou entraînement, d'après les clés du document."""
    keys = set(data)
    if keys & _PLAN_KEYS:
        return "plan"
    if "game" in keys and keys & (_TRAINING_KEYS | {"evaluation", "env"}):
        return "training"
    return "game"


def _read(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"fichier introuvable : {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide : {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("le document doit être un dictionnaire YAML")
    return data


def check(path: str | Path, kind: str = "auto") -> Checked:
    """Charge et valide en profondeur (agents et heuristiques instanciés) ; ``ConfigError``."""
    from tron.agents.base import build_agents  # noqa: PLC0415
    from tron.experiments.plan import Plan  # noqa: PLC0415
    from tron.learning.train import TrainingSpec  # noqa: PLC0415

    if kind not in KINDS:
        raise ConfigError(f"type inconnu : {kind} (attendu : {KINDS})")
    p = Path(path)
    data = _read(p)
    kind = detect_kind(data) if kind == "auto" else kind
    try:
        if kind == "game":
            cfg = GameConfig.from_dict(data)
            agents = build_agents(cfg)  # paramètres d'agents et d'heuristiques
            teams = ", ".join(cfg.team_names)
            return Checked(
                kind,
                cfg,
                f"partie {cfg.name} : {cfg.width}x{cfg.height}, {cfg.n_players} joueurs "
                f"({', '.join(a.describe() for a in agents)}), équipes {teams}, "
                f"mode {cfg.rules.mode}",
            )
        if kind == "plan":
            plan = Plan.from_dict(data)
            conditions = plan.conditions()  # chaque condition construit sa configuration
            for cond in conditions:
                build_agents(GameConfig.from_dict(cond.config))
            games = sum(c.games for c in conditions)
            return Checked(
                kind,
                plan,
                f"plan {plan.name} : scénario {plan.scenario}, {len(conditions)} conditions, "
                f"{games} parties",
            )
        spec = TrainingSpec.from_dict(data, base=p.parent)
        return Checked(
            kind,
            spec,
            f"entraînement {spec.name} : {spec.total_episodes} épisodes, partie "
            f"{spec.game.name}, apprenant {spec.player}, évaluation {spec.eval_games} parties "
            f"par référence",
        )
    except ConfigError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError(str(exc)) from exc


def normalized(checked: Checked) -> dict[str, Any]:
    """Forme normalisée (valeurs par défaut explicites) d'un fichier validé."""
    if checked.kind in ("game", "plan"):
        data: dict[str, Any] = checked.obj.to_dict()
        return data
    spec = checked.obj
    return {
        "name": spec.name,
        "game": spec.game.to_dict(),
        "player": spec.player,
        "seed": spec.seed,
        "episodes": spec.episodes,
        "generations": spec.generations,
        "learning": {
            "alpha": spec.alpha,
            "gamma": spec.gamma,
            "epsilon": {"start": spec.epsilon_start, "end": spec.epsilon_end},
        },
        "env": {
            "shaping": spec.shaping,
            "shaping_scale": spec.shaping_scale,
            "max_steps": spec.max_steps,
        },
        "opponents": [dict(o) for o in spec.opponents],
        "self_play": {"enabled": spec.self_play, "share": spec.self_play_share},
        "evaluation": {
            "games": spec.eval_games,
            "references": list(spec.eval_references),
            "depth": spec.eval_depth,
            "configs": [cfg.to_dict() for _source, cfg in spec.eval_configs],
        },
    }


def dump(data: Mapping[str, Any], fmt: str = "yaml") -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    text: str = yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, width=100)
    return text


# modèles
TEMPLATES: dict[str, str] = {
    "game": """\
# Partie simple : quatre joueurs, deux equipes (tron config template game).
#   tron config validate partie.yaml && tron play partie.yaml
name: ma_partie
grid: 20                                                  # ou {width: 30, height: 20}
seed: 1                                                   # tout l'alea derive de cette graine
first_player: 0
placement: {mode: regular, radius: 0.35, rotate: true}    # regular | random | manual
heuristic: {name: voronoi, tie: order}                    # voronoi | composite | zone
players:                                                  # agents : tron agents list
  - {name: p1, team: A, agent: paranoid, depth: 4}
  - {name: m1, team: B, agent: maxn, depth: 3}
  - {name: p2, team: A, agent: greedy}
  - {name: m2, team: B, agent: mcts, simulations: 200}
""",
    "teams": """\
# Equipes contre equipes avec preferences sociales (profil team, alpha = 1).
name: equipes_2v2
grid: 20
seed: 2
placement: {mode: regular, rotate: true}
heuristic: {name: voronoi, tie: order}
defaults: {agent: social, depth: 3, profile: team, alpha: 1.0}
players:
  - {name: a1, team: A}
  - {name: b1, team: B}
  - {name: a2, team: A}
  - {name: b2, team: B}
""",
    "dynamic": """\
# Coalitions dynamiques (diplomatie tactique, docs/coalitions-dynamiques.md).
name: coalitions_dynamiques
grid: 20
seed: 3
placement: {mode: regular, rotate: true}
heuristic: {name: voronoi, tie: order}
diplomacy:
  enabled: true
  proposal_ttl_plies: 8       # duree de vie d'une proposition
  max_allies_per_player: 2
  alliance_weight: 0.65       # poids d'un allie dans la matrice W
  betrayal_penalty: 0.0
  break_cooldown_plies: 4
players:
  - {name: j1, team: J1, agent: social,   depth: 3, profile: dynamic_team}
  - {name: j2, team: J2, agent: social,   depth: 3, profile: dynamic_team}
  - {name: j3, team: J3, agent: social,   depth: 3, profile: dynamic_team}
  - {name: j4, team: J4, agent: paranoid, depth: 3}
""",
    "zone": """\
# Controle de zone (docs/modes-de-jeu.md) : un point par zone tenue seule a chaque tour.
name: controle_de_zone
grid: 16
seed: 4
placement: {mode: regular, rotate: true}
heuristic: {name: zone}
rules:
  mode: zone_control
  zones: [[2, 2, 3, 3], [11, 11, 3, 3]]
  target_score: 12
  max_rounds: 120
players:
  - {name: a1, team: A, agent: social, depth: 2, profile: team}
  - {name: b1, team: B, agent: maxn,   depth: 2}
  - {name: a2, team: A, agent: social, depth: 2, profile: team}
  - {name: b2, team: B, agent: greedy}
""",
    "campaign": """\
# Plan d'experiences (docs/experiences.md) : tron run plan.yaml, puis tron analyze.
name: ma_campagne
scenario: solo_vs_coalition   # solo_vs_coalition | teams | custom
games: 20                     # parties par condition
seed: 2026                    # graines appariees entre conditions
seats: cyclic                 # chaque graine est jouee avec chaque premier joueur
base:
  solo_agent: paranoid
  coalition_mode: team
  alpha: 1.0
  placement: {mode: regular, rotate: true}
  heuristic: {name: voronoi, tie: order}
factors:
  solo_depth: [2, 4]
  coalition_depth: [1, 2]
  k: [2, 3]
  grid: [12]
pilot: {games: 2}             # tron run plan.yaml --pilot
""",
    "training": """\
# Entrainement de la politique lineaire (docs/learning.md) : tron train entrainement.yaml
name: mon_modele
game:
  name: duel_appris
  grid: 10
  seed: 3
  placement: {mode: regular, rotate: true}
  players:
    - {name: apprenant,  team: A, agent: greedy}
    - {name: adversaire, team: B, agent: greedy}
player: 0
seed: 7
episodes: 100
generations: 2
learning: {alpha: 0.05, gamma: 1.0, epsilon: {start: 0.3, end: 0.02}}
env: {shaping: territory, shaping_scale: 0.1, max_steps: 200}
opponents: [greedy, random, {agent: paranoid, depth: 2}]
self_play: {enabled: true, share: 0.3}
evaluation: {games: 10, references: [greedy, maxn, paranoid, social], depth: 2}
""",
}


def write_template(name: str, out: str | Path | None, force: bool = False) -> str:
    """Texte du modèle ``name`` ; écrit dans ``out`` (refus d'écraser sans ``force``)."""
    try:
        text = TEMPLATES[name]
    except KeyError:
        raise ConfigError(f"modèle inconnu : {name} (attendu : {sorted(TEMPLATES)})") from None
    if out is not None and str(out) != "-":
        path = Path(out)
        if path.exists() and not force:
            raise ConfigError(f"{path} existe déjà (--force pour l'écraser)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


# catalogues
def _doc(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.splitlines()[0] if doc else ""


def _params(fn: Any, skip: set[str]) -> list[dict[str, Any]]:
    out = []
    for name, par in inspect.signature(fn).parameters.items():
        if name in skip or par.kind in (par.VAR_KEYWORD, par.VAR_POSITIONAL):
            continue
        default = None if par.default is par.empty else par.default
        annotation = "" if par.annotation is par.empty else str(par.annotation)
        out.append(
            {
                "name": name,
                "default": default if isinstance(default, int | float | str | bool) else None,
                "required": par.default is par.empty,
                "type": annotation,
            }
        )
    return out


def agent_catalog() -> list[dict[str, Any]]:
    """Agents enregistrés (intégrés et extensions) avec leurs paramètres."""
    from tron.agents.base import AGENTS, Agent  # noqa: PLC0415

    common = _params(Agent.__init__, {"self", "player", "config"})
    rows = []
    for name, cls in sorted(AGENTS.items()):
        own = [
            p
            for p in _params(cls.__init__, {"self", "player", "config"})
            if p["name"] not in {c["name"] for c in common}
        ]
        rows.append(
            {
                "name": name,
                "class": f"{cls.__module__}:{cls.__qualname__}",
                "description": _doc(cls),
                "parameters": own,
                "common": [c["name"] for c in common],
            }
        )
    return rows


def heuristic_catalog() -> list[dict[str, Any]]:
    """Heuristiques enregistrées avec leurs paramètres (hors dimensions de la grille)."""
    from tron.eval.base import HEURISTICS  # noqa: PLC0415

    rows = []
    for name, cls in sorted(HEURISTICS.items()):
        rows.append(
            {
                "name": name,
                "class": f"{cls.__module__}:{cls.__qualname__}",
                "description": _doc(cls),
                "parameters": _params(cls.__init__, {"self", "width", "height", "n_players"}),
            }
        )
    return rows


def catalog_text(rows: list[dict[str, Any]], title: str) -> str:
    lines = [title]
    for r in rows:
        params = ", ".join(
            f"{p['name']}" + ("" if p["required"] else f"={p['default']!r}")
            for p in r["parameters"]
        )
        lines.append(f"  {r['name']:<10} {r['description']}")
        lines.append(f"  {'':<10} paramètres : {params or '—'}")
        lines.append(f"  {'':<10} classe : {r['class']}")
    if rows and "common" in rows[0]:
        lines.append(f"paramètres communs à tous les agents : {', '.join(rows[0]['common'])}")
    return "\n".join(lines)


__all__ = [
    "KINDS",
    "TEMPLATES",
    "Checked",
    "agent_catalog",
    "catalog_text",
    "check",
    "detect_kind",
    "dump",
    "heuristic_catalog",
    "normalized",
    "write_template",
]
