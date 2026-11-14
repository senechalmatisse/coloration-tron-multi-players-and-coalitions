"""Sauvegarde et reprise exacte d'une partie.

Un journal JSONL permet de **rejouer** une partie ; une sauvegarde permet de la
**reprendre** sans rejouer les décisions et sans changer les décisions futures.
Les deux contrats sont distincts : la sauvegarde est un document JSON versionné
(``format`` = :data:`CHECKPOINT_FORMAT`) qui contient :

* la configuration complète, la version du code et le hash Git ;
* l'état de jeu (grille, têtes, joueurs vivants, joueur au trait, demi-coup,
  compteurs, ordre d'élimination, historique ``apply``/``undo``) et sa clé de
  Zobrist, vérifiée à la lecture ;
* l'état interne de chaque agent : générateur aléatoire, mémoire de recherche
  (table de transposition, historique), modèle d'adversaires de l'agent adaptatif ;
* l'état diplomatique, les accumulateurs des métriques de partie et les compteurs
  de statistiques de la partie ;
* les lignes ``move`` déjà jouées (pour écrire un journal complet après reprise)
  et, si la partie est finie, son résultat (pour information).

L'écriture est atomique (fichier temporaire puis renommage) : une interruption ne
laisse jamais une sauvegarde partielle à la place d'une sauvegarde valide.
Limite : la durée mesurée (``duration``) et les temps de réflexion dépendent de la
machine ; ils ne sont pas comparables entre une partie reprise et une partie d'un seul tenant.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tron
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState
from tron.sim.diplomacy import DiplomacyState
from tron.sim.match import Match, MatchObserver
from tron.sim.recorder import move_record
from tron.sim.result import git_hash

CHECKPOINT_FORMAT = "tron-checkpoint/1"
_COUNTERS = ("_decisions", "_nodes", "_time", "_max_depth", "_sum_depth", "_leaves")


class CheckpointError(ValueError):
    """Sauvegarde illisible, incomplète, corrompue ou incompatible."""


def checkpoint_dict(match: Match) -> dict[str, Any]:
    """Document JSON de la sauvegarde de ``match``."""
    return {
        "format": CHECKPOINT_FORMAT,
        "code_version": tron.__version__,
        "git_hash": git_hash(),
        "config": match.config.to_dict(),
        "state": match.state.to_snapshot(),
        "agents": [
            {"agent": spec.agent, "describe": agent.describe(), "state": agent.snapshot_state()}
            for spec, agent in zip(match.config.players, match.agents, strict=True)
        ],
        "diplomacy": match.diplomacy.to_dict() if match.diplomacy is not None else None,
        "metrics": match.metrics.to_dict() if match.metrics is not None else None,
        "counters": {name.lstrip("_"): list(getattr(match, name)) for name in _COUNTERS},
        "started": match._started,
        "zone_controllers": list(match._controllers),
        "moves": [*match.prior_moves, *(move_record(e) for e in match.events)],
        "result": match.result.to_dict() if match.result is not None else None,
    }


def save_checkpoint(match: Match, path: str | Path) -> Path:
    """Écrit la sauvegarde de ``match`` dans ``path`` (écriture atomique)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    data = json.dumps(checkpoint_dict(match), ensure_ascii=False, allow_nan=False)
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Lit et vérifie l'enveloppe d'une sauvegarde (format, champs obligatoires)."""
    src = Path(path)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{src} : sauvegarde corrompue ou incomplète ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise CheckpointError(f"{src} : sauvegarde invalide")
    fmt = data.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{src} : version de sauvegarde incompatible : {fmt!r} (attendu : {CHECKPOINT_FORMAT})"
        )
    missing = {"config", "state", "agents", "counters", "moves"} - set(data)
    if missing:
        raise CheckpointError(f"{src} : sauvegarde incomplète, champs absents : {sorted(missing)}")
    if data.get("code_version") != tron.__version__:
        warnings.warn(
            f"{src} : sauvegarde écrite par la version {data.get('code_version')} "
            f"(version courante {tron.__version__})",
            UserWarning,
            stacklevel=2,
        )
    return data


def _zone_controllers(state: GameState, saved: Any) -> list[int]:
    """Contrôleurs des zones au dernier tour complet (événements ``zone`` exacts)."""
    rules = state._rules
    if rules is None:
        return []
    zones = len(rules.spec.zones)
    teams = len(state.config.victory_names)
    if (
        not isinstance(saved, list)
        or len(saved) != zones
        or not all(type(t) is int and -1 <= t < teams for t in saved)
    ):
        raise CheckpointError("contrôleurs de zone absents ou incohérents avec la configuration")
    return list(saved)


def load_checkpoint(path: str | Path, observers: Sequence[MatchObserver] = ()) -> Match:
    """Reconstruit la partie sauvegardée, prête à continuer (``step`` ou ``run``)."""
    data = read_checkpoint(path)
    try:
        config = GameConfig.from_dict(data["config"])
        state = GameState.from_snapshot(config, data["state"])
        metrics_data = data.get("metrics")
        every = int(metrics_data["every"]) if metrics_data is not None else None
        match = Match(
            config,
            observers=observers,
            positions=state.positions,
            metrics=metrics_data is not None,
            territory_every=every,
        )
        match.state = state
        agents = data["agents"]
        if len(agents) != config.n_players:
            raise CheckpointError("nombre d'agents incohérent avec la configuration")
        for spec, agent, saved in zip(config.players, match.agents, agents, strict=True):
            if saved.get("agent") != spec.agent:
                raise CheckpointError(f"agent {saved.get('agent')!r} au lieu de {spec.agent!r}")
            agent.restore_state(saved["state"])
        if (data.get("diplomacy") is None) != (match.diplomacy is None):
            raise CheckpointError("état diplomatique incohérent avec la configuration")
        if match.diplomacy is not None:
            match.diplomacy = DiplomacyState.from_dict(
                config.diplomacy, config.n_players, data["diplomacy"]
            )
        if match.metrics is not None and metrics_data is not None:
            match.metrics.restore(metrics_data)
        counters = data["counters"]
        for name in _COUNTERS:
            values = list(counters[name.lstrip("_")])
            if len(values) != config.n_players:
                raise CheckpointError(f"compteur {name} de taille incohérente")
            setattr(match, name, values)
        match.prior_moves = [dict(m) for m in data["moves"]]
        if len(match.prior_moves) != state.ply:
            raise CheckpointError("nombre de coups enregistrés incohérent avec le demi-coup")
        match._resumed = bool(data.get("started", True))  # partie déjà commencée
        match._controllers = _zone_controllers(state, data.get("zone_controllers"))
    except CheckpointError:
        raise
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"{path} : sauvegarde invalide : {exc}") from exc
    return match


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "checkpoint_dict",
    "load_checkpoint",
    "read_checkpoint",
    "save_checkpoint",
]
