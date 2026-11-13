"""API Python publique et stable (P11).

Façade pour un notebook, un autre programme Python ou un outil externe, sans
passer par la CLI. Les noms de ``__all__`` forment l'API stable (version
:data:`API_VERSION`) ; le reste du paquet peut évoluer.

.. code-block:: python

    from tron.api import load_game_config, run_game, run_games, load_replay

    config = load_game_config("configs/games/duel_20x20.yaml")
    result = run_game(config, seed=3, log="partie.jsonl")
    print(result.winning_team, result.n_moves)
    final = load_replay("partie.jsonl").final_state()

**Contrats de mutabilité**

* :class:`GameConfig` est **immuable** (``dataclass`` gelée) : ``with_seed`` et
  ``from_dict`` renvoient de nouveaux objets ; une configuration se partage sans
  risque entre parties, processus et fils.
* :class:`GameState` est **mutable** : ``apply``/``undo`` le modifient en place.
  Un agent reçoit l'état de la partie et doit le rendre intact (``try/finally``
  autour de ``apply``/``undo``) ; ``copy()`` donne un état indépendant.
* :class:`Match` est **à usage unique** : ``run()`` joue la partie jusqu'au bout
  et la consomme ; un nouvel appel renvoie le même :class:`MatchResult` sans
  rejouer. Pour une autre partie, créer un autre ``Match``.
* :class:`MatchResult` et :class:`GameLog` sont des valeurs : à lire, pas à
  modifier.

**Exceptions publiques** : :class:`ConfigError` (configuration invalide),
:class:`LogError` (journal illisible ou incohérent), :class:`CheckpointError`
(sauvegarde), :class:`ModelError` (modèle appris) — toutes sous-classes de
``ValueError``.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tron.agents.base import (
    AGENTS,
    Agent,
    Decision,
    DecisionDetails,
    SearchStats,
    build_agents,
    make_agent,
    register_agent,
)
from tron.core.config import ConfigError, GameConfig
from tron.core.state import GameState, Outcome
from tron.eval.base import HEURISTICS, Heuristic, make_heuristic, register_heuristic
from tron.learning.model import ModelError
from tron.sim.arena import game_config, run_arena
from tron.sim.checkpoint import CheckpointError, load_checkpoint
from tron.sim.match import Match, MatchObserver, MoveEvent
from tron.sim.recorder import GameLog, LogError, Recorder
from tron.sim.result import MatchResult

API_VERSION = "1.0"
"""Version de l'API publique (indépendante de la version du paquet)."""

ConfigSource = str | Path | Mapping[str, Any] | GameConfig
"""Chemin YAML/JSON, dictionnaire ou configuration déjà chargée."""


def load_game_config(source: ConfigSource) -> GameConfig:
    """Charge une configuration de partie (fichier YAML ou JSON, dictionnaire, ou objet)."""
    if isinstance(source, GameConfig):
        return source
    if isinstance(source, Mapping):
        return GameConfig.from_dict(source)
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise ConfigError(f"configuration introuvable : {path}")
        return GameConfig.from_yaml(path)
    raise ConfigError(f"source de configuration invalide : {type(source).__name__}")


def _configured(config: ConfigSource, seed: int | None, first_player: int | None) -> GameConfig:
    cfg = load_game_config(config)
    if seed is not None or first_player is not None:
        cfg = cfg.with_seed(cfg.seed if seed is None else seed, first_player)
    return cfg


def create_match(
    config: ConfigSource,
    observers: Sequence[MatchObserver] = (),
    *,
    seed: int | None = None,
    first_player: int | None = None,
    agents: Sequence[Agent] | None = None,
    metrics: bool = True,
) -> Match:
    """Prépare une partie (sans la jouer) : ``match.step()`` ou ``match.run()`` ensuite.

    ``agents`` remplace les agents de la configuration (un par joueur, dans
    l'ordre) ; ``metrics=False`` désactive les métriques de coopération.
    """
    cfg = _configured(config, seed, first_player)
    if agents is not None and len(agents) != cfg.n_players:
        raise ConfigError(f"{len(agents)} agents pour {cfg.n_players} joueurs")
    return Match(cfg, agents=agents, observers=observers, metrics=metrics)


def run_game(
    config: ConfigSource,
    seed: int | None = None,
    observers: Sequence[MatchObserver] = (),
    *,
    first_player: int | None = None,
    log: str | Path | None = None,
    metrics: bool = True,
) -> MatchResult:
    """Joue une partie sans affichage ; ``log`` écrit son journal JSONL rejouable."""
    obs = list(observers)
    if log is not None:
        obs.append(Recorder(log))
    return create_match(config, obs, seed=seed, first_player=first_player, metrics=metrics).run()


def _play_one(task: tuple[GameConfig, int, int, str | None]) -> tuple[int, MatchResult]:
    """Partie ``game`` d'une série (fonction de premier niveau : exécutable dans un processus)."""
    config, game, games, log_dir = task
    observers: list[MatchObserver] = []
    if log_dir is not None:
        observers.append(Recorder(Path(log_dir) / f"game_{game:04d}.jsonl"))
    return game, Match(game_config(config, game, games), observers=observers).run()


def run_games(
    config: ConfigSource,
    games: int,
    workers: int = 1,
    out_dir: str | Path | None = None,
    *,
    logs: bool = False,
    on_result: Callable[[int, MatchResult], None] | None = None,
) -> list[MatchResult]:
    """Joue ``games`` parties (graines dérivées, premier joueur tournant, comme ``tron play``).

    Avec ``workers > 1``, les parties sont réparties sur des processus ; les
    résultats sont identiques à l'exécution séquentielle (hors durées mesurées)
    et rendus dans l'ordre des parties. ``out_dir`` reçoit ``results.csv`` et,
    avec ``logs=True``, un journal ``game_XXXX.jsonl`` par partie.
    """
    cfg = load_game_config(config)
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ConfigError(f"games doit être un entier ≥ 1 : {games!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"workers doit être un entier ≥ 1 : {workers!r}")
    if workers == 1:
        return run_arena(cfg, games, out_dir, logs=logs, on_result=on_result)
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    log_dir = str(out) if out is not None and logs else None
    tasks = [(cfg, g, games, log_dir) for g in range(games)]
    results: dict[int, MatchResult] = {}
    with ProcessPoolExecutor(max_workers=min(workers, games)) as pool:
        for g, result in pool.map(_play_one, tasks):
            results[g] = result
            if on_result is not None:
                on_result(g, result)
    ordered = [results[g] for g in range(games)]
    if out is not None:
        rows = [{"game": g, **r.to_row()} for g, r in enumerate(ordered)]
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return ordered


def load_replay(path: str | Path) -> GameLog:
    """Charge un journal JSONL (``tron-log/1`` ou ``tron-log/2``) pour le rejouer.

    ``log.states()`` itère les états successifs, ``log.final_state()`` donne
    l'état final ; une incohérence entre le journal et les règles lève
    :class:`LogError`.
    """
    p = Path(path)
    if not p.is_file():
        raise LogError(f"journal introuvable : {p}")
    return GameLog.load(p)


def resume_game(path: str | Path, observers: Sequence[MatchObserver] = ()) -> Match:
    """Reprend une partie sauvegardée (``tron-checkpoint/1``) ; ``run()`` la termine."""
    return load_checkpoint(path, observers)


__all__ = [
    "AGENTS",
    "API_VERSION",
    "HEURISTICS",
    "Agent",
    "CheckpointError",
    "ConfigError",
    "ConfigSource",
    "Decision",
    "DecisionDetails",
    "GameConfig",
    "GameLog",
    "GameState",
    "Heuristic",
    "LogError",
    "Match",
    "MatchObserver",
    "MatchResult",
    "ModelError",
    "MoveEvent",
    "Outcome",
    "Recorder",
    "SearchStats",
    "build_agents",
    "create_match",
    "load_game_config",
    "load_replay",
    "make_agent",
    "make_heuristic",
    "register_agent",
    "register_heuristic",
    "resume_game",
    "run_game",
    "run_games",
]
