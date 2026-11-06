"""Arène : une ou plusieurs parties d'une même configuration (F5.2).

Chaque partie reçoit une graine dérivée de la graine de la configuration et du
numéro de partie, et le premier joueur tourne d'une partie à l'autre (F1.2). Un
résultat CSV et, sur demande, un journal JSONL rejouable sont produits par partie.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from tron.core.config import GameConfig
from tron.core.rng import derive_seed
from tron.sim.match import Match, MatchObserver
from tron.sim.recorder import Recorder
from tron.sim.result import MatchResult


def game_config(config: GameConfig, game: int, games: int) -> GameConfig:
    """Configuration de la partie ``game`` : graine dérivée et premier joueur tournant."""
    if games == 1:
        return config
    first = (config.first_player + game) % config.n_players
    return config.with_seed(derive_seed(config.seed, game), first_player=first)


def run_arena(
    config: GameConfig,
    games: int = 1,
    out_dir: str | Path | None = None,
    logs: bool = True,
    on_result: Callable[[int, MatchResult], None] | None = None,
) -> list[MatchResult]:
    """Joue ``games`` parties ; écrit ``results.csv`` et ``game_XXXX.jsonl`` dans ``out_dir``."""
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    results: list[MatchResult] = []
    writer = None
    fh = None
    try:
        for g in range(games):
            cfg = game_config(config, g, games)
            observers: list[MatchObserver] = []
            if out is not None and logs:
                observers.append(Recorder(out / f"game_{g:04d}.jsonl"))
            result = Match(cfg, observers=observers).run()
            results.append(result)
            if out is not None:
                row = {"game": g, **result.to_row()}
                if writer is None:
                    fh = (out / "results.csv").open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                assert fh is not None
                fh.flush()
            if on_result is not None:
                on_result(g, result)
    finally:
        if fh is not None:
            fh.close()
    return results
