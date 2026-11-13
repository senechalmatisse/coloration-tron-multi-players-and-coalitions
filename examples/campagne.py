"""Plusieurs parties d'une configuration, en parallèle, et un résumé (API publique, P11).

python examples/campagne.py configs/games/teams_2v2_20x20.yaml 8 2
"""

from __future__ import annotations

import sys
from collections import Counter

from tron.api import load_game_config, run_games


def main(config_path: str, games: int, workers: int) -> None:
    config = load_game_config(config_path)
    results = run_games(config, games, workers=workers, out_dir=None)
    wins = Counter(r.winning_team or "égalité" for r in results)
    mean = sum(r.n_moves for r in results) / len(results)
    print(f"{config.name} : {games} parties, victoires {dict(wins)}, {mean:.1f} coups en moyenne")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        args[0] if args else "configs/games/teams_2v2_20x20.yaml",
        int(args[1]) if len(args) > 1 else 4,
        int(args[2]) if len(args) > 2 else 2,
    )
