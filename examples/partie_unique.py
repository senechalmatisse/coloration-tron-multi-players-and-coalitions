"""Une partie sans affichage, son journal et son état final (API publique, P11).

python examples/partie_unique.py configs/games/duel_20x20.yaml
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from tron.api import load_game_config, load_replay, run_game


def main(config_path: str) -> None:
    config = load_game_config(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "partie.jsonl"
        result = run_game(config, seed=3, log=log)
        print(
            f"{config.name} : vainqueur {result.winning_team or 'aucun'} en {result.n_moves} coups"
        )
        final = load_replay(log).final_state()
        print(final.to_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/games/duel_20x20.yaml")
