"""Rejoue un journal JSONL coup par coup (API publique, P11).

tron play configs/games/duel_20x20.yaml --no-gui --out /tmp/duel
python examples/rejeu.py /tmp/duel/game_0000.jsonl
"""

from __future__ import annotations

import sys

from tron.api import load_replay


def main(log_path: str) -> None:
    log = load_replay(log_path)
    print(f"{log.config.name} ({log.format}) : {len(log.moves)} demi-coups, agents {log.agents}")
    for ply, state in enumerate(log.states()):
        if ply % 20 == 0 or state.is_terminal():
            alive = [p for p in range(state.n) if state.is_alive(p)]
            print(f"demi-coup {ply:4d} : joueurs vivants {alive}")
    out = log.final_state().outcome()
    print(f"équipe gagnante : {out.winning_team if out else '?'}")


if __name__ == "__main__":
    main(sys.argv[1])
