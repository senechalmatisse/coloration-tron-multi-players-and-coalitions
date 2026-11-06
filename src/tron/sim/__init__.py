"""Simulation : boucle de partie, journal JSONL, rejeu, arène."""

from tron.sim.arena import game_config, run_arena
from tron.sim.match import Match, MatchObserver, MoveEvent, ensure_warmup
from tron.sim.recorder import LOG_FORMAT, GameLog, LogError, Recorder
from tron.sim.result import MatchResult, PlayerTotals, environment_info, git_hash

__all__ = [
    "LOG_FORMAT",
    "GameLog",
    "LogError",
    "Match",
    "MatchObserver",
    "MatchResult",
    "MoveEvent",
    "PlayerTotals",
    "Recorder",
    "ensure_warmup",
    "environment_info",
    "game_config",
    "git_hash",
    "run_arena",
]
