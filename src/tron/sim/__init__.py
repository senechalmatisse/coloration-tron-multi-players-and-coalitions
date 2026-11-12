"""Simulation : boucle de partie, journal JSONL, rejeu, arène, métriques de partie."""

from tron.sim.arena import game_config, run_arena
from tron.sim.diplomacy import (
    AllianceEvent,
    DiplomacyError,
    DiplomacyIntent,
    DiplomacySnapshot,
    DiplomacyState,
    DiplomacySummary,
    summarize_diplomacy,
)
from tron.sim.match import Match, MatchObserver, MoveEvent, ensure_warmup
from tron.sim.metrics import MetricsCollector, metrics_from_moves
from tron.sim.recorder import LOG_FORMAT, READABLE_FORMATS, GameLog, LogError, Recorder
from tron.sim.result import (
    METRICS_VERSION,
    MatchMetrics,
    MatchResult,
    PlayerTotals,
    environment_info,
    git_hash,
)

__all__ = [
    "LOG_FORMAT",
    "METRICS_VERSION",
    "READABLE_FORMATS",
    "AllianceEvent",
    "DiplomacyError",
    "DiplomacyIntent",
    "DiplomacySnapshot",
    "DiplomacyState",
    "DiplomacySummary",
    "GameLog",
    "LogError",
    "Match",
    "MatchMetrics",
    "MatchObserver",
    "MatchResult",
    "MetricsCollector",
    "MoveEvent",
    "PlayerTotals",
    "Recorder",
    "ensure_warmup",
    "environment_info",
    "game_config",
    "git_hash",
    "metrics_from_moves",
    "run_arena",
    "summarize_diplomacy",
]
