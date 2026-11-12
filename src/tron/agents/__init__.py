"""Agents : interface commune, agents de référence, Negamax, Paranoid, MaxN, SOS, MCTS."""

from tron.agents.base import (
    AGENTS,
    Agent,
    Decision,
    SearchStats,
    build_agents,
    make_agent,
    register_agent,
)
from tron.agents.context import (
    AllianceEvent,
    DecisionContext,
    DiplomacyIntent,
    DiplomacySnapshot,
)
from tron.agents.mcts import ROLLOUT_POLICIES, MCTSAgent
from tron.agents.search import SearchMemory, SearchOptions, SearchResult, search
from tron.agents.searchers import MaxNAgent, NegamaxAgent, ParanoidAgent, SearchAgent, SocialAgent
from tron.agents.simple import GreedyAgent, RandomAgent
from tron.agents.social import (
    PROFILES,
    LinearBound,
    TwoCamp,
    dynamic_matrix,
    lp_upper,
    social_matrix,
    tactical_intent,
    two_camp_reduction,
)

__all__ = [
    "AGENTS",
    "PROFILES",
    "ROLLOUT_POLICIES",
    "Agent",
    "AllianceEvent",
    "Decision",
    "DecisionContext",
    "DiplomacyIntent",
    "DiplomacySnapshot",
    "GreedyAgent",
    "LinearBound",
    "MCTSAgent",
    "MaxNAgent",
    "NegamaxAgent",
    "ParanoidAgent",
    "RandomAgent",
    "SearchAgent",
    "SearchMemory",
    "SearchOptions",
    "SearchResult",
    "SearchStats",
    "SocialAgent",
    "TwoCamp",
    "build_agents",
    "dynamic_matrix",
    "lp_upper",
    "make_agent",
    "register_agent",
    "search",
    "social_matrix",
    "tactical_intent",
    "two_camp_reduction",
]
