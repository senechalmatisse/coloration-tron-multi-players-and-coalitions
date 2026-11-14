"""Agents : interface commune, agents de référence, Negamax, Paranoid, MaxN, SOS, MCTS."""

from tron.agents.adaptive import AdaptiveAgent, adapted_weights, base_weights
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
from tron.agents.context import (
    AllianceEvent,
    DecisionContext,
    DiplomacyIntent,
    DiplomacySnapshot,
)
from tron.agents.learned import LearnedAgent
from tron.agents.mcts import ROLLOUT_POLICIES, MCTSAgent
from tron.agents.opponents import BEHAVIOURS, MoveFeatures, OpponentModel
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
    "BEHAVIOURS",
    "PROFILES",
    "ROLLOUT_POLICIES",
    "AdaptiveAgent",
    "Agent",
    "AllianceEvent",
    "Decision",
    "DecisionContext",
    "DecisionDetails",
    "DiplomacyIntent",
    "DiplomacySnapshot",
    "GreedyAgent",
    "LearnedAgent",
    "LinearBound",
    "MCTSAgent",
    "MaxNAgent",
    "MoveFeatures",
    "NegamaxAgent",
    "OpponentModel",
    "ParanoidAgent",
    "RandomAgent",
    "SearchAgent",
    "SearchMemory",
    "SearchOptions",
    "SearchResult",
    "SearchStats",
    "SocialAgent",
    "TwoCamp",
    "adapted_weights",
    "base_weights",
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

# Extensions installées : chargées une fois, après tous les agents intégrés.
from tron.plugins import AGENT_GROUP as _AGENT_GROUP
from tron.plugins import load_plugins as _load_plugins

_load_plugins(_AGENT_GROUP)
