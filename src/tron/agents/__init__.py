"""Agents : interface commune, agents de référence, Negamax, Paranoid, MaxN, SOS."""

from tron.agents.base import (
    AGENTS,
    Agent,
    Decision,
    SearchStats,
    build_agents,
    make_agent,
    register_agent,
)
from tron.agents.search import SearchMemory, SearchOptions, SearchResult, search
from tron.agents.searchers import MaxNAgent, NegamaxAgent, ParanoidAgent, SearchAgent, SocialAgent
from tron.agents.simple import GreedyAgent, RandomAgent
from tron.agents.social import (
    PROFILES,
    LinearBound,
    TwoCamp,
    lp_upper,
    social_matrix,
    two_camp_reduction,
)

__all__ = [
    "AGENTS",
    "PROFILES",
    "Agent",
    "Decision",
    "GreedyAgent",
    "LinearBound",
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
    "lp_upper",
    "make_agent",
    "register_agent",
    "search",
    "social_matrix",
    "two_camp_reduction",
]
