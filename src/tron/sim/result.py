"""Résultat d'une partie."""

from __future__ import annotations

import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

import tron
from tron._jit import backend_name
from tron.schema import (
    DIPLOMACY_COLUMNS,
    MAX_PLAYERS,
    MAX_TEAMS,
    METRICS_VERSION,
    TEAM_METRIC_COLUMNS,
)

__all__ = [
    "DIPLOMACY_COLUMNS",
    "MAX_PLAYERS",
    "MAX_TEAMS",
    "METRICS_VERSION",
    "TEAM_METRIC_COLUMNS",
    "MatchMetrics",
    "MatchResult",
    "PlayerTotals",
    "environment_info",
    "git_hash",
    "metrics_row",
    "nan_to_none",
]


@cache
def git_hash() -> str:
    """Hash Git du dépôt du code (``unknown`` hors dépôt)."""
    root = Path(tron.__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


@cache
def environment_info() -> dict[str, str]:
    """Versions du code et des bibliothèques."""
    return {
        "code_version": tron.__version__,
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "backend": backend_name(),
    }


@dataclass(frozen=True, slots=True)
class PlayerTotals:
    """Cumul des statistiques de décision d'un joueur sur une partie."""

    decisions: int = 0
    nodes: int = 0
    think_time: float = 0.0
    max_depth: int = 0
    mean_depth: float = 0.0
    leaves: int = 0
    """Feuilles évaluées (recherches exactes) ou simulations (MCTS)."""


@dataclass(frozen=True, slots=True)
class MatchMetrics:
    """Métriques de coopération, de cohésion et de stabilité d'une partie.

    Les définitions opérationnelles sont dans :mod:`tron.sim.metrics`. Les
    tuples « par équipe » suivent l'ordre de ``GameConfig.team_names`` (l'équipe
    0 est celle du joueur 0) ; ``None`` signale une métrique non calculable.
    """

    version: int
    teams: tuple[str, ...]
    team_sizes: tuple[int, ...]
    survival_plies: tuple[int, ...]
    team_survival_plies: tuple[int, ...]
    team_rank: tuple[int, ...]
    intra_team_distance_mean: tuple[float | None, ...]
    ally_collisions: tuple[int, ...]
    blocking_moves: tuple[int, ...]
    team_moves: tuple[int, ...]
    cooperation_proxy: tuple[float | None, ...]
    territory_share_mean: tuple[float | None, ...]
    collective_vs_individual_gap: tuple[float | None, ...]
    territory_observations: int
    diplomacy_enabled: bool = False
    proposals: int = 0
    alliances_formed: int = 0
    alliance_breaks: int = 0
    betrayals: int = 0
    rejected_intents: int = 0
    alliance_duration_mean: float | None = None

    def team_value(self, field_name: str, team: int) -> Any:
        values: tuple[Any, ...] = getattr(self, field_name)
        return values[team]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Résultat complet d'une partie."""

    name: str
    seed: int
    first_player: int
    width: int
    height: int
    players: tuple[str, ...]
    teams: tuple[str, ...]
    agents: tuple[str, ...]
    depths: tuple[int, ...]
    winning_team: str
    winners: tuple[int, ...]
    ranks: tuple[int, ...]
    elimination_order: tuple[int, ...]
    n_moves: int
    n_plies: int
    duration: float
    totals: tuple[PlayerTotals, ...]
    positions: tuple[tuple[int, int], ...]
    metrics: MatchMetrics | None = None
    mode: str = "classic"
    """Mode de jeu."""
    individual_winner: int | None = None
    """Vainqueur individuel : dernier survivant en survie individuelle, ou seul survivant."""
    survival_order: tuple[int, ...] = ()
    """Joueurs du plus long au plus court temps de survie (à égalité, ordre des indices)."""
    rounds: int = 0
    """Tours complets joués (modes à règles ; 0 sinon)."""
    scores: tuple[int, ...] = ()
    """Score de chaque équipe de victoire (contrôle de zone) ; vide sinon."""
    obstacles: int = 0
    """Obstacles dynamiques posés pendant la partie."""
    end_reason: str = "elimination"
    """Cause de fin : ``elimination``, ``survive``, ``target_score`` ou ``max_rounds``."""

    @property
    def n_players(self) -> int:
        return len(self.players)

    def team_won(self, team: str) -> bool:
        return self.winning_team == team

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["totals"] = [asdict(t) for t in self.totals]
        data["positions"] = [list(p) for p in self.positions]
        data["metrics"] = self.metrics.to_dict() if self.metrics is not None else None
        return data

    def to_row(self, max_players: int = MAX_PLAYERS) -> dict[str, Any]:
        """Une ligne CSV : champs de la partie puis colonnes par joueur (``_p0`` … ``_p7``)."""
        row: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "first_player": self.first_player,
            "width": self.width,
            "height": self.height,
            "n_players": self.n_players,
            "n_teams": len(set(self.teams)),
            "winning_team": self.winning_team,
            "winners": ";".join(str(w) for w in self.winners),
            "focus_team": self.teams[0],
            "focus_win": int(self.winning_team == self.teams[0]),
            "n_moves": self.n_moves,
            "n_plies": self.n_plies,
            "duration": round(self.duration, 6),
        }
        for i in range(max_players):
            present = i < self.n_players
            tot = self.totals[i] if present else None
            row[f"team_p{i}"] = self.teams[i] if present else ""
            row[f"agent_p{i}"] = self.agents[i] if present else ""
            row[f"depth_p{i}"] = self.depths[i] if present else ""
            row[f"rank_p{i}"] = self.ranks[i] if present else ""
            row[f"decisions_p{i}"] = tot.decisions if tot else ""
            row[f"nodes_p{i}"] = tot.nodes if tot else ""
            row[f"time_p{i}"] = round(tot.think_time, 6) if tot else ""
            row[f"maxdepth_p{i}"] = tot.max_depth if tot else ""
        row.update(environment_info())
        for i in range(max_players):
            tot = self.totals[i] if i < self.n_players else None
            row[f"leaves_p{i}"] = tot.leaves if tot else ""
            row[f"meandepth_p{i}"] = round(tot.mean_depth, 4) if tot else ""
        row.update(metrics_row(self.metrics, max_players))
        row["rules_mode"] = self.mode
        row["individual_winner"] = "" if self.individual_winner is None else self.individual_winner
        row["survival_order"] = ";".join(str(p) for p in self.survival_order)
        row["rounds"] = self.rounds
        row["team_scores"] = ";".join(str(s) for s in self.scores)
        row["obstacles"] = self.obstacles
        row["end_reason"] = self.end_reason
        return row


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else ""
    return value


def metrics_row(metrics: MatchMetrics | None, max_players: int = MAX_PLAYERS) -> dict[str, Any]:
    """Colonnes CSV des métriques : toujours les mêmes, vides si ``metrics`` est ``None``.

    ``survival_p<i>`` par joueur, par équipe ``team_t<i>`` (nom) et les colonnes
    de :data:`TEAM_METRIC_COLUMNS`, puis la diplomatie (:data:`DIPLOMACY_COLUMNS`,
    ``dipl_enabled`` valant 0 ou 1). Le jeu de colonnes ne dépend ni de
    la partie ni de l'activation des métriques : un fichier de résultats garde
    un en-tête unique.
    """
    m = metrics
    row: dict[str, Any] = {"metrics_version": m.version if m is not None else ""}
    for i in range(max_players):
        present = m is not None and i < len(m.survival_plies)
        row[f"survival_p{i}"] = m.survival_plies[i] if m is not None and present else ""
    for t in range(MAX_TEAMS):
        present = m is not None and t < len(m.teams)
        row[f"team_t{t}"] = m.teams[t] if m is not None and present else ""
        for prefix, field_name in TEAM_METRIC_COLUMNS:
            value = m.team_value(field_name, t) if m is not None and present else None
            row[f"{prefix}_t{t}"] = _cell(value)
    for column, field_name in DIPLOMACY_COLUMNS:
        value = getattr(m, field_name) if m is not None else None
        row[column] = int(value) if isinstance(value, bool) else _cell(value)
    return row


def nan_to_none(value: float) -> float | None:
    """Convertit NaN et ±inf en ``None`` (JSON strict)."""
    return value if math.isfinite(value) else None
