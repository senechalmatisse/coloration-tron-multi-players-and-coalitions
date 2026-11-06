"""Résultat d'une partie (F1.5, F5.5)."""

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

MAX_PLAYERS = 8


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
    """Versions du code et des bibliothèques (traçabilité, §7.3)."""
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

    @property
    def n_players(self) -> int:
        return len(self.players)

    def team_won(self, team: str) -> bool:
        return self.winning_team == team

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["totals"] = [asdict(t) for t in self.totals]
        data["positions"] = [list(p) for p in self.positions]
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
        return row


def nan_to_none(value: float) -> float | None:
    """Convertit NaN et ±inf en ``None`` (JSON strict)."""
    return value if math.isfinite(value) else None
