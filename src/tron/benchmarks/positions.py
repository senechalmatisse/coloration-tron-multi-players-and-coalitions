"""Jeux de positions déterministes pour les mesures de performance (P13).

Chaque position vient d'une partie à coups aléatoires prudents (générateur de
:mod:`tron.core.rng`, graine dérivée de la graine du jeu, de la phase et du rang
de l'essai) jouée jusqu'au bout. Seuls les états « jouables » sont retenus :
partie non finie, au moins deux équipes en vie, joueur au trait avec un coup
légal. Quatre phases, définies par rapport à la partie (donc valables pour
toutes les tailles de grille) :

* ``opening`` : premier état rempli à 2–8 %, tous les joueurs vivants ;
* ``midgame`` : premier état après 35–50 % de la durée de la partie, joueurs
  encore en contact (au moins deux joueurs vivants peuvent atteindre une même
  case libre) ;
* ``partitioned`` : premier état cloisonné — les régions accessibles depuis les
  têtes vivantes sont deux à deux disjointes (propriété définitive une fois
  atteinte : les régions ne font que rétrécir) ;
* ``endgame`` : dernier état au moins 2·n demi-coups avant la fin (n joueurs)
  d'une partie d'au moins 4·n demi-coups.

Même graine, mêmes positions : l'empreinte (SHA-256 des instantanés) le vérifie.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from tron.core.config import GameConfig
from tron.core.rng import derive_seed, make_rng
from tron.core.state import FREE, GameState

POSITION_SET_VERSION = "positions/1"
PHASES = ("opening", "midgame", "partitioned", "endgame")
STREAM_BENCH = 8
_MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class BenchPosition:
    """Une position du jeu : phase, rang, configuration et instantané."""

    phase: str
    index: int
    config: GameConfig
    snapshot: dict[str, Any]

    def state(self) -> GameState:
        """Nouvel état reconstruit depuis l'instantané (indépendant des autres)."""
        return GameState.from_snapshot(self.config, self.snapshot)


def benchmark_config(
    grid: int = 20, players: int = 4, seed: int = 0, heuristic: str = "voronoi"
) -> GameConfig:
    """Configuration des mesures : placement régulier, équipes alternées A/B."""
    return GameConfig.from_dict(
        {
            "name": f"bench_{grid}x{grid}_{players}p",
            "grid": grid,
            "seed": seed,
            "placement": {"mode": "regular", "rotate": True},
            "heuristic": {"name": heuristic},
            "players": [
                {"name": f"J{i}", "team": "AB"[i % 2] if players > 2 else f"T{i}"}
                for i in range(players)
            ],
        }
    )


def regions(state: GameState) -> list[set[int]]:
    """Cases libres accessibles depuis la tête de chaque joueur vivant (vide si éliminé)."""
    g = state._g
    out: list[set[int]] = []
    for p in range(state.n):
        seen: set[int] = set()
        if state.is_alive(p):
            queue = deque([int(state.heads[p])])
            while queue:
                c = queue.popleft()
                for off in state._off:
                    t = c + off
                    if t not in seen and g[t] == FREE:
                        seen.add(t)
                        queue.append(t)
        out.append(seen)
    return out


def is_partitioned(state: GameState) -> bool:
    """Vrai si les régions accessibles des joueurs vivants sont deux à deux disjointes."""
    live = [r for p, r in enumerate(regions(state)) if state.is_alive(p)]
    if len(live) < 2:
        return False
    return all(not (a & b) for i, a in enumerate(live) for b in live[i + 1 :])


def _playable(state: GameState) -> bool:
    return not state.is_terminal() and bool(state.legal_moves()) and state.teams_alive >= 2


def _policy(state: GameState, rng: Any) -> int:
    """Coup aléatoire prudent : parmi les coups légaux, ceux qui laissent le plus de
    voisins libres à la tête (une fois sur cinq : n'importe quel coup légal)."""
    moves = state.legal_moves()
    if not moves:
        return -1
    if len(moves) > 1 and float(rng.random()) >= 0.2:
        g = state._g
        head = int(state.heads[state.current])
        offs = state._off

        def free_after(m: int) -> int:
            t = head + offs[m]
            return sum(1 for o in offs if t + o != head and g[t + o] == FREE)

        best = max(free_after(m) for m in moves)
        moves = [m for m in moves if free_after(m) == best]
    return moves[int(rng.integers(len(moves)))]


def _play_out(cfg: GameConfig, rng: Any) -> tuple[list[dict[str, Any]], int]:
    """Instantanés des états jouables d'une partie complète et sa durée en demi-coups."""
    st = GameState(cfg)
    playable: list[dict[str, Any]] = []
    while not st.is_terminal():
        if _playable(st):
            playable.append(st.to_snapshot())
        st.apply(_policy(st, rng))
    return playable, st.ply


def _choose(
    phase: str, cfg: GameConfig, playable: list[dict[str, Any]], length: int, rng: Any
) -> dict[str, Any] | None:
    """Instantané de la phase dans la partie jouée, ``None`` si elle n'y figure pas."""
    n = cfg.n_players
    cells = cfg.width * cfg.height
    if not playable:
        return None
    if phase == "opening":
        target = (0.02 + 0.06 * float(rng.random())) * cells
        for snap in playable:
            if int(snap["n_moves"]) >= target:
                return snap if all(snap["alive"]) else None
        return None
    if phase == "midgame":
        start = (0.35 + 0.15 * float(rng.random())) * length
        later = [x for x in playable if int(x["ply"]) >= start]
        if not later or is_partitioned(GameState.from_snapshot(cfg, later[0])):
            return None
        return later[0]
    if phase == "partitioned":  # cloisonnement définitif : recherche dichotomique
        lo, hi = 0, len(playable)
        while lo < hi:
            mid = (lo + hi) // 2
            if is_partitioned(GameState.from_snapshot(cfg, playable[mid])):
                hi = mid
            else:
                lo = mid + 1
        return playable[lo] if lo < len(playable) else None
    if length < 4 * n:
        return None
    before = [x for x in playable if int(x["ply"]) <= length - 2 * n]
    return before[-1] if before else None


def position_set(
    grid: int = 20, players: int = 4, per_phase: int = 5, seed: int = 0
) -> list[BenchPosition]:
    """``per_phase`` positions par phase, déterministes pour (grid, players, seed)."""
    if per_phase < 1:
        raise ValueError("per_phase doit être ≥ 1")
    config = benchmark_config(grid, players, seed)
    out: list[BenchPosition] = []
    for ph, phase in enumerate(PHASES):
        found = 0
        attempt = 0
        while found < per_phase:
            if attempt >= _MAX_ATTEMPTS:
                raise RuntimeError(
                    f"phase {phase} introuvable sur {grid}x{grid}, {players} joueurs"
                )
            rng = make_rng(derive_seed(seed, STREAM_BENCH, ph, attempt))
            attempt += 1
            cfg = config.with_seed(derive_seed(seed, ph, attempt), int(rng.integers(players)))
            playable, length = _play_out(cfg, rng)
            snap = _choose(phase, cfg, playable, length, rng)
            if snap is not None:
                out.append(BenchPosition(phase, found, cfg, snap))
                found += 1
    return out


def fingerprint(positions: list[BenchPosition]) -> str:
    """Empreinte SHA-256 du jeu de positions (configurations et instantanés)."""
    payload = [
        {"phase": p.phase, "config": p.config.to_dict(), "snapshot": p.snapshot} for p in positions
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "PHASES",
    "POSITION_SET_VERSION",
    "BenchPosition",
    "benchmark_config",
    "fingerprint",
    "is_partitioned",
    "position_set",
    "regions",
]
