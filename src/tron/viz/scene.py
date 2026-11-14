"""Scène observée, sans bibliothèque graphique : traces, orientations, événements visuels.

:class:`SceneTracker` reconstruit les traces de chaque joueur à partir de
l'historique de l'état observé (:attr:`GameState.history`), de façon
incrémentale : un demi-coup joué ajoute une case, un retour arrière (rejeu) en
retire une, une nouvelle partie remet tout à zéro. Il en déduit les événements
visuels (déplacement, virage, élimination et sa cause, alliance, zone,
obstacles, victoire) que consomment les effets et le journal du HUD.

Le moteur ignore tout de ce module : il ne fait qu'observer un état.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tron.agents.context import AllianceEvent
from tron.core.directions import DX, DY, PASS
from tron.core.rules import OBSTACLE
from tron.core.state import GameState

Cell = tuple[int, int]
HistoryEntry = tuple[int, int, int, int]

NORTH_BIT, EAST_BIT, SOUTH_BIT, WEST_BIT = 1, 2, 4, 8
"""Bits de connexion d'une case de trace (bit ``1 << direction``)."""


def heading_between(a: Cell, b: Cell) -> int:
    """Direction (0 N, 1 E, 2 S, 3 O) du pas de ``a`` vers la case voisine ``b`` (−1 sinon)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    for d in range(4):
        if (DX[d], DY[d]) == (dx, dy):
            return d
    return -1


def opposite(direction: int) -> int:
    return (direction + 2) % 4 if direction >= 0 else -1


def path_mask(path: Sequence[Cell], i: int) -> int:
    """Connexions de la ``i``-ème case d'une trace vers ses voisines dans la trace."""
    mask = 0
    if i > 0:
        d = heading_between(path[i], path[i - 1])
        if d >= 0:
            mask |= 1 << d
    if i + 1 < len(path):
        d = heading_between(path[i], path[i + 1])
        if d >= 0:
            mask |= 1 << d
    return mask


@dataclass(frozen=True, slots=True)
class PlyExtras:
    """Informations d'un demi-coup absentes de l'état : direction jouée, diplomatie, zones."""

    direction: int | None = None
    diplomacy: tuple[AllianceEvent, ...] = ()
    zone: tuple[tuple[int, int], ...] = ()
    obstacles: tuple[Cell, ...] = ()


def record_extras(record: Mapping[str, Any], direction: int | None = None) -> PlyExtras:
    """Informations d'une ligne ``move`` d'un journal (``direction`` : coup décodé)."""
    events = []
    for raw in record.get("diplomacy", ()):
        try:
            events.append(AllianceEvent.from_dict(raw))
        except (KeyError, TypeError, ValueError):
            continue
    zone = tuple((int(z[0]), int(z[1])) for z in record.get("zone", ()))
    obstacles = tuple((int(c[0]), int(c[1])) for c in record.get("obstacles", ()))
    return PlyExtras(direction, tuple(events), zone, obstacles)


@dataclass(frozen=True, slots=True)
class VisualEvent:
    """Événement à mettre en scène (effets, journal du HUD).

    ``kind`` : ``move``, ``turn``, ``elimination``, ``alliance``, ``rupture``,
    ``zone``, ``obstacles`` ou ``victory``. ``point`` est une position en unités
    de case (origine au coin haut gauche de la grille, y vers le bas).
    """

    kind: str
    ply: int
    player: int = -1
    other: int = -1
    """Joueur heurté (élimination), allié (alliance), équipe (zone, victoire)."""
    heading: int = -1
    cause: str = ""
    """Cause d'une élimination : ``bordure``, ``trace``, ``propre trace``, ``obstacle``,
    ``bloqué`` (vide si la direction jouée est inconnue)."""
    point: tuple[float, float] = (0.0, 0.0)
    cells: tuple[Cell, ...] = ()
    index: int = -1
    """Numéro de la zone (événement ``zone``)."""


@dataclass(slots=True)
class SyncResult:
    """Changements constatés par :meth:`SceneTracker.sync`."""

    reset: bool = False
    added: list[HistoryEntry] = field(default_factory=list)
    removed: list[HistoryEntry] = field(default_factory=list)
    events: list[VisualEvent] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.reset or bool(self.added) or bool(self.removed)

    @property
    def rewound(self) -> bool:
        return bool(self.removed)


def _center(cell: Cell) -> tuple[float, float]:
    return cell[0] + 0.5, cell[1] + 0.5


class SceneTracker:
    """Traces des joueurs reconstruites de l'historique de l'état observé.

    Si l'historique n'explique pas la grille (copie sans historique, état écrit
    à la main), le suivi passe en *mode grille* : les cases de chaque joueur
    sont lues sur la grille et reliées à leurs voisines de même propriétaire,
    sans ordre de parcours ni événements.
    """

    def __init__(self) -> None:
        self._state: GameState | None = None
        self._positions: tuple[Cell, ...] = ()
        self.entries: list[HistoryEntry] = []
        self._hpaths: list[list[Cell]] = []
        self.paths: list[list[Cell]] = []
        """Cases de chaque trace, dans l'ordre du parcours (tête en dernier)."""
        self.headings: list[int] = []
        self.alive: list[bool] = []
        self.terminal = False
        self.grid_mode = False
        self._grid_masks: list[list[int]] = []
        self._grid_key: int | None = None
        self.version = 0
        """Incrémenté à chaque changement (cache des rendus)."""

    # --------------------------------------------------------------- lecture
    @property
    def n(self) -> int:
        return len(self.paths)

    def mask(self, player: int, i: int) -> int:
        """Connexions (bits N, E, S, O) de la ``i``-ème case de la trace de ``player``."""
        if self.grid_mode:
            return self._grid_masks[player][i]
        return path_mask(self.paths[player], i)

    def head(self, player: int) -> Cell:
        return self.paths[player][-1]

    # ------------------------------------------------------------ synchronisation
    def reset(self, state: GameState) -> None:
        self._state = state
        self._positions = tuple(state.positions)
        self.entries = []
        self._hpaths = [[pos] for pos in state.positions]
        self.paths = self._hpaths
        self.headings = [-1] * state.n
        self.alive = [True] * state.n
        self.terminal = False
        self.grid_mode = False
        self._grid_key = None
        self.version += 1

    def sync(
        self,
        state: GameState,
        extras: Callable[[int], PlyExtras] | None = None,
        events: bool = True,
    ) -> SyncResult:
        """Rattrape l'état observé ; ``extras(ply)`` complète les événements visuels."""
        result = SyncResult()
        if self._state is not state or self._positions != tuple(state.positions):
            self.reset(state)
            result.reset = True
        hist = state.history
        while self.entries and (
            len(self.entries) > len(hist) or self.entries[-1] != hist[len(self.entries) - 1]
        ):
            entry = self.entries.pop()
            self._undo(entry)
            result.removed.append(entry)
        for i in range(len(self.entries), len(hist)):
            entry = hist[i]
            self.entries.append(entry)
            found = self._apply(state, entry, i, extras, events)
            result.added.append(entry)
            result.events.extend(found)
        terminal = state.is_terminal()
        if events and terminal and not self.terminal:
            result.events.append(_victory(state))
        self.terminal = terminal
        self._check_grid(state, result)
        if result.changed:
            self.version += 1
        return result

    def _check_grid(self, state: GameState, result: SyncResult) -> None:
        """Passe en mode grille si l'historique n'explique pas les cases occupées."""
        g = state.grid2d()
        owned = int(np.count_nonzero((g >= 0) & (g < state.n)))
        explained = sum(len(path) for path in self._hpaths)
        if owned == explained:
            if self.grid_mode:
                self.grid_mode = False
                self.paths = self._hpaths
                result.reset = True
            return
        key = state.key
        if self.grid_mode and key == self._grid_key:
            return
        self.grid_mode = True
        self._grid_key = key
        self.paths, self._grid_masks = _grid_paths(state, g)
        result.reset = True
        result.events.clear()

    def _apply(
        self,
        state: GameState,
        entry: HistoryEntry,
        ply: int,
        extras: Callable[[int], PlyExtras] | None,
        emit: bool,
    ) -> list[VisualEvent]:
        p, h, t, _ = entry
        out: list[VisualEvent] = []
        info = extras(ply) if extras is not None and emit else PlyExtras()
        path = self._hpaths[p]
        if t >= 0:
            cell = state.coords(t)
            prev = path[-1]
            path.append(cell)
            d = heading_between(prev, cell)
            turned = self.headings[p] >= 0 and d != self.headings[p]
            self.headings[p] = d
            if emit:
                out.append(
                    VisualEvent(
                        "turn" if turned else "move",
                        ply,
                        player=p,
                        heading=d,
                        point=_center(prev),
                        cells=(prev, cell),
                    )
                )
        else:
            self.alive[p] = False
            if emit:
                out.append(_elimination(state, p, state.coords(h), info.direction, ply))
        if not emit:
            return out
        for ev in info.diplomacy:
            if ev.kind == "formed":
                out.append(VisualEvent("alliance", ply, player=ev.player, other=ev.target))
            elif ev.kind == "broken":
                out.append(VisualEvent("rupture", ply, player=ev.player, other=ev.target))
        for index, team in info.zone:
            out.append(VisualEvent("zone", ply, other=team, index=index))
        if info.obstacles:
            out.append(VisualEvent("obstacles", ply, cells=info.obstacles))
        return out

    def _undo(self, entry: HistoryEntry) -> None:
        p, _, t, _ = entry
        if t >= 0:
            path = self._hpaths[p]
            path.pop()
            self.headings[p] = heading_between(path[-2], path[-1]) if len(path) >= 2 else -1
        else:
            self.alive[p] = True


def _grid_paths(state: GameState, g: np.ndarray) -> tuple[list[list[Cell]], list[list[int]]]:
    """Cases de chaque joueur lues sur la grille (tête en dernier) et leurs connexions."""
    paths: list[list[Cell]] = []
    masks: list[list[int]] = []
    h, w = g.shape
    for p in range(state.n):
        head = state.head(p)
        cells = [(int(x), int(y)) for y, x in np.argwhere(g == p) if (int(x), int(y)) != head]
        cells.append(head)
        paths.append(cells)
        row = []
        for x, y in cells:
            mask = 0
            for d in range(4):
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < w and 0 <= ny < h and g[ny, nx] == p:
                    mask |= 1 << d
            row.append(mask)
        masks.append(row)
    return paths, masks


def _elimination(
    state: GameState, player: int, head: Cell, direction: int | None, ply: int
) -> VisualEvent:
    """Élimination de ``player`` et ce qu'il a heurté (si la direction jouée est connue)."""
    x, y = head
    if direction is None:
        return VisualEvent("elimination", ply, player=player, point=_center(head), cells=(head,))
    if direction == PASS or not 0 <= direction <= 3:
        return VisualEvent(
            "elimination", ply, player=player, cause="bloqué", point=_center(head), cells=(head,)
        )
    tx, ty = x + DX[direction], y + DY[direction]
    point = (x + 0.5 + 0.5 * DX[direction], y + 0.5 + 0.5 * DY[direction])
    other = -1
    if not (0 <= tx < state.width and 0 <= ty < state.height):
        cause = "bordure"
    else:
        v = state.cell(tx, ty)
        if v == OBSTACLE:
            cause = "obstacle"
        elif v == player:
            cause = "propre trace"
        elif 0 <= v < state.n:
            cause, other = "trace", v
        else:  # pragma: no cover - case libre : coup légal, pas d'élimination
            cause = ""
    return VisualEvent(
        "elimination",
        ply,
        player=player,
        other=other,
        heading=direction,
        cause=cause,
        point=point,
        cells=(head,),
    )


def _victory(state: GameState) -> VisualEvent:
    out = state.outcome()
    winners = out.winners if out is not None else ()
    team = out.winning_team if out is not None else -1
    return VisualEvent(
        "victory",
        state.ply,
        other=team,
        cells=tuple(state.head(p) for p in winners),
    )


# --------------------------------------------------------------- coalitions
def alliance_pairs(events: Iterable[AllianceEvent]) -> tuple[tuple[int, int], ...]:
    """Alliances actives après une suite d'événements diplomatiques."""
    active: set[tuple[int, int]] = set()
    for ev in events:
        pair = (min(ev.player, ev.target), max(ev.player, ev.target))
        if ev.kind == "formed":
            active.add(pair)
        elif ev.kind in ("broken", "dissolved"):
            active.discard(pair)
        if ev.kind == "dissolved":  # le joueur éliminé perd toutes ses alliances
            active = {q for q in active if ev.player not in q}
    return tuple(sorted(active))


def team_links(state: GameState) -> tuple[tuple[int, int], ...]:
    """Paires de coéquipiers (équipes permanentes) encore en vie."""
    team = state.config.team_index
    alive = state.alive_players()
    return tuple((a, b) for i, a in enumerate(alive) for b in alive[i + 1 :] if team[a] == team[b])


# --------------------------------------------------------------- territoires
def territory_edges(owners: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Frontières entre territoires de Voronoi : segments ``(x0, y0, x1, y1)`` en cases.

    Une frontière sépare deux cases voisines de propriétaires différents dont
    l'un au moins est un joueur (−1 : mur ou case neutre).
    """
    edges: list[tuple[float, float, float, float]] = []
    h, w = owners.shape
    diff_x = (owners[:, 1:] != owners[:, :-1]) & ((owners[:, 1:] >= 0) | (owners[:, :-1] >= 0))
    for y, x in np.argwhere(diff_x):
        edges.append((float(x + 1), float(y), float(x + 1), float(y + 1)))
    diff_y = (owners[1:, :] != owners[:-1, :]) & ((owners[1:, :] >= 0) | (owners[:-1, :] >= 0))
    for y, x in np.argwhere(diff_y):
        edges.append((float(x), float(y + 1), float(x + 1), float(y + 1)))
    del h, w
    return edges


@dataclass(frozen=True, slots=True)
class ArenaStats:
    """Statistiques affichées par le HUD."""

    active: int
    total: int
    occupied: float
    """Part des cases occupées (traces et obstacles)."""
    eliminations: int
    alliances: int
    ply: int
    round: int


def arena_stats(state: GameState, alliances: int = 0) -> ArenaStats:
    free = int(np.count_nonzero(state.grid2d() == -1))
    cells = state.width * state.height
    return ArenaStats(
        active=len(state.alive_players()),
        total=state.n,
        occupied=1.0 - free / cells if cells else 0.0,
        eliminations=len(state.elim_order),
        alliances=alliances,
        ply=state.ply,
        round=state.round,
    )


__all__ = [
    "ArenaStats",
    "PlyExtras",
    "SceneTracker",
    "SyncResult",
    "VisualEvent",
    "alliance_pairs",
    "arena_stats",
    "heading_between",
    "opposite",
    "path_mask",
    "record_extras",
    "team_links",
    "territory_edges",
]
