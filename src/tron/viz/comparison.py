"""Comparaison synchronisée de deux rejeux, sans dépendance à Arcade.

Deux journaux JSONL (``tron-log/1`` ou ``tron-log/2``) sont rejoués côte à côte,
synchronisés par un curseur commun de demi-coup. Rien n'est recalculé par les
agents : seuls les coups enregistrés sont rejoués ; les parts de territoire sont
recalculées par Voronoi sur l'état affiché. Un journal plus court reste figé sur
son état final, ce que signale :attr:`SideView.frozen`.

La comparaison n'est dite **appariée** que si les deux parties partagent la
grille, le nombre de joueurs, les équipes par siège, les règles, la graine, le
placement initial et le premier joueur ; sinon :class:`Pairing` liste les écarts
et l'interface affiche un avertissement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.recorder import GameLog
from tron.viz.controller import ReplaySource
from tron.viz.palette import Palette, hex_to_rgb, make_palette
from tron.viz.render import render_frame


@dataclass(frozen=True)
class Pairing:
    """Résultat du contrôle d'appariement de deux parties."""

    paired: bool
    reasons: tuple[str, ...] = ()

    @property
    def warning(self) -> str:
        if self.paired:
            return ""
        return "comparaison non appariée : " + " ; ".join(self.reasons)


def pairing(a: GameLog, b: GameLog) -> Pairing:
    """Vérifie que deux parties sont comparables coup à coup (même départ, mêmes sièges)."""
    ca, cb = a.config, b.config
    reasons: list[str] = []
    if (ca.width, ca.height) != (cb.width, cb.height):
        reasons.append(f"grilles différentes ({ca.width}x{ca.height} ≠ {cb.width}x{cb.height})")
    if ca.n_players != cb.n_players:
        reasons.append(f"nombres de joueurs différents ({ca.n_players} ≠ {cb.n_players})")
    else:
        if [p.team for p in ca.players] != [p.team for p in cb.players]:
            reasons.append("équipes par siège différentes")
        if [tuple(p) for p in a.positions] != [tuple(p) for p in b.positions]:
            reasons.append("placements initiaux différents")
    if ca.seed != cb.seed:
        reasons.append(f"graines différentes ({ca.seed} ≠ {cb.seed})")
    if ca.first_player != cb.first_player:
        reasons.append(f"premiers joueurs différents ({ca.first_player} ≠ {cb.first_player})")
    if ca.rules.to_dict() != cb.rules.to_dict():
        reasons.append("règles différentes")
    return Pairing(not reasons, tuple(reasons))


def first_divergence(a: GameLog, b: GameLog) -> int | None:
    """Premier demi-coup où les parties diffèrent (joueur ou direction), ``None`` si identiques."""
    for i, (ra, rb) in enumerate(zip(a.moves, b.moves, strict=False)):
        if (ra.get("player"), ra.get("dir")) != (rb.get("player"), rb.get("dir")):
            return i
    if len(a.moves) != len(b.moves):
        return min(len(a.moves), len(b.moves))
    return None


@dataclass(frozen=True)
class TimelineEvent:
    ply: int
    kind: str
    """``elimination``, ``diplomacy``, ``zone`` ou ``obstacles``."""
    text: str


def timeline(log: GameLog) -> list[TimelineEvent]:
    """Événements du journal par demi-coup : éliminations, diplomatie, zones, obstacles."""
    names = [p.name for p in log.config.players]
    out: list[TimelineEvent] = []
    for i, rec in enumerate(log.moves):
        ply = i + 1  # état après le coup
        player = int(rec.get("player", 0))
        who = names[player] if 0 <= player < len(names) else str(player)
        if rec.get("elim"):
            out.append(TimelineEvent(ply, "elimination", f"{who} éliminé"))
        for ev in rec.get("diplomacy", ()):
            a, b = int(ev.get("player", -1)), int(ev.get("target", -1))
            na = names[a] if 0 <= a < len(names) else "?"
            nb = names[b] if 0 <= b < len(names) else "?"
            out.append(TimelineEvent(ply, "diplomacy", f"{ev.get('kind', '?')} {na} → {nb}"))
        for zone, team in rec.get("zone", ()):
            owner = log.config.victory_names[team] if team >= 0 else "personne"
            out.append(TimelineEvent(ply, "zone", f"zone {zone} → {owner}"))
        if rec.get("obstacles"):
            out.append(TimelineEvent(ply, "obstacles", f"{len(rec['obstacles'])} obstacle(s)"))
    return out


@dataclass
class PlayerTotals:
    decisions: int = 0
    time: float = 0.0
    nodes: int = 0


@dataclass
class SideView:
    """Ce qu'affiche un plateau au curseur courant."""

    label: str
    name: str
    seed: int
    first_player: int
    agents: list[str]
    seats: list[str]
    shown_ply: int
    length: int
    frozen: bool
    alive: list[bool]
    shares: list[float]
    team_shares: dict[str, float]
    totals: list[PlayerTotals]
    winner: str
    ranks: list[int]
    events_so_far: int
    events_total: int
    recent: list[TimelineEvent] = field(default_factory=list)


def _result(log: GameLog) -> tuple[str, list[int], int]:
    """Gagnant (nom d'équipe, vide si égalité), rangs et durée en coups de la partie."""
    res = log.result or {}
    if "winning_team" in res and "ranks" in res:
        ranks = [int(r) for r in res["ranks"]]
        return str(res.get("winning_team") or ""), ranks, int(res.get("n_moves", 0))
    final = log.final_state()
    out = final.outcome()
    if out is None:
        return "", [], final.n_moves
    winner = log.config.victory_names[out.winning_team] if out.winning_team >= 0 else ""
    return winner, list(out.ranks), final.n_moves


class ComparisonSession:
    """Deux rejeux synchronisés par un curseur commun ``ply``."""

    def __init__(self, log_a: GameLog, log_b: GameLog, labels: Sequence[str] = ("A", "B")):
        self.logs = (log_a, log_b)
        self.labels = (labels[0], labels[1])
        self.sources = (ReplaySource(log_a), ReplaySource(log_b))
        self._neon_cache: dict[tuple[int, int, str], Any] = {}
        self.pairing = pairing(log_a, log_b)
        self.divergence = first_divergence(log_a, log_b)
        self.timelines = (timeline(log_a), timeline(log_b))
        self._results = (_result(log_a), _result(log_b))
        self._heur = [
            VoronoiHeuristic(log.config.width, log.config.height, log.config.n_players)
            for log in self.logs
        ]
        self.ply = 0

    # curseur
    @property
    def max_ply(self) -> int:
        return max(len(log.moves) for log in self.logs)

    def seek(self, ply: int) -> int:
        """Place le curseur (borné) et rejoue chaque journal jusqu'à ``min(ply, sa longueur)``."""
        self.ply = max(0, min(int(ply), self.max_ply))
        for src in self.sources:
            src.seek(self.ply)
        return self.ply

    def forward(self) -> bool:
        if self.ply >= self.max_ply:
            return False
        self.seek(self.ply + 1)
        return True

    def backward(self) -> bool:
        if self.ply == 0:
            return False
        self.seek(self.ply - 1)
        return True

    def start(self) -> None:
        self.seek(0)

    def end(self) -> None:
        self.seek(self.max_ply)

    def frozen(self, side: int) -> bool:
        """Vrai si le curseur dépasse la fin de ce journal (affichage figé sur l'état final)."""
        return self.ply > len(self.logs[side].moves)

    # données
    def view(self, side: int) -> SideView:
        log = self.logs[side]
        src = self.sources[side]
        st = src.state
        cfg = log.config
        shown = src.index
        alive = [st.is_alive(p) for p in range(st.n)]
        shares = self._heur[side].shares(st) if any(alive) and not st.is_terminal() else []
        if not shares:
            shares = [0.0] * st.n
        team_shares: dict[str, float] = {}
        for p, s in enumerate(shares):
            team = cfg.players[p].team
            team_shares[team] = team_shares.get(team, 0.0) + s
        totals = [PlayerTotals() for _ in range(st.n)]
        for rec in log.moves[:shown]:
            if "nodes" in rec:
                t = totals[int(rec["player"])]
                t.decisions += 1
                t.nodes += int(rec.get("nodes") or 0)
                t.time += float(rec.get("time") or 0.0)
        winner, ranks, _length = self._results[side]
        events = self.timelines[side]
        so_far = [e for e in events if e.ply <= shown]
        return SideView(
            label=self.labels[side],
            name=cfg.name,
            seed=cfg.seed,
            first_player=cfg.first_player,
            agents=list(log.agents or [p.agent for p in cfg.players]),
            seats=[p.team for p in cfg.players],
            shown_ply=shown,
            length=len(log.moves),
            frozen=self.frozen(side),
            alive=alive,
            shares=list(shares),
            team_shares=team_shares,
            totals=totals,
            winner=winner,
            ranks=ranks,
            events_so_far=len(so_far),
            events_total=len(events),
            recent=so_far[-3:],
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """Lignes du panneau de comparaison : (libellé, valeur A, valeur B)."""
        a, b = self.view(0), self.view(1)

        def pct(d: dict[str, float]) -> str:
            return ", ".join(f"{k} {100 * v:.0f} %" for k, v in d.items())

        def tot(v: SideView) -> str:
            return " ; ".join(
                f"{p}: {t.decisions} déc., {t.time:.2f} s, {t.nodes} nœuds"
                for p, t in enumerate(v.totals)
            )

        def shown(v: SideView) -> str:
            return f"{v.shown_ply}/{v.length}" + (" (figé : partie terminée)" if v.frozen else "")

        rows = [
            ("configuration", a.name, b.name),
            (
                "graine · premier joueur",
                f"{a.seed} · {a.first_player}",
                f"{b.seed} · {b.first_player}",
            ),
            ("sièges (équipes)", ", ".join(a.seats), ", ".join(b.seats)),
            ("agents", ", ".join(a.agents), ", ".join(b.agents)),
            ("gagnant", a.winner or "égalité", b.winner or "égalité"),
            ("rangs", " ".join(map(str, a.ranks)), " ".join(map(str, b.ranks))),
            ("durée (demi-coups)", str(a.length), str(b.length)),
            ("demi-coup affiché", shown(a), shown(b)),
            ("territoire (Voronoi)", pct(a.team_shares), pct(b.team_shares)),
            ("décisions, temps, nœuds", tot(a), tot(b)),
            (
                "événements",
                f"{a.events_so_far}/{a.events_total}",
                f"{b.events_so_far}/{b.events_total}",
            ),
        ]
        div = "aucune" if self.divergence is None else f"demi-coup {self.divergence}"
        rows.append(("première divergence", div, div))
        return rows

    def text(self) -> str:
        """Panneau de comparaison en texte (sortie de ``tron compare --no-gui``)."""
        lines = [f"Comparaison au demi-coup {self.ply}/{self.max_ply}"]
        if not self.pairing.paired:
            lines.append(f"ATTENTION : {self.pairing.warning}")
        width = max(len(r[0]) for r in self.rows())
        for label, va, vb in self.rows():
            if label == "première divergence":
                lines.append(f"{label:<{width}}  {va}")
            else:
                lines.append(f"{label:<{width}}  {self.labels[0]} : {va}")
                lines.append(f"{'':<{width}}  {self.labels[1]} : {vb}")
        return "\n".join(lines)

    def owners(self, side: int) -> np.ndarray | None:
        """Carte des propriétaires de Voronoi de l'état affiché (``None`` en fin de partie)."""
        st = self.sources[side].state
        return None if st.is_terminal() else self._heur[side].owner_map(st)

    # images
    def palettes(self) -> tuple[Palette, Palette]:
        return (
            make_palette(self.logs[0].config.team_index),
            make_palette(self.logs[1].config.team_index),
        )

    def frame(
        self,
        cell: int = 12,
        gap: int = 12,
        voronoi: bool = False,
        theme: str = "classic",
        quality: str = "medium",
    ) -> np.ndarray:
        """Image côte à côte des deux plateaux au curseur (hauteurs complétées si besoin).

        ``theme="neon"`` : rendu néon des deux arènes (sans effets animés).
        """
        images = []
        neon = theme == "neon"
        for side, pal in enumerate(self.palettes()):
            st = self.sources[side].state
            owners = self.owners(side) if voronoi else None
            if neon:
                images.append(self._neon(side, cell, quality).render(st, owners=owners))
            else:
                images.append(render_frame(st, pal, cell=cell, owners=owners))
        h = max(img.shape[0] for img in images)
        if neon:
            from tron.viz.theme import BACKGROUND  # noqa: PLC0415

            surface = np.array(hex_to_rgb(BACKGROUND), dtype=np.uint8)
        else:
            surface = np.array(self.palettes()[0].surface, dtype=np.uint8)
        padded = []
        for img in images:
            pad = np.empty((h, img.shape[1], 3), dtype=np.uint8)
            pad[:] = surface
            pad[: img.shape[0]] = img
            padded.append(pad)
        spacer = np.empty((h, gap, 3), dtype=np.uint8)
        spacer[:] = surface if neon else np.array(self.palettes()[0].grid, dtype=np.uint8)
        return np.concatenate([padded[0], spacer, padded[1]], axis=1)

    def _neon(self, side: int, cell: int, quality: str) -> Any:
        """Rendu néon d'un côté, conservé d'une image à l'autre (calques incrémentaux)."""
        from tron.viz.neon import NeonRenderer  # noqa: PLC0415 - rendu néon à la demande

        key = (side, cell, quality)
        if key not in self._neon_cache:
            self._neon_cache[key] = NeonRenderer(self.logs[side].config, cell, quality=quality)
        return self._neon_cache[key]

    def frames(
        self,
        cell: int = 12,
        every: int = 1,
        hold_end: int = 10,
        theme: str = "classic",
        quality: str = "medium",
    ) -> list[np.ndarray]:
        """Images successives de toute la comparaison (le curseur revient à sa place)."""
        here = self.ply
        out = []
        for ply in range(0, self.max_ply + 1, max(1, every)):
            self.seek(ply)
            out.append(self.frame(cell, theme=theme, quality=quality))
        if self.ply != self.max_ply:
            self.seek(self.max_ply)
            out.append(self.frame(cell, theme=theme, quality=quality))
        out.extend([out[-1]] * hold_end)
        self.seek(here)
        return out


def describe(session: ComparisonSession) -> dict[str, Any]:
    """Données de la comparaison au curseur, sérialisables en JSON."""
    return {
        "ply": session.ply,
        "max_ply": session.max_ply,
        "paired": session.pairing.paired,
        "reasons": list(session.pairing.reasons),
        "divergence": session.divergence,
        "rows": [list(r) for r in session.rows()],
    }


__all__ = [
    "ComparisonSession",
    "Pairing",
    "SideView",
    "TimelineEvent",
    "describe",
    "first_divergence",
    "pairing",
    "timeline",
]
