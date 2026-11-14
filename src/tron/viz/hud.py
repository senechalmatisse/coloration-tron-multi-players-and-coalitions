"""Données du HUD holographique, indépendantes de toute bibliothèque graphique.

* :func:`player_cards` : une carte par joueur (couleur, nom, équipe, état
  ACTIF / ÉLIMINÉ / rang, part de territoire, alliés, dernière décision) ;
* :func:`hud_stats` : lignes de statistiques (joueurs actifs, cases occupées,
  éliminations, alliances, tour, temps écoulé) ;
* :class:`EventLog` : journal horodaté des événements marquants
  (``[042] JOUEUR 3 ÉLIMINÉ · trace de 1``), élagué lors d'un retour arrière.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tron.core.state import GameState
from tron.viz.controller import PanelInfo
from tron.viz.palette import RGB
from tron.viz.scene import ArenaStats, VisualEvent, arena_stats
from tron.viz.theme import NeonTheme

CAUSES = {
    "bordure": "bordure",
    "propre trace": "sa propre trace",
    "obstacle": "obstacle",
    "bloqué": "bloqué",
}


@dataclass(frozen=True, slots=True)
class PlayerCard:
    """Carte d'un joueur dans le panneau."""

    index: int
    title: str
    """``0 · Nom [équipe]``."""
    badge: str
    """``ACTIF``, ``ÉLIMINÉ`` ou ``RANG n`` (fin de partie)."""
    badge_color: RGB
    color: RGB
    detail: str
    """Algorithme, territoire et dernière décision."""
    share: float
    alive: bool
    current: bool
    """Joueur au trait."""


def player_cards(
    info: PanelInfo,
    theme: NeonTheme,
    alliances: Sequence[tuple[int, int]] = (),
    winners: Sequence[int] | None = None,
) -> list[PlayerCard]:
    """Cartes des joueurs ; ``winners`` (partie finie) distingue vainqueurs et vaincus."""
    cards = []
    for line in info.players:
        p = line.index
        if winners is not None and p in winners:
            badge, badge_color = "VAINQUEUR", theme.success
        elif winners is not None and line.alive:
            badge, badge_color = "VAINCU", theme.text_dim
        elif line.rank is not None:
            badge = f"RANG {line.rank}"
            badge_color = theme.success if line.rank == 1 else theme.text_dim
        elif line.alive:
            badge, badge_color = "ACTIF", theme.success
        else:
            badge, badge_color = "ÉLIMINÉ", theme.danger
        allies = sorted({b if a == p else a for a, b in alliances if p in (a, b)})
        parts = [f"{100 * line.share:.0f} %"]
        if allies:
            parts.append("allié de " + ", ".join(str(a) for a in allies))
        if line.depth is not None:
            parts.append(f"prof. {line.depth}")
        if line.nodes is not None:
            parts.append(f"{_compact(line.nodes)} nœuds")
        if line.value is not None:
            parts.append(f"v {line.value:+.2f}")
        parts.append(line.agent)
        detail = " · ".join(parts)
        cards.append(
            PlayerCard(
                index=p,
                title=f"{p} · {line.name} [{line.team}]",
                badge=badge,
                badge_color=badge_color,
                color=theme.player[p] if line.alive else theme.dead[p],
                detail=detail,
                share=line.share,
                alive=line.alive,
                current=line.alive and info.current == p and line.rank is None,
            )
        )
    return cards


def fit(text: str, limit: int) -> str:
    """Texte tronqué à ``limit`` caractères (points de suspension compris)."""
    if limit <= 1 or len(text) <= limit:
        return text if limit > 1 else text[:1]
    return text[: limit - 1].rstrip() + "…"


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    if n >= 10_000:
        return f"{n / 1000:.0f} k"
    return str(n)


def format_clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def hud_stats(
    state: GameState,
    alliances: int = 0,
    elapsed: float | None = None,
    speed: float | None = None,
) -> list[tuple[str, str]]:
    """Lignes (libellé, valeur) du bloc de statistiques."""
    st: ArenaStats = arena_stats(state, alliances)
    rows = [
        ("joueurs actifs", f"{st.active} / {st.total}"),
        ("cases occupées", f"{100 * st.occupied:.0f} %"),
        ("éliminations", str(st.eliminations)),
        ("alliances", str(st.alliances)),
        (
            "demi-coup",
            str(st.ply) + (f" · tour {st.round}" if state.config.rules.needs_engine else ""),
        ),
    ]
    if elapsed is not None:
        rows.append(("temps", format_clock(elapsed)))
    if speed is not None:
        rows.append(("vitesse", f"×{speed:g}"))
    return rows


@dataclass(frozen=True, slots=True)
class LogLine:
    ply: int
    text: str
    color: RGB


class EventLog:
    """Journal des derniers événements marquants (au plus ``size`` lignes)."""

    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.lines: deque[LogLine] = deque(maxlen=64)

    def clear(self) -> None:
        self.lines.clear()

    def prune(self, ply: int) -> None:
        """Retire les lignes postérieures au demi-coup ``ply`` (retour arrière)."""
        while self.lines and self.lines[-1].ply >= ply:
            self.lines.pop()

    def add(
        self,
        events: Iterable[VisualEvent],
        state: GameState,
        theme: NeonTheme,
    ) -> None:
        for ev in events:
            line = describe_event(ev, state, theme)
            if line is not None:
                self.lines.append(line)

    def recent(self, n: int | None = None) -> list[LogLine]:
        k = self.size if n is None else n
        return list(self.lines)[-k:] if k > 0 else []


def describe_event(ev: VisualEvent, state: GameState, theme: NeonTheme) -> LogLine | None:
    """Ligne de journal d'un événement (``None`` pour un simple déplacement)."""
    cfg = state.config
    names = [p.name for p in cfg.players]

    def who(p: int) -> str:
        return f"{p} ({names[p]})" if 0 <= p < len(names) else str(p)

    if ev.kind == "elimination":
        cause = CAUSES.get(ev.cause, "")
        if ev.cause == "trace" and ev.other >= 0:
            cause = f"trace de {ev.other}"
        text = f"JOUEUR {who(ev.player)} ÉLIMINÉ" + (f" · {cause}" if cause else "")
        return LogLine(ev.ply, text, theme.danger)
    if ev.kind == "alliance":
        return LogLine(ev.ply, f"ALLIANCE {ev.player} + {ev.other}", theme.accent)
    if ev.kind == "rupture":
        return LogLine(ev.ply, f"RUPTURE {ev.player} × {ev.other}", theme.warning)
    if ev.kind == "zone":
        if ev.other >= 0:
            team = cfg.victory_names[ev.other] if ev.other < len(cfg.victory_names) else ev.other
            return LogLine(ev.ply, f"ZONE {ev.index + 1} → {team}", theme.text_dim)
        return LogLine(ev.ply, f"ZONE {ev.index + 1} NEUTRE", theme.text_muted)
    if ev.kind == "obstacles":
        n = len(ev.cells)
        return LogLine(ev.ply, f"{n} OBSTACLE{'S' if n > 1 else ''}", theme.warning)
    if ev.kind == "victory":
        if ev.other < 0:
            return LogLine(ev.ply, "ÉGALITÉ", theme.text)
        team = cfg.victory_names[ev.other] if ev.other < len(cfg.victory_names) else ev.other
        return LogLine(ev.ply, f"VICTOIRE · {team}", theme.success)
    return None


def stamp(line: LogLine) -> str:
    """Texte horodaté d'une ligne : ``[042] …`` (numéro de demi-coup)."""
    return f"[{line.ply:03d}] {line.text}"


__all__ = [
    "EventLog",
    "LogLine",
    "PlayerCard",
    "describe_event",
    "fit",
    "format_clock",
    "hud_stats",
    "player_cards",
    "stamp",
]
