"""Diplomatie tactique (P03) : propositions, alliances, ruptures, expirations.

Règles de la V1 (alliances souples, sans effet sur les règles de victoire) :

* **proposer** : un joueur formule au plus une intention par décision, à la fin
  de celle-ci. Une proposition reste en attente ``proposal_ttl_plies``
  demi-coups : faite au demi-coup r, elle expire au demi-coup r + TTL ;
* **former** : une alliance naît quand deux joueurs se sont proposé
  mutuellement avant expiration. Elle est symétrique ; chaque joueur a au plus
  ``max_allies_per_player`` alliés ;
* **rompre** : unilatéral ; la paire ne peut plus se proposer d'alliance pendant
  ``break_cooldown_plies`` demi-coups ;
* **élimination** : un joueur éliminé perd ses alliances (``dissolved``) et ses
  propositions (``expired``) ; il ne peut plus agir ;
* une intention invalide (cible éliminée ou inconnue, soi-même, doublon,
  plafond, délai après rupture) est refusée (``rejected``) sans interrompre la
  partie ;
* les alliances ne changent **ni les équipes permanentes ni la victoire** : elles
  ne modifient que les préférences sociales des agents qui les exploitent. Par
  hypothèse (``free_action``), une action diplomatique ne coûte pas de demi-coup.

**Trahison** (règle explicite) : la rupture par q de son alliance avec p est une
trahison si p n'avait jamais rompu d'alliance avec q auparavant (q fait défection
le premier dans cette relation). La confiance de p envers q vaut
1 / (1 + trahisons de q envers p).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tron.agents.context import (
    AllianceEvent,
    DecisionContext,
    DiplomacyIntent,
    DiplomacySnapshot,
)
from tron.core.config import DiplomacySpec


class DiplomacyError(ValueError):
    """Action diplomatique invalide."""


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


class DiplomacyState:
    """Gestionnaire mutable des alliances d'une partie (voir les règles du module)."""

    def __init__(self, spec: DiplomacySpec, n_players: int) -> None:
        self.spec = spec
        self.n = n_players
        self._pending: dict[tuple[int, int], int] = {}
        self._active: dict[tuple[int, int], int] = {}
        self._cooldown: dict[tuple[int, int], int] = {}
        self._breaks = [[0] * n_players for _ in range(n_players)]
        self._betrayals = [[0] * n_players for _ in range(n_players)]
        self._alive = [True] * n_players
        self.events: list[AllianceEvent] = []

    # ------------------------------------------------------------ requêtes
    def are_allied(self, a: int, b: int) -> bool:
        return _pair(a, b) in self._active

    def allies(self, player: int) -> tuple[int, ...]:
        return tuple(sorted(b if a == player else a for a, b in self._active if player in (a, b)))

    def events_since(self, index: int) -> list[AllianceEvent]:
        """Événements à partir de l'indice ``index`` de l'historique."""
        return self.events[index:]

    def snapshot(self, ply: int) -> DiplomacySnapshot:
        n = self.n
        return DiplomacySnapshot(
            ply=ply,
            active_pairs=tuple(sorted(self._active)),
            pending=tuple(sorted((p, t, r) for (p, t), r in self._pending.items())),
            trust=tuple(
                tuple(1.0 / (1 + self._betrayals[q][p]) for q in range(n)) for p in range(n)
            ),
            betrayals=tuple(tuple(row) for row in self._betrayals),
            cooldowns=tuple(
                sorted((a, b, until) for (a, b), until in self._cooldown.items() if until > ply)
            ),
        )

    # ------------------------------------------------------------- actions
    def _check(self, player: int, target: int) -> None:
        n = self.n
        for who in (player, target):
            if isinstance(who, bool) or not isinstance(who, int) or not 0 <= who < n:
                raise DiplomacyError(f"joueur invalide : {who!r}")
        if player == target:
            raise DiplomacyError("un joueur ne peut pas s'allier avec lui-même")
        if not self._alive[player]:
            raise DiplomacyError(f"le joueur {player} est éliminé")
        if not self._alive[target]:
            raise DiplomacyError(f"le joueur {target} est éliminé")

    def propose(self, player: int, target: int, ply: int) -> list[AllianceEvent]:
        """``player`` propose une alliance à ``target`` ; réciproque, elle forme l'alliance."""
        self._check(player, target)
        pair = _pair(player, target)
        cap = self.spec.max_allies_per_player
        if pair in self._active:
            raise DiplomacyError(f"{player} et {target} sont déjà alliés")
        if (player, target) in self._pending:
            raise DiplomacyError(f"proposition de {player} à {target} déjà en attente")
        if self._cooldown.get(pair, -1) > ply:
            raise DiplomacyError(f"délai après rupture entre {player} et {target}")
        if len(self.allies(player)) >= cap:
            raise DiplomacyError(f"le joueur {player} a déjà {cap} allié(s)")
        new = [AllianceEvent(ply, player, target, "proposed")]
        if (target, player) in self._pending:
            if len(self.allies(target)) >= cap:
                raise DiplomacyError(f"le joueur {target} a déjà {cap} allié(s)")
            del self._pending[(target, player)]
            self._active[pair] = ply
            new.append(AllianceEvent(ply, player, target, "formed"))
        else:
            self._pending[(player, target)] = ply
        self.events.extend(new)
        return new

    def break_alliance(self, player: int, target: int, ply: int) -> list[AllianceEvent]:
        """``player`` rompt son alliance avec ``target`` (trahison selon la règle du module)."""
        self._check(player, target)
        pair = _pair(player, target)
        if pair not in self._active:
            raise DiplomacyError(f"{player} et {target} ne sont pas alliés")
        del self._active[pair]
        self._cooldown[pair] = ply + self.spec.break_cooldown_plies
        if self._breaks[target][player] == 0:
            self._betrayals[player][target] += 1
        self._breaks[player][target] += 1
        new = [AllianceEvent(ply, player, target, "broken")]
        self.events.extend(new)
        return new

    def apply_intent(self, player: int, intent: DiplomacyIntent, ply: int) -> list[AllianceEvent]:
        """Applique une intention d'agent ; une intention invalide est tracée comme refusée."""
        target = intent.target if intent.target is not None else -1
        try:
            if intent.kind == "propose":
                return self.propose(player, target, ply)
            return self.break_alliance(player, target, ply)
        except DiplomacyError:
            ev = AllianceEvent(ply, player, target, "rejected")
            self.events.append(ev)
            return [ev]

    def advance(self, ply: int, alive: Sequence[bool]) -> list[AllianceEvent]:
        """Nettoyage au demi-coup ``ply`` : éliminations puis propositions expirées."""
        new: list[AllianceEvent] = []
        for p in range(self.n):
            if self._alive[p] and not alive[p]:
                self._alive[p] = False
                for pair in sorted(self._active):
                    if p in pair:
                        del self._active[pair]
                        other = pair[1] if pair[0] == p else pair[0]
                        new.append(AllianceEvent(ply, p, other, "dissolved"))
                for key in sorted(self._pending):
                    if p in key:
                        del self._pending[key]
                        new.append(AllianceEvent(ply, key[0], key[1], "expired"))
        ttl = self.spec.proposal_ttl_plies
        for key in sorted(self._pending):
            if ply - self._pending[key] >= ttl:
                del self._pending[key]
                new.append(AllianceEvent(ply, key[0], key[1], "expired"))
        self.events.extend(new)
        return new

    def context(self, ply: int) -> DecisionContext:
        return DecisionContext(ply=ply, diplomacy=self.snapshot(ply))


@dataclass(frozen=True, slots=True)
class DiplomacySummary:
    """Bilan diplomatique d'une partie (métriques P02 étendues par P03)."""

    proposals: int = 0
    alliances_formed: int = 0
    alliance_breaks: int = 0
    betrayals: int = 0
    rejected_intents: int = 0
    alliance_duration_mean: float | None = None


def summarize_diplomacy(events: Sequence[AllianceEvent], final_ply: int) -> DiplomacySummary:
    """Comptes et durée moyenne des alliances (une alliance encore active dure jusqu'à la fin).

    Les trahisons sont recomptées avec la règle du module, à partir des seuls
    événements : le bilan d'un journal rejoué est identique au bilan en direct.
    """
    proposals = formed = breaks = betrayals = rejected = 0
    start: dict[tuple[int, int], int] = {}
    durations: list[int] = []
    broken_by: dict[tuple[int, int], int] = {}
    for ev in events:
        pair = _pair(ev.player, ev.target)
        if ev.kind == "proposed":
            proposals += 1
        elif ev.kind == "formed":
            formed += 1
            start[pair] = ev.ply
        elif ev.kind == "broken":
            breaks += 1
            if broken_by.get((ev.target, ev.player), 0) == 0:
                betrayals += 1
            broken_by[(ev.player, ev.target)] = broken_by.get((ev.player, ev.target), 0) + 1
            if pair in start:
                durations.append(ev.ply - start.pop(pair))
        elif ev.kind == "dissolved":
            if pair in start:
                durations.append(ev.ply - start.pop(pair))
        elif ev.kind == "rejected":
            rejected += 1
    durations.extend(final_ply - s for s in start.values())
    return DiplomacySummary(
        proposals=proposals,
        alliances_formed=formed,
        alliance_breaks=breaks,
        betrayals=betrayals,
        rejected_intents=rejected,
        alliance_duration_mean=sum(durations) / len(durations) if durations else None,
    )


__all__ = [
    "AllianceEvent",
    "DecisionContext",
    "DiplomacyError",
    "DiplomacyIntent",
    "DiplomacySnapshot",
    "DiplomacyState",
    "DiplomacySummary",
    "summarize_diplomacy",
]
