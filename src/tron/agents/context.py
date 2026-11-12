"""Contexte immuable d'une décision et types diplomatiques (P03).

Ces types sont définis dans la couche ``agents`` pour que les agents les
reçoivent sans dépendre de la simulation ; le gestionnaire mutable des alliances
(:class:`tron.sim.diplomacy.DiplomacyState`) les produit et les ré-exporte.

* :class:`DiplomacyIntent` : intention d'un agent à la fin de sa décision
  (proposer une alliance ou rompre une alliance avec ``target``) ;
* :class:`AllianceEvent` : événement diplomatique horodaté (demi-coup) ;
* :class:`DiplomacySnapshot` : vue figée des alliances, propositions, ruptures et
  confiance au moment d'une décision ;
* :class:`DecisionContext` : ce que l'agent reçoit en plus de l'état.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

INTENT_KINDS = ("propose", "break")
EVENT_KINDS = ("proposed", "formed", "broken", "expired", "dissolved", "rejected")
"""``dissolved`` : alliance close par l'élimination d'un membre ; ``rejected`` :
intention refusée par la validation (cible invalide, doublon, plafond…)."""


@dataclass(frozen=True, slots=True)
class DiplomacyIntent:
    """Intention diplomatique formulée à la fin d'une décision."""

    kind: Literal["propose", "break"]
    target: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in INTENT_KINDS:
            raise ValueError(f"intention diplomatique inconnue : {self.kind!r}")
        if self.target is None or isinstance(self.target, bool) or self.target < 0:
            raise ValueError(f"cible diplomatique invalide : {self.target!r}")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "target": self.target}


@dataclass(frozen=True, slots=True)
class AllianceEvent:
    """Événement diplomatique : ``player`` agit envers ``target`` au demi-coup ``ply``.

    Pour ``formed``, ``player`` est le joueur dont la proposition réciproque a
    scellé l'alliance ; pour ``dissolved``, c'est le joueur éliminé ; pour
    ``rejected`` sans cible valide, ``target`` vaut −1.
    """

    ply: int
    player: int
    target: int
    kind: Literal["proposed", "formed", "broken", "expired", "dissolved", "rejected"]

    def to_dict(self) -> dict[str, object]:
        return {"ply": self.ply, "player": self.player, "target": self.target, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AllianceEvent:
        kind = str(data["kind"])
        if kind not in EVENT_KINDS:
            raise ValueError(f"événement diplomatique inconnu : {kind!r}")
        return cls(
            ply=int(data["ply"]),  # type: ignore[call-overload]
            player=int(data["player"]),  # type: ignore[call-overload]
            target=int(data["target"]),  # type: ignore[call-overload]
            kind=kind,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DiplomacySnapshot:
    """État diplomatique figé, transmis aux agents.

    * ``active_pairs`` : alliances actives (a < b) ;
    * ``pending`` : propositions en attente (proposant, cible, demi-coup de la proposition) ;
    * ``trust[p][q]`` = 1 / (1 + trahisons de q envers p), dans ]0, 1] ;
    * ``betrayals[q][p]`` : trahisons de q envers p (règle : voir
      :mod:`tron.sim.diplomacy`) ;
    * ``cooldowns`` : paires (a < b) qui ne peuvent pas se proposer d'alliance
      avant le demi-coup indiqué.
    """

    ply: int
    active_pairs: tuple[tuple[int, int], ...]
    pending: tuple[tuple[int, int, int], ...]
    trust: tuple[tuple[float, ...], ...]
    betrayals: tuple[tuple[int, ...], ...]
    cooldowns: tuple[tuple[int, int, int], ...] = ()

    def are_allied(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.active_pairs

    def allies(self, player: int) -> tuple[int, ...]:
        out = [b if a == player else a for a, b in self.active_pairs if player in (a, b)]
        return tuple(sorted(out))

    def has_pending(self, proposer: int, target: int) -> bool:
        return any(p == proposer and t == target for p, t, _ in self.pending)

    def proposals_to(self, player: int) -> tuple[int, ...]:
        """Joueurs dont une proposition adressée à ``player`` est en attente (ordre croissant)."""
        return tuple(sorted(p for p, t, _ in self.pending if t == player))

    def in_cooldown(self, a: int, b: int) -> bool:
        lo, hi = min(a, b), max(a, b)
        return any(x == lo and y == hi and self.ply < until for x, y, until in self.cooldowns)

    def to_dict(self) -> dict[str, object]:
        return {
            "ply": self.ply,
            "active_pairs": [list(p) for p in self.active_pairs],
            "pending": [list(p) for p in self.pending],
            "trust": [list(r) for r in self.trust],
            "betrayals": [list(r) for r in self.betrayals],
            "cooldowns": [list(c) for c in self.cooldowns],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiplomacySnapshot:
        return cls(
            ply=int(data["ply"]),
            active_pairs=tuple((int(a), int(b)) for a, b in data.get("active_pairs", [])),
            pending=tuple((int(a), int(b), int(c)) for a, b, c in data.get("pending", [])),
            trust=tuple(tuple(float(x) for x in r) for r in data.get("trust", [])),
            betrayals=tuple(tuple(int(x) for x in r) for r in data.get("betrayals", [])),
            cooldowns=tuple((int(a), int(b), int(c)) for a, b, c in data.get("cooldowns", [])),
        )


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Contexte d'une décision : demi-coup courant et, si elle est active, la diplomatie."""

    ply: int = 0
    diplomacy: DiplomacySnapshot | None = None


__all__ = [
    "EVENT_KINDS",
    "INTENT_KINDS",
    "AllianceEvent",
    "DecisionContext",
    "DiplomacyIntent",
    "DiplomacySnapshot",
]
