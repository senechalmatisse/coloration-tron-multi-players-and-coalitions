"""Journal de partie rejouable au format JSON Lines (F5.2).

Première ligne : ``{"type": "header", ...}`` avec la configuration complète et
les positions initiales ; puis une ligne ``{"type": "move", ...}`` par demi-coup
et enfin ``{"type": "result", ...}``. Le rejeu recalcule chaque état en appliquant
les coups depuis la position initiale, et vérifie la cohérence avec le journal.

Versions du format (champ ``format`` de l'en-tête) :

* ``tron-log/1`` : format historique, toujours lisible ;
* ``tron-log/2`` : l'en-tête porte la configuration diplomatique complète
  (``diplomacy``) ; une ligne ``move`` peut porter ``diplomacy`` (événements
  d'alliance du demi-coup) ; la ligne ``result`` porte les métriques de partie.
  Une ligne ``move`` peut aussi porter ``details`` : l'explication typée de la
  décision (:class:`~tron.agents.base.DecisionDetails`), désactivable
  (``Recorder(path, details=False)``). Les champs ajoutés sont facultatifs : un
  journal v2 sans diplomatie ni détails se lit comme un journal v1.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import tron
from tron.agents.base import DecisionDetails
from tron.agents.context import AllianceEvent
from tron.core.config import GameConfig
from tron.core.directions import direction_name, parse_direction
from tron.core.state import GameState
from tron.sim.match import Match, MoveEvent
from tron.sim.metrics import metrics_from_moves
from tron.sim.result import MatchMetrics, MatchResult, nan_to_none

LOG_FORMAT = "tron-log/2"
READABLE_FORMATS = ("tron-log/1", "tron-log/2")


class LogError(ValueError):
    """Journal illisible ou incohérent."""


class Recorder:
    """Observateur qui écrit le journal JSONL d'une partie."""

    def __init__(self, path: str | Path, details: bool = True) -> None:
        self.path = Path(path)
        self.details = details
        self._fh: IO[str] | None = None

    def _write(self, record: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def on_start(self, match: Match) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "type": "header",
                "format": LOG_FORMAT,
                "version": tron.__version__,
                "config": match.config.to_dict(),
                "positions": [list(p) for p in match.state.positions],
                "agents": [a.describe() for a in match.agents],
                "diplomacy": match.config.diplomacy.to_dict(),
            }
        )

    def on_event(self, match: Match, event: MoveEvent) -> None:
        record: dict[str, Any] = {
            "type": "move",
            "ply": event.ply,
            "player": event.player,
            "dir": direction_name(event.direction),
            "to": list(event.target) if event.target else None,
            "elim": event.eliminated,
        }
        if event.stats is not None:
            s = event.stats
            record.update(
                nodes=s.nodes,
                time=round(s.time, 6),
                value=nan_to_none(s.value),
                depth=s.depth,
                ties=s.ties,
                engine=s.engine,
                leaves=s.leaves,
                prunes=s.prunes,
                pv="".join(direction_name(d) for d in s.pv),
            )
        if event.diplomacy:
            record["diplomacy"] = [ev.to_dict() for ev in event.diplomacy]
        if self.details and event.details is not None:
            record["details"] = event.details.to_dict()
        self._write(record)

    def on_end(self, match: Match, result: MatchResult) -> None:
        self._write({"type": "result", **result.to_dict()})
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class GameLog:
    """Contenu d'un journal : configuration, positions, coups et résultat."""

    config: GameConfig
    positions: list[tuple[int, int]]
    moves: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    agents: list[str] = field(default_factory=list)
    format: str = LOG_FORMAT

    @classmethod
    def load(cls, path: str | Path) -> GameLog:
        header: dict[str, Any] | None = None
        moves: list[dict[str, Any]] = []
        result = None
        with Path(path).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LogError(f"{path}:{lineno} : JSON invalide") from exc
                kind = rec.get("type")
                if kind == "header":
                    header = rec
                elif kind == "move":
                    moves.append(rec)
                elif kind == "result":
                    result = rec
        if header is None:
            raise LogError(f"{path} : en-tête absent")
        fmt = str(header.get("format", "tron-log/1"))
        if fmt not in READABLE_FORMATS:
            raise LogError(f"{path} : format de journal non pris en charge : {fmt}")
        return cls(
            config=GameConfig.from_dict(header["config"]),
            positions=[(int(x), int(y)) for x, y in header["positions"]],
            moves=moves,
            result=result,
            agents=list(header.get("agents", [])),
            format=fmt,
        )

    @property
    def directions(self) -> list[int]:
        return [parse_direction(m["dir"]) for m in self.moves]

    def initial_state(self) -> GameState:
        return GameState(self.config, positions=self.positions)

    def states(self) -> Iterator[GameState]:
        """États successifs (le même objet, modifié en place), de l'initial au final."""
        st = self.initial_state()
        yield st
        for rec, d in zip(self.moves, self.directions, strict=True):
            apply_logged(st, rec, d)
            yield st

    def final_state(self) -> GameState:
        st = self.initial_state()
        for rec, d in zip(self.moves, self.directions, strict=True):
            apply_logged(st, rec, d)
        return st

    def details(self, index: int) -> DecisionDetails | None:
        """Explication enregistrée du demi-coup ``index`` (``None`` si absente)."""
        data = self.moves[index].get("details")
        if not data:
            return None
        try:
            return DecisionDetails.from_dict(data)
        except (TypeError, ValueError, AttributeError) as exc:
            raise LogError(f"demi-coup {index} : détails de décision invalides") from exc

    def diplomacy_events(self) -> list[AllianceEvent]:
        """Événements diplomatiques du journal, dans l'ordre (vide pour tron-log/1)."""
        out: list[AllianceEvent] = []
        for rec in self.moves:
            for ev in rec.get("diplomacy", ()):
                try:
                    out.append(AllianceEvent.from_dict(ev))
                except (KeyError, TypeError, ValueError) as exc:
                    raise LogError(f"demi-coup {rec.get('ply')} : événement invalide") from exc
        return out

    def metrics(self, territory_every: int | None = None) -> MatchMetrics:
        """Recalcule les métriques de coopération en rejouant le journal (tout format)."""
        moves = [
            (int(rec["player"]), d) for rec, d in zip(self.moves, self.directions, strict=True)
        ]
        return metrics_from_moves(
            self.config, self.positions, moves, territory_every, self.diplomacy_events()
        )


def apply_logged(state: GameState, record: dict[str, Any], direction: int) -> None:
    """Applique un coup du journal en vérifiant joueur et issue."""
    if record["player"] != state.current:
        raise LogError(
            f"demi-coup {record['ply']} : joueur {record['player']} au lieu de {state.current}"
        )
    moved = state.apply(direction)
    if moved == bool(record["elim"]):
        raise LogError(f"demi-coup {record['ply']} : issue incohérente avec le journal")
