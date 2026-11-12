"""Métriques de coopération, de cohésion et de stabilité d'une partie (P02).

Les métriques sont calculées à partir des événements observables (coups,
éliminations), de l'état du plateau et des équipes statiques, sans modifier les
règles ni les décisions : un collecteur n'utilise que ses propres instances
d'heuristique. Elles sont déterministes à configuration et graine identiques.

Définitions (t est une équipe statique, ses joueurs sont ses « membres ») :

* ``survival_plies[p]`` : nombre de demi-coups après lesquels p est encore
  vivant (indice du demi-coup de son élimination, ou nombre total de demi-coups
  de la partie s'il survit). Unité : demi-coups.
* ``team_survival_plies[t]`` : maximum des ``survival_plies`` des membres, soit
  le nombre de demi-coups pendant lesquels au moins un membre reste vivant.
* ``team_rank[t]`` : meilleur rang final d'un membre (1 = équipe gagnante).
* ``intra_team_distance_mean[t]`` : moyenne, sur les états observés (état
  initial puis après chaque demi-coup) où au moins deux membres vivent, de la
  distance de Manhattan moyenne entre les têtes des membres vivants. Unité :
  cases. ``None`` si deux membres ne sont jamais vivants ensemble (équipe d'un
  joueur notamment).
* ``ally_collisions[t]`` : éliminations de membres dont la cause est, sans
  ambiguïté, un mur laissé par un coéquipier : collision directe avec ce mur, ou
  enfermement (aucun coup légal) dont toutes les cases bloquantes, hors bordure
  et murs du joueur lui-même, appartiennent à des coéquipiers. Une cause mixte ou
  adverse n'est pas comptée.
* ``blocking_moves[t]`` : déplacements d'un membre qui l'amènent sur une case
  voisine de la tête d'un coéquipier vivant (ce dernier perd une case libre
  voisine). Un déplacement compte au plus une fois.
* ``team_moves[t]`` : déplacements des membres (hors éliminations).
* ``cooperation_proxy[t]`` = ½·proximité + ½·(1 − blocking_moves / team_moves),
  avec proximité = 1 − distance moyenne / (W + H − 2), dans [0, 1]. ``None`` si
  la distance est ``None``. C'est un **indicateur de comportement observable**
  (rester proches sans se gêner), pas une mesure d'intention.
* ``territory_share_mean[t]`` : moyenne de la somme des parts de Voronoi des
  membres aux points d'observation (état initial, puis tous les ``k``
  demi-coups, états non terminaux seulement ; ``k`` = nombre de joueurs par
  défaut, 0 désactive). ``None`` sans observation.
* ``collective_vs_individual_gap[t]`` = 1 − moyenne(``survival_plies`` des
  membres) / ``team_survival_plies[t]`` : 0 si tous les membres tiennent aussi
  longtemps que l'équipe, proche de 1 si l'équipe ne survit que grâce à un
  membre. ``None`` pour une équipe d'un joueur ou une survie d'équipe nulle.

Diplomatie (P03, si elle est activée) : nombre de propositions, d'alliances
formées, de ruptures, de trahisons (règle explicite de :mod:`tron.sim.diplomacy`),
d'intentions refusées et durée moyenne des alliances (demi-coups ; une alliance
encore active dure jusqu'à la fin de la partie).

Limites : ces grandeurs décrivent des trajectoires, pas des intentions. Une
« trahison » n'est pas déductible d'un simple déplacement ; elle n'est mesurée
qu'avec la diplomatie explicite (propositions et ruptures d'alliance).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tron.agents.context import AllianceEvent
from tron.core.config import GameConfig
from tron.core.state import FREE, GameState
from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.diplomacy import DiplomacySummary, summarize_diplomacy
from tron.sim.result import METRICS_VERSION, MatchMetrics


class MetricsCollector:
    """Accumule les observations d'une partie et produit un :class:`MatchMetrics`.

    Utilisation : ``start(état initial)``, ``record(état après coup, joueur,
    origine, direction, cible, éliminé)`` à chaque demi-coup, puis
    ``finish(état final)``. Le collecteur ne modifie jamais l'état.
    """

    def __init__(self, config: GameConfig, territory_every: int | None = None) -> None:
        n = config.n_players
        every = n if territory_every is None else int(territory_every)
        if every < 0:
            raise ValueError("territory_every doit être >= 0 (0 désactive l'observation)")
        self.config = config
        self.every = every
        self.team = list(config.team_index)
        self.n_teams = len(config.team_names)
        self.members = [[p for p in range(n) if self.team[p] == t] for t in range(self.n_teams)]
        tie = str(config.heuristic.params.get("tie", "order"))
        self._heur = (
            VoronoiHeuristic(
                config.width, config.height, n, tie=tie if tie == "neutral" else "order"
            )
            if every > 0
            else None
        )
        t = self.n_teams
        self._dist_sum = [0.0] * t
        self._dist_obs = [0] * t
        self._terr_sum = [0.0] * t
        self._terr_obs = 0
        self._collisions = [0] * t
        self._blocks = [0] * t
        self._moves = [0] * t

    # ---------------------------------------------------------- observations
    def start(self, state: GameState) -> None:
        self._observe_distances(state)
        if self._heur is not None and not state.is_terminal():
            self._observe_territory(state)

    def record(
        self,
        state: GameState,
        player: int,
        origin: tuple[int, int],
        direction: int,
        target: tuple[int, int] | None,
        eliminated: bool,
        ply: int,
    ) -> None:
        """Observation après le demi-coup ``ply`` (``state`` est l'état résultant)."""
        t = self.team[player]
        if eliminated:
            if self._ally_caused(state, player, origin, direction):
                self._collisions[t] += 1
        elif target is not None:
            self._moves[t] += 1
            tx, ty = target
            for q in self.members[t]:
                if q != player and state.is_alive(q):
                    qx, qy = state.head(q)
                    if abs(qx - tx) + abs(qy - ty) == 1:
                        self._blocks[t] += 1
                        break
        self._observe_distances(state)
        if (
            self._heur is not None
            and (ply + 1) % self.every == 0
            and not state.is_terminal()
            and state.alive_players()
        ):
            self._observe_territory(state)

    def _ally_caused(
        self, state: GameState, player: int, origin: tuple[int, int], direction: int
    ) -> bool:
        """Vrai si l'élimination est due sans ambiguïté aux murs de coéquipiers."""
        team = self.team
        c = state.index(*origin)
        g = state._g
        if 0 <= direction <= 3:
            occupant = int(g[c + state._off[direction]])
            return occupant >= 0 and occupant != player and team[occupant] == team[player]
        blockers: set[int] = set()
        for off in state._off:
            v = int(g[c + off])
            if v == FREE:
                return False  # abandon volontaire : cause indéterminée
            if v >= 0 and v != player:
                blockers.add(v)
        return bool(blockers) and all(team[b] == team[player] for b in blockers)

    def _observe_distances(self, state: GameState) -> None:
        for t, members in enumerate(self.members):
            alive = [state.head(p) for p in members if state.is_alive(p)]
            if len(alive) < 2:
                continue
            total = 0
            pairs = 0
            for i, (x0, y0) in enumerate(alive):
                for x1, y1 in alive[i + 1 :]:
                    total += abs(x0 - x1) + abs(y0 - y1)
                    pairs += 1
            self._dist_sum[t] += total / pairs
            self._dist_obs[t] += 1

    def _observe_territory(self, state: GameState) -> None:
        assert self._heur is not None
        shares = self._heur.shares(state)
        for t, members in enumerate(self.members):
            self._terr_sum[t] += sum(shares[p] for p in members)
        self._terr_obs += 1

    # ------------------------------------------------------------- sauvegarde
    _FIELDS = (
        "_dist_sum",
        "_dist_obs",
        "_terr_sum",
        "_terr_obs",
        "_collisions",
        "_blocks",
        "_moves",
    )

    def to_dict(self) -> dict[str, Any]:
        """Accumulateurs, pour une sauvegarde exacte (P08)."""
        return {name: getattr(self, name) for name in self._FIELDS} | {"every": self.every}

    def restore(self, data: Mapping[str, Any]) -> None:
        if int(data["every"]) != self.every:
            raise ValueError("pas d'observation du territoire incohérent")
        for name in self._FIELDS:
            value = data[name]
            setattr(self, name, list(value) if isinstance(value, list) else value)

    # --------------------------------------------------------------- bilan
    def finish(self, state: GameState, diplomacy: DiplomacySummary | None = None) -> MatchMetrics:
        """Métriques de la partie à partir de l'état final (terminal ou non).

        ``diplomacy`` : bilan diplomatique, ``None`` si la diplomatie est inactive.
        """
        n = state.n
        total = state.ply
        survival = tuple(total if state.is_alive(p) else int(state.death_ply[p]) for p in range(n))
        out = state.outcome()
        ranks = out.ranks if out is not None else tuple(1 for _ in range(n))
        cfg = self.config
        span = cfg.width + cfg.height - 2
        team_surv: list[int] = []
        team_rank: list[int] = []
        dist: list[float | None] = []
        coop: list[float | None] = []
        terr: list[float | None] = []
        gap: list[float | None] = []
        for t, members in enumerate(self.members):
            ts = max(survival[p] for p in members)
            team_surv.append(ts)
            team_rank.append(min(ranks[p] for p in members))
            d = self._dist_sum[t] / self._dist_obs[t] if self._dist_obs[t] else None
            dist.append(d)
            if d is None:
                coop.append(None)
            else:
                block_rate = self._blocks[t] / self._moves[t] if self._moves[t] else 0.0
                coop.append(0.5 * (1.0 - d / span) + 0.5 * (1.0 - block_rate))
            terr.append(self._terr_sum[t] / self._terr_obs if self._terr_obs else None)
            if len(members) < 2 or ts == 0:
                gap.append(None)
            else:
                mean = sum(survival[p] for p in members) / len(members)
                gap.append(1.0 - mean / ts)
        return MatchMetrics(
            version=METRICS_VERSION,
            teams=cfg.team_names,
            team_sizes=cfg.team_sizes,
            survival_plies=survival,
            team_survival_plies=tuple(team_surv),
            team_rank=tuple(team_rank),
            intra_team_distance_mean=tuple(dist),
            ally_collisions=tuple(self._collisions),
            blocking_moves=tuple(self._blocks),
            team_moves=tuple(self._moves),
            cooperation_proxy=tuple(coop),
            territory_share_mean=tuple(terr),
            collective_vs_individual_gap=tuple(gap),
            territory_observations=self._terr_obs,
            **_diplomacy_fields(diplomacy),
        )


def _diplomacy_fields(summary: DiplomacySummary | None) -> dict[str, Any]:
    if summary is None:
        return {}
    return {
        "diplomacy_enabled": True,
        "proposals": summary.proposals,
        "alliances_formed": summary.alliances_formed,
        "alliance_breaks": summary.alliance_breaks,
        "betrayals": summary.betrayals,
        "rejected_intents": summary.rejected_intents,
        "alliance_duration_mean": summary.alliance_duration_mean,
    }


def metrics_from_moves(
    config: GameConfig,
    positions: Iterable[tuple[int, int]],
    moves: Iterable[tuple[int, int]],
    territory_every: int | None = None,
    diplomacy_events: Sequence[AllianceEvent] = (),
) -> MatchMetrics:
    """Recalcule les métriques en rejouant ``moves`` (couples joueur, direction).

    Fonction pure : sert à analyser un journal de partie après coup, y compris
    un journal produit avant l'introduction des métriques. Les événements
    diplomatiques du journal donnent le bilan diplomatique si la diplomatie de
    la configuration est active.
    """
    state = GameState(config, positions=list(positions))
    collector = MetricsCollector(config, territory_every)
    collector.start(state)
    for player, direction in moves:
        if player != state.current:
            raise ValueError(f"demi-coup {state.ply} : joueur {player} au lieu de {state.current}")
        origin = state.head(player)
        ply = state.ply
        moved = state.apply(direction)
        collector.record(
            state, player, origin, direction, state.head(player) if moved else None, not moved, ply
        )
    summary = summarize_diplomacy(diplomacy_events, state.ply) if config.diplomacy.enabled else None
    return collector.finish(state, summary)


__all__ = ["MetricsCollector", "metrics_from_moves"]
