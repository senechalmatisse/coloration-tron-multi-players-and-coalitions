"""Agent Monte-Carlo Tree Search à utilité vectorielle (P04).

Chaque nœud mémorise le nombre de visites et la **somme des vecteurs de
récompense** (une composante par joueur) des simulations qui l'ont traversé.
Au nœud où le joueur p a le trait, la sélection maximise

    UCT(c, p) = U_p(c) + C · √(ln N(nœud) / N(c)),   U_p(c) = W[p] · (somme(c) / N(c))

où W est la matrice de préférences sociales (profil ``maxn`` par défaut : W = I,
chacun maximise sa propre composante). C'est un MCTS multijoueur à utilité
vectorielle, pas un UCT à deux joueurs : le vecteur brut est rétropropagé et
l'utilité est calculée à la sélection, comme le moteur MaxN de SOS.

Déroulement d'une simulation, sur une copie de l'état (``apply``/``undo``,
restauration garantie par ``try/finally``) :

1. **sélection** : descente par UCT tant que le nœud est entièrement développé ;
2. **expansion** : un coup non essayé (ordre tiré avec la graine) ; un joueur sans
   coup légal joue ``PASS`` (élimination), comme dans les moteurs exacts ;
3. **simulation** : politique ``random`` (coup sûr au hasard) ou ``greedy`` (coup
   qui maximise le nombre de cases libres voisines de la case d'arrivée, départage
   au hasard), jusqu'à la fin de partie ou ``rollout_depth`` demi-coups ;
4. **évaluation** : :meth:`~tron.eval.valuation.Valuation.vector` de l'état atteint,
   vu de la racine (victoire dominante, parts de l'heuristique sinon, défaites
   datées) ;
5. **rétropropagation** du vecteur brut sur le chemin.

Coup joué : l'enfant de la racine le plus visité ; à égalité, la règle de
départage de l'agent (F3.7). Aucun élagage alpha-bêta ou de Korf n'est utilisé.

Statistiques : ``nodes`` = nœuds créés (racine comprise), ``leaves`` = simulations,
``depth`` = profondeur maximale de l'arbre atteinte, ``ties`` = coups racine à
égalité de visites, ``value`` = utilité moyenne du coup choisi pour l'agent,
``pv`` = suite des enfants les plus visités.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from tron.agents.base import Agent, DecisionDetails, SearchStats, register_agent
from tron.agents.social import PROFILES, social_matrix, sparse_rows
from tron.core.config import ConfigError, GameConfig
from tron.core.directions import PASS
from tron.core.state import FREE, GameState

ROLLOUT_POLICIES = ("random", "greedy")


class _Node:
    """Nœud de l'arbre : coup entrant, joueur au trait, visites, somme des récompenses."""

    __slots__ = ("children", "move", "player", "sum", "terminal", "untried", "visits")

    def __init__(self, move: int, n: int) -> None:
        self.move = move
        self.children: list[_Node] = []
        self.sum = [0.0] * n
        self.visits = 0
        self.player = -1
        self.terminal = False
        self.untried: list[int] = []


@register_agent("mcts")
class MCTSAgent(Agent):
    """MCTS / UCT multijoueur à utilité vectorielle (voir le module).

    Paramètres : ``simulations`` (≥ 1), ``exploration`` (C ≥ 0), ``rollout_policy``
    (``random`` ou ``greedy``), ``rollout_depth`` (≥ 1), ``time_limit`` (secondes,
    facultatif : rend les parties dépendantes de la machine, à éviter dans les
    campagnes), ``profile``/``alpha``/``matrix`` (préférences sociales),
    ``forced_shortcut`` (coup unique joué sans simulation).
    """

    def __init__(
        self,
        player: int,
        config: GameConfig,
        *,
        simulations: int = 1000,
        exploration: float = math.sqrt(2.0),
        rollout_policy: str = "greedy",
        rollout_depth: int = 80,
        time_limit: float | None = None,
        profile: str = "maxn",
        alpha: float = 1.0,
        matrix: Sequence[Sequence[float]] | None = None,
        forced_shortcut: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(player, config, **kwargs)
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
            raise ConfigError(f"simulations doit être un entier >= 1 : {simulations!r}")
        if isinstance(exploration, bool) or not isinstance(exploration, int | float):
            raise ConfigError(f"exploration doit être un nombre : {exploration!r}")
        if not math.isfinite(exploration) or exploration < 0:
            raise ConfigError(f"exploration doit être finie et >= 0 : {exploration!r}")
        if rollout_policy not in ROLLOUT_POLICIES:
            raise ConfigError(
                f"politique de simulation inconnue : {rollout_policy} "
                f"(attendu : {ROLLOUT_POLICIES})"
            )
        if isinstance(rollout_depth, bool) or not isinstance(rollout_depth, int):
            raise ConfigError(f"rollout_depth doit être un entier : {rollout_depth!r}")
        if rollout_depth < 1:
            raise ConfigError("rollout_depth doit être >= 1")
        if time_limit is not None and not (math.isfinite(time_limit) and time_limit > 0):
            raise ConfigError("time_limit doit être > 0")
        if profile not in PROFILES:
            raise ConfigError(f"profil inconnu : {profile} (attendu : {PROFILES})")
        try:
            w = social_matrix(profile, config.team_index, player, alpha, matrix)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self.simulations = simulations
        self.exploration = float(exploration)
        self.rollout_policy = rollout_policy
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit
        self.profile = profile
        self.matrix = w
        self._rows = sparse_rows(w)
        self.forced_shortcut = forced_shortcut
        self.last_root: dict[int, tuple[int, float]] = {}
        """Dernière décision : coup racine → (visites, utilité moyenne pour l'agent)."""

    # ----------------------------------------------------------------- outils
    def _utility(self, p: int, node: _Node) -> float:
        s = node.sum
        row = self._rows[p]
        u = s[p] if row is None else sum(w * s[j] for j, w in row)
        return u / node.visits

    def _init_node(self, node: _Node, st: GameState, rnd: random.Random) -> None:
        if st.teams_alive <= 1:
            node.terminal = True
            return
        node.player = st.current
        moves = st.legal_moves()
        if moves:
            rnd.shuffle(moves)
            node.untried = moves
        else:
            node.untried = [PASS]

    def _select(self, node: _Node) -> _Node:
        p = node.player
        row = self._rows[p]
        log_n = math.log(node.visits)
        c = self.exploration
        best = node.children[0]
        best_val = -math.inf
        for ch in node.children:
            s = ch.sum
            u = (s[p] if row is None else sum(w * s[j] for j, w in row)) / ch.visits
            val = u + c * math.sqrt(log_n / ch.visits)
            if val > best_val:
                best_val = val
                best = ch
        return best

    def _greedy(self, st: GameState, moves: list[int], rnd: random.Random) -> int:
        g = st._g
        h = st._h[st.current]
        off = st._off
        s = st.stride
        best = -1
        cands: list[int] = []
        for m in moves:
            t = h + off[m]
            free = (g[t - s] == FREE) + (g[t + 1] == FREE) + (g[t + s] == FREE) + (g[t - 1] == FREE)
            if free > best:
                best = free
                cands = [m]
            elif free == best:
                cands.append(m)
        return cands[0] if len(cands) == 1 else cands[rnd.randrange(len(cands))]

    def _rollout(self, st: GameState, root_ply: int, rnd: random.Random) -> list[float]:
        steps = 0
        greedy = self.rollout_policy == "greedy"
        try:
            while st.teams_alive > 1 and steps < self.rollout_depth:
                moves = st.legal_moves()
                if not moves:
                    st.apply(PASS)
                elif len(moves) == 1:
                    st.apply(moves[0])
                elif greedy:
                    st.apply(self._greedy(st, moves, rnd))
                else:
                    st.apply(moves[rnd.randrange(len(moves))])
                steps += 1
            return self.valuation.vector(st, root_ply)
        finally:
            for _ in range(steps):
                st.undo()

    # ---------------------------------------------------------------- décision
    def _decide(self, state: GameState, moves: list[int]) -> tuple[int, SearchStats]:
        if len(moves) == 1 and self.forced_shortcut:
            self.last_root = {moves[0]: (0, math.nan)}
            self._details = DecisionDetails(scores=((moves[0], None),), note="coup forcé")
            return moves[0], SearchStats(nodes=1, depth=0, engine="mcts", pv=(moves[0],))
        st = state.copy(history=False)  # l'état reçu n'est jamais modifié
        n = st.n
        root_ply = st.ply
        rnd = random.Random(int(self.rng.integers(1 << 62)))
        root = _Node(PASS, n)
        self._init_node(root, st, rnd)
        if root.terminal:  # pragma: no cover - garde-fou : partie déjà finie
            move = self.pick(moves)
            return move, SearchStats(nodes=1, depth=0, engine="mcts", pv=(move,))
        deadline = perf_counter() + self.time_limit if self.time_limit else None
        created = 1
        rollouts = 0
        max_depth = 0
        for _ in range(self.simulations):
            if deadline is not None and rollouts > 0 and perf_counter() > deadline:
                break
            node = root
            path = [root]
            applied = 0
            try:
                while not node.terminal and not node.untried:
                    node = self._select(node)
                    st.apply(node.move)
                    applied += 1
                    path.append(node)
                if not node.terminal and node.untried:
                    move = node.untried.pop()
                    st.apply(move)
                    applied += 1
                    child = _Node(move, n)
                    self._init_node(child, st, rnd)
                    node.children.append(child)
                    path.append(child)
                    created += 1
                max_depth = max(max_depth, applied)
                reward = self._rollout(st, root_ply, rnd)
            finally:
                for _ in range(applied):
                    st.undo()
            for nd in path:
                nd.visits += 1
                acc = nd.sum
                for j in range(n):
                    acc[j] += reward[j]
            rollouts += 1
        p = self.player
        most = max(ch.visits for ch in root.children)
        ties = [ch.move for ch in root.children if ch.visits == most]
        move = self.pick(ties)
        chosen = next(ch for ch in root.children if ch.move == move)
        self.last_root = {ch.move: (ch.visits, self._utility(p, ch)) for ch in root.children}
        self._details = DecisionDetails(
            scores=tuple((m, self.last_root[m][1]) for m in moves if m in self.last_root),
            visits=tuple((m, self.last_root[m][0]) for m in moves if m in self.last_root),
            social_weights=tuple(
                (j, w) for j, w in enumerate(self.matrix[p]) if w != 0.0 and self.profile != "maxn"
            ),
            note="utilité moyenne des simulations",
        )
        pv = [move]
        node = chosen
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            pv.append(node.move)
        return move, SearchStats(
            nodes=created,
            leaves=rollouts,
            prunes=0,
            value=self._utility(p, chosen),
            depth=max_depth,
            ties=len(ties),
            engine="mcts",
            pv=tuple(pv),
        )

    def describe(self) -> str:
        return f"mcts(n={self.simulations},c={self.exploration:.2f},{self.rollout_policy})"


__all__ = ["ROLLOUT_POLICIES", "MCTSAgent"]
