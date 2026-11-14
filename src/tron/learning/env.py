"""Environnement d'apprentissage, inspiré de Gymnasium sans en dépendre.

.. code-block:: python

    env = TronEnv(GameConfig.from_yaml("configs/games/learning_duel_10x10.yaml"), player=0)
    obs = env.reset(seed=1)
    while not env.done:
        action = int(np.flatnonzero(obs["legal"])[0])
        obs, reward, terminated, truncated, info = env.step(action)

Un épisode est une partie vue par l'**apprenant** (joueur ``player``) ; les autres
joueurs sont joués par des agents (ceux de la configuration par défaut). Après
``reset`` et après chaque ``step``, les autres joueurs jouent jusqu'au prochain
tour de l'apprenant ou jusqu'à la fin de la partie.

* **Action** : direction 0..3 (N, E, S, O). Quand l'apprenant n'a aucun coup
  légal, l'environnement joue ``PASS`` à sa place (il est éliminé) : ce n'est
  jamais une action choisie.
* **Action illégale** (direction bloquée alors qu'un coup légal existe), règle
  ``illegal`` explicite : ``lose`` (défaut, règle du jeu : le joueur heurte un
  mur et est éliminé), ``raise`` (:class:`IllegalActionError`), ``random`` (coup
  légal tiré au hasard, flux ``STREAM_LEARNING``). Une action hors de 0..3 lève
  toujours ``ValueError``.
* **Récompense terminale** : +1 si l'équipe de victoire de l'apprenant gagne,
  −1 si une autre gagne, 0 en cas d'égalité — exactement le résultat officiel.
  Si l'apprenant est éliminé alors que son équipe joue encore, la partie est
  jouée jusqu'au bout avant de conclure l'épisode.
* **Façonnage** (``shaping="territory"``) : F = γ·Φ(s') − Φ(s), Φ = part de
  Voronoi de l'apprenant à ses tours de jeu, Φ(fin) = 0. Avec γ = 1, la somme sur
  un épisode vaut −Φ(s₀), constante : aucune incitation à prolonger ou à écourter
  la partie (façonnage par potentiel, Ng et al., 1999).
* ``terminated`` : fin de partie ; ``truncated`` : limite ``max_steps`` de
  décisions de l'apprenant atteinte, partie non terminée.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from tron.agents.base import Agent, build_agents
from tron.agents.social import social_matrix
from tron.core.config import ConfigError, GameConfig
from tron.core.directions import PASS
from tron.core.rng import STREAM_LEARNING, make_rng
from tron.core.state import GameState
from tron.learning.features import FeatureExtractor, ObservationSpec, observe

ILLEGAL_RULES = ("lose", "raise", "random")
SHAPINGS = ("none", "territory")


class IllegalActionError(ValueError):
    """Action bloquée refusée (règle ``illegal="raise"``)."""


class TronEnv:
    """Partie de Tron vue par un joueur apprenant (voir le module)."""

    def __init__(
        self,
        config: GameConfig,
        player: int = 0,
        *,
        opponents: Mapping[int, Agent] | None = None,
        illegal: str = "lose",
        shaping: str = "none",
        shaping_scale: float = 0.1,
        gamma: float = 1.0,
        max_steps: int | None = None,
        observation: ObservationSpec | None = None,
    ) -> None:
        if not 0 <= player < config.n_players:
            raise ConfigError(f"apprenant {player} hors de [0, {config.n_players - 1}]")
        if illegal not in ILLEGAL_RULES:
            raise ConfigError(f"règle d'action illégale inconnue : {illegal} ({ILLEGAL_RULES})")
        if shaping not in SHAPINGS:
            raise ConfigError(f"façonnage inconnu : {shaping} (attendu : {SHAPINGS})")
        for name, value in (("shaping_scale", shaping_scale), ("gamma", gamma)):
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"{name} doit être un réel ≥ 0 : {value}")
        if gamma > 1:
            raise ConfigError(f"gamma doit être ≤ 1 : {gamma}")
        if max_steps is not None and max_steps < 1:
            raise ConfigError(f"max_steps doit être ≥ 1 : {max_steps}")
        self.base_config = config
        self.config = config
        self.player = player
        self.illegal = illegal
        self.shaping = shaping
        self.shaping_scale = float(shaping_scale)
        self.gamma = float(gamma)
        self.max_steps = max_steps
        self.observation = observation or ObservationSpec()
        self.extractor = FeatureExtractor(config)
        self._fixed_opponents = dict(opponents) if opponents is not None else None
        self.agents: dict[int, Agent] = {}
        self._state: GameState | None = None
        self._rng = make_rng(config.seed, STREAM_LEARNING)
        self._potential = 0.0
        self._auto_pass = False
        self.steps = 0
        self.done = True

    # accès
    @property
    def state(self) -> GameState:
        """État courant (à ne pas modifier : travailler sur ``state.copy()``)."""
        if self._state is None:
            raise RuntimeError("environnement non initialisé : appeler reset()")
        return self._state

    def legal_actions(self) -> list[int]:
        """Actions légales de l'apprenant (vide si l'épisode est fini)."""
        st = self.state
        if self.done or st.current != self.player:
            return []
        return st.legal_moves()

    # épisode
    def reset(
        self,
        seed: int | None = None,
        *,
        first_player: int | None = None,
        opponents: Mapping[int, Agent] | None = None,
    ) -> dict[str, np.ndarray]:
        """Nouvelle partie (graine et premier joueur facultatifs) ; renvoie l'observation.

        Sans ``opponents``, les autres joueurs sont les agents de la configuration,
        reconstruits à chaque épisode (donc reproductibles pour une graine donnée).
        Des agents fournis sont réinitialisés par ``reset()`` mais gardent leur
        générateur aléatoire : en créer de nouveaux pour un épisode reproductible.
        """
        cfg = self.base_config
        if seed is not None or first_player is not None:
            cfg = cfg.with_seed(cfg.seed if seed is None else seed, first_player)
        self.config = cfg
        self._state = GameState(cfg)
        chosen = opponents if opponents is not None else self._fixed_opponents
        if chosen is None:
            built = build_agents(cfg)
            self.agents = {q: a for q, a in enumerate(built) if q != self.player}
        else:
            missing = set(range(cfg.n_players)) - {self.player} - set(chosen)
            if missing:
                raise ConfigError(f"agents manquants pour les joueurs {sorted(missing)}")
            self.agents = {q: a for q, a in chosen.items() if q != self.player}
            for agent in self.agents.values():
                agent.reset()
        self._rng = make_rng(cfg.seed, STREAM_LEARNING)
        self.steps = 0
        self.done = False
        self._auto_pass = False
        self._advance()
        self.done = self.state.over
        self._potential = self._phi()
        return self._observe()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Joue l'action de l'apprenant puis les autres joueurs jusqu'à son prochain tour."""
        if self.done:
            raise RuntimeError("épisode terminé : appeler reset()")
        if isinstance(action, bool) or not isinstance(action, int | np.integer):
            raise ValueError(f"action invalide : {action!r} (entier 0..3 attendu)")
        action = int(action)
        if not 0 <= action <= 3:
            raise ValueError(f"action hors de 0..3 : {action}")
        st = self.state
        legal = st.legal_moves()
        illegal = action not in legal
        move = action
        if illegal:
            if self.illegal == "raise":
                raise IllegalActionError(f"action {action} bloquée (légales : {legal})")
            if self.illegal == "random":
                move = legal[int(self._rng.integers(len(legal)))]
        self._play(move)
        self.steps += 1
        self._advance()
        terminated = st.over
        truncated = not terminated and self.max_steps is not None and self.steps >= self.max_steps
        reward = self._terminal_reward() if terminated else 0.0
        if self.shaping != "none":
            phi = 0.0 if terminated else self._phi()
            reward += self.shaping_scale * (self.gamma * phi - self._potential)
            self._potential = phi
        self.done = terminated or truncated
        info = self._info(illegal=illegal, played=move)
        return self._observe(), reward, terminated, truncated, info

    # interne
    def _play(self, move: int) -> None:
        st = self.state
        p = st.current
        origin = st.head(p)
        moved = st.apply(move)
        target = st.head(p) if moved else None
        for agent in self.agents.values():
            agent.observe_move(p, move, origin, target, not moved, st)

    def _advance(self) -> None:
        """Fait jouer les autres joueurs jusqu'au tour de l'apprenant ou à la fin."""
        st = self.state
        while not st.over:
            q = st.current
            if q == self.player:
                if st.has_legal_move(q):
                    return
                self._auto_pass = True  # aucun coup légal : PASS automatique, élimination
                self._play(PASS)
                continue
            self._play(self.agents[q].choose_move(st).move)

    def _terminal_reward(self) -> float:
        winner = self.state.winning_team()
        if winner < 0:
            return 0.0
        return 1.0 if winner == self.config.victory_team_index[self.player] else -1.0

    def _phi(self) -> float:
        st = self.state
        if self.shaping == "none" or st.over or not st.is_alive(self.player):
            return 0.0
        return float(self.extractor.territory(st)[self.player])

    def _observe(self) -> dict[str, np.ndarray]:
        st = self.state
        spec = self.observation
        territory = self.extractor.territory(st) if spec.territory else None
        prefs = None
        if spec.preferences:
            ps = self.config.players[self.player]
            profile = str(ps.params.get("profile", "team"))
            alpha = float(ps.params.get("alpha", 1.0))
            matrix = ps.params.get("matrix")
            prefs = social_matrix(profile, self.config.team_index, self.player, alpha, matrix)
        return observe(st, self.player, spec, territory=territory, preferences=prefs)

    def _info(self, *, illegal: bool = False, played: int | None = None) -> dict[str, Any]:
        st = self.state
        cfg = self.config
        info: dict[str, Any] = {
            "seed": cfg.seed,
            "first_player": cfg.first_player,
            "ply": st.ply,
            "round": st.round,
            "steps": self.steps,
            "illegal": illegal,
            "played": played,
            "auto_pass": self._auto_pass,
            "learner_alive": st.is_alive(self.player),
            "alive": [st.is_alive(q) for q in range(st.n)],
        }
        if st.over:
            winner = st.winning_team()
            info["winning_team"] = winner
            info["winner"] = cfg.victory_names[winner] if winner >= 0 else ""
            info["end_reason"] = st.end_reason()
            info["n_moves"] = st.n_moves
        elif st.is_alive(self.player):
            info["territory"] = float(self.extractor.territory(st)[self.player])
        return info

    @property
    def info(self) -> dict[str, Any]:
        """Informations sur l'état courant (comme celles renvoyées par ``step``)."""
        return self._info()


__all__ = ["ILLEGAL_RULES", "SHAPINGS", "IllegalActionError", "TronEnv"]
