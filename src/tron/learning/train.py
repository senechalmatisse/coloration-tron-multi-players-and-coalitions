"""Entraînement et évaluation de la politique linéaire de référence (P10).

Algorithme : SARSA(0) semi-gradient sur les **états après coup**. À chaque
décision, l'apprenant évalue l'état obtenu après chacun de ses coups légaux,
V(φ) = w · φ (caractéristiques ``afterstate-v1``), choisit ε-glouton, puis met à
jour la valeur de l'état après son coup précédent :

    w ← w + α · (r + γ · V(φ_suivant) − V(φ_précédent)) · φ_précédent

et, en fin de partie, vers la récompense terminale. Les adversaires sont tirés
dans un ensemble de référence (agents figés) et, en auto-jeu, parmi les
générations précédentes du modèle, figées à la fin de chaque génération.

Les graines d'entraînement (flux ``STREAM_TRAIN``) et d'évaluation (flux
``STREAM_EVAL``) sont dérivées de flux distincts ; leur disjonction est vérifiée.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import tron
from tron.agents.base import Agent, build_agents, make_agent
from tron.agents.learned import LearnedAgent
from tron.core.config import ConfigError, GameConfig
from tron.core.rng import STREAM_EVAL, STREAM_TRAIN, derive_seed, make_rng
from tron.learning.env import SHAPINGS, TronEnv
from tron.learning.features import FEATURE_NAMES
from tron.learning.model import LinearModel
from tron.sim.match import Match
from tron.sim.result import MatchResult

REFERENCES = ("greedy", "maxn", "paranoid", "social")
"""Agents de comparaison par défaut du rapport d'évaluation."""

_TOP_KEYS = frozenset(
    {
        "name",
        "game",
        "player",
        "seed",
        "episodes",
        "generations",
        "learning",
        "env",
        "opponents",
        "self_play",
        "evaluation",
    }
)


def _number(raw: Any, name: str, low: float, high: float = math.inf) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise ConfigError(f"{name} doit être un nombre fini : {raw!r}")
    if not low <= raw <= high:
        raise ConfigError(f"{name} doit être dans [{low}, {high}] : {raw}")
    return float(raw)


def _integer(raw: Any, name: str, low: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < low:
        raise ConfigError(f"{name} doit être un entier ≥ {low} : {raw!r}")
    return raw


def _mapping(raw: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} doit être un dictionnaire : {raw!r}")
    unknown = set(raw) - keys
    if unknown:
        raise ConfigError(f"clés inconnues dans {name} : {sorted(unknown)}")
    return raw


def _game(raw: Any, base: Path | None, name: str) -> tuple[str, GameConfig]:
    if isinstance(raw, Mapping):
        return "<inline>", GameConfig.from_dict(raw)
    if isinstance(raw, str):
        path = Path(raw)
        if not path.is_absolute() and base is not None:
            path = base / path
        return str(raw), GameConfig.from_yaml(path)
    raise ConfigError(f"{name} : chemin ou dictionnaire de configuration attendu ({raw!r})")


@dataclass(frozen=True)
class TrainingSpec:
    """Configuration d'entraînement (YAML de ``configs/training/``)."""

    name: str
    game: GameConfig
    game_source: str = "<inline>"
    player: int = 0
    seed: int = 0
    episodes: int = 100
    """Épisodes par génération."""
    generations: int = 1
    alpha: float = 0.05
    gamma: float = 1.0
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    shaping: str = "none"
    shaping_scale: float = 0.1
    max_steps: int | None = None
    opponents: tuple[Mapping[str, Any], ...] = ()
    """Agents de référence des places adverses (vide : agents de la configuration)."""
    self_play: bool = False
    self_play_share: float = 0.5
    eval_games: int = 20
    eval_references: tuple[str, ...] = REFERENCES
    eval_depth: int = 2
    eval_configs: tuple[tuple[str, GameConfig], ...] = ()
    """Configurations jamais vues à l'entraînement, évaluées en plus de ``game``."""

    @property
    def total_episodes(self) -> int:
        return self.episodes * self.generations

    def with_overrides(self, **changes: Any) -> TrainingSpec:
        """Copie validée où les valeurs non ``None`` de ``changes`` remplacent les champs."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None}).validated()

    def validated(self) -> TrainingSpec:
        n = self.game.n_players
        if not 0 <= self.player < n:
            raise ConfigError(f"apprenant {self.player} hors de [0, {n - 1}]")
        _integer(self.episodes, "episodes", 1)
        _integer(self.generations, "generations", 1)
        _number(self.alpha, "learning.alpha", 1e-9, 10.0)
        _number(self.gamma, "learning.gamma", 0.0, 1.0)
        _number(self.epsilon_start, "learning.epsilon.start", 0.0, 1.0)
        _number(self.epsilon_end, "learning.epsilon.end", 0.0, 1.0)
        if self.shaping not in SHAPINGS:
            raise ConfigError(f"env.shaping inconnu : {self.shaping} (attendu : {SHAPINGS})")
        _number(self.shaping_scale, "env.shaping_scale", 0.0)
        if self.max_steps is not None:
            _integer(self.max_steps, "env.max_steps", 1)
        _number(self.self_play_share, "self_play.share", 0.0, 1.0)
        _integer(self.eval_games, "evaluation.games", 0)
        _integer(self.eval_depth, "evaluation.depth", 1)
        for spec in self.opponents:
            if "agent" not in spec:
                raise ConfigError(f"adversaire sans clé agent : {dict(spec)}")
            make_agent(str(spec["agent"]), 1 if self.player == 0 else 0, self.game, **_params(spec))
        for ref in self.eval_references:
            make_agent(ref, 0, self.game, depth=self.eval_depth)
        for source, cfg in self.eval_configs:
            if cfg.n_players <= self.player:
                raise ConfigError(f"{source} : pas de joueur {self.player}")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base: Path | None = None) -> TrainingSpec:
        if not isinstance(data, Mapping):
            raise ConfigError("configuration d'entraînement invalide : dictionnaire attendu")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise ConfigError(f"clés d'entraînement inconnues : {sorted(unknown)}")
        if "game" not in data:
            raise ConfigError("clé game obligatoire (chemin ou configuration de partie)")
        source, game = _game(data["game"], base, "game")
        learn = _mapping(data.get("learning"), "learning", {"alpha", "gamma", "epsilon"})
        eps = learn.get("epsilon", {})
        if isinstance(eps, int | float) and not isinstance(eps, bool):
            eps = {"start": eps, "end": eps}
        eps = _mapping(eps, "learning.epsilon", {"start", "end"})
        env = _mapping(data.get("env"), "env", {"shaping", "shaping_scale", "max_steps"})
        sp = data.get("self_play", {})
        if isinstance(sp, bool):
            sp = {"enabled": sp}
        sp = _mapping(sp, "self_play", {"enabled", "share"})
        ev = _mapping(
            data.get("evaluation"), "evaluation", {"games", "references", "depth", "configs"}
        )
        opponents = data.get("opponents") or ()
        if not isinstance(opponents, Sequence) or isinstance(opponents, str):
            raise ConfigError("opponents doit être une liste d'agents")
        opp_specs = tuple(_agent_spec(o) for o in opponents)
        enabled = sp.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"self_play.enabled doit être un booléen : {enabled!r}")
        refs = ev.get("references", REFERENCES)
        if isinstance(refs, str) or not isinstance(refs, Sequence):
            raise ConfigError("evaluation.references doit être une liste d'agents")
        configs = tuple(_game(c, base, "evaluation.configs") for c in ev.get("configs") or ())
        max_steps = env.get("max_steps")
        spec = cls(
            name=str(data.get("name", "training")),
            game=game,
            game_source=source,
            player=_integer(data.get("player", 0), "player", 0),
            seed=_integer(data.get("seed", 0), "seed", 0),
            episodes=_integer(data.get("episodes", 100), "episodes", 1),
            generations=_integer(data.get("generations", 1), "generations", 1),
            alpha=_number(learn.get("alpha", 0.05), "learning.alpha", 1e-9, 10.0),
            gamma=_number(learn.get("gamma", 1.0), "learning.gamma", 0.0, 1.0),
            epsilon_start=_number(eps.get("start", 0.2), "learning.epsilon.start", 0.0, 1.0),
            epsilon_end=_number(eps.get("end", 0.02), "learning.epsilon.end", 0.0, 1.0),
            shaping=str(env.get("shaping", "none")),
            shaping_scale=_number(env.get("shaping_scale", 0.1), "env.shaping_scale", 0.0),
            max_steps=None if max_steps is None else _integer(max_steps, "env.max_steps", 1),
            opponents=opp_specs,
            self_play=enabled,
            self_play_share=_number(sp.get("share", 0.5), "self_play.share", 0.0, 1.0),
            eval_games=_integer(ev.get("games", 20), "evaluation.games", 0),
            eval_references=tuple(str(r) for r in refs),
            eval_depth=_integer(ev.get("depth", 2), "evaluation.depth", 1),
            eval_configs=configs,
        )
        return spec.validated()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingSpec:
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data, base=p.parent)


def _agent_spec(raw: Any) -> Mapping[str, Any]:
    """Agent adverse : nom (``greedy``) ou dictionnaire ``{agent: nom, paramètres…}``."""
    if isinstance(raw, str):
        return {"agent": raw}
    if not isinstance(raw, Mapping) or "agent" not in raw:
        raise ConfigError(f"adversaire invalide : {raw!r} (nom ou {{agent: nom, ...}})")
    return dict(raw)


def _params(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if k != "agent"}


@dataclass
class TrainingResult:
    """Modèle appris, journal des épisodes et rapport d'évaluation."""

    model: LinearModel
    episodes: list[dict[str, Any]] = field(default_factory=list)
    evaluation: list[dict[str, Any]] = field(default_factory=list)
    train_seeds: list[int] = field(default_factory=list)
    eval_seeds: list[int] = field(default_factory=list)


def train_seeds(seed: int, count: int) -> list[int]:
    return [derive_seed(seed, STREAM_TRAIN, k) for k in range(count)]


def eval_seeds(seed: int, count: int) -> list[int]:
    return [derive_seed(seed, STREAM_EVAL, g) for g in range(count)]


def _learner_won(result: MatchResult, config: GameConfig, player: int) -> int:
    """+1 victoire, 0 égalité, −1 défaite de l'équipe de victoire de ``player``."""
    if result.winning_team == "":
        return 0
    if config.rules.individual:
        return 1 if result.individual_winner == player else -1
    return 1 if result.winning_team == config.players[player].team else -1


def train(
    spec: TrainingSpec,
    *,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Entraîne le modèle linéaire selon ``spec`` (sans évaluation : voir :func:`run_training`)."""
    cfg = spec.game
    n = cfg.n_players
    learner = spec.player
    env = TronEnv(
        cfg,
        learner,
        shaping=spec.shaping,
        shaping_scale=spec.shaping_scale,
        gamma=spec.gamma,
        max_steps=spec.max_steps,
    )
    ext = env.extractor
    model = LinearModel()
    w = model.weights
    rng = make_rng(spec.seed, STREAM_TRAIN)
    total = spec.total_episodes
    seeds = train_seeds(spec.seed, total)
    team = cfg.victory_team_index
    opp_seats = [q for q in range(n) if team[q] != team[learner]]
    frozen: list[LinearModel] = []
    rows: list[dict[str, Any]] = []
    for gen in range(spec.generations):
        for e in range(spec.episodes):
            k = gen * spec.episodes + e
            frac = k / (total - 1) if total > 1 else 1.0
            eps = spec.epsilon_start + (spec.epsilon_end - spec.epsilon_start) * frac
            first = k % n
            game_cfg = cfg.with_seed(seeds[k], first)
            agents: dict[int, Agent] = {
                q: a for q, a in enumerate(build_agents(game_cfg)) if q != learner
            }
            for q in opp_seats:
                if spec.self_play and frozen and float(rng.random()) < spec.self_play_share:
                    past = frozen[int(rng.integers(len(frozen)))]
                    agents[q] = LearnedAgent(q, game_cfg, model=past)
                elif spec.opponents:
                    ref = spec.opponents[int(rng.integers(len(spec.opponents)))]
                    agents[q] = make_agent(str(ref["agent"]), q, game_cfg, **_params(ref))
            env.reset(seed=seeds[k], first_player=first, opponents=agents)
            prev: np.ndarray | None = None
            pending = 0.0
            ret = 0.0
            steps = 0
            truncated = False
            while not env.done:
                st = env.state
                moves = st.legal_moves()
                phis = ext.action_features(st, learner, moves)
                values = {m: float(w @ phis[m]) for m in moves}
                if float(rng.random()) < eps:
                    move = moves[int(rng.integers(len(moves)))]
                else:
                    best = max(values.values())
                    ties = [m for m in moves if values[m] > best - 1e-12]
                    move = ties[int(rng.integers(len(ties)))]
                if prev is not None:
                    w += spec.alpha * (pending + spec.gamma * values[move] - float(w @ prev)) * prev
                _obs, pending, _terminated, truncated, _info = env.step(move)
                ret += pending
                prev = phis[move]
                steps += 1
            if prev is not None:
                target = pending
                if truncated and env.state.current == learner and env.state.legal_moves():
                    nxt = ext.action_features(env.state, learner, env.state.legal_moves())
                    target += spec.gamma * max(float(w @ phi) for phi in nxt.values())
                w += spec.alpha * (target - float(w @ prev)) * prev
            if not np.isfinite(w).all():
                raise RuntimeError(
                    f"divergence de l'apprentissage à l'épisode {k} (alpha trop grand)"
                )
            st = env.state
            if truncated or not st.over:
                outcome = "truncated"
            else:
                final = env._terminal_reward()
                outcome = "win" if final > 0 else "loss" if final < 0 else "draw"
            row = {
                "episode": k,
                "generation": gen,
                "seed": seeds[k],
                "first_player": first,
                "epsilon": round(eps, 6),
                "steps": steps,
                "plies": st.ply,
                "return": round(ret, 6),
                "outcome": outcome,
                "opponents": ";".join(agents[q].describe() for q in sorted(agents)),
            }
            rows.append(row)
            if on_episode is not None:
                on_episode(row)
        frozen.append(model.copy())
    last = rows[-min(len(rows), 100) :]
    model.metadata = {
        "name": spec.name,
        "game": cfg.name,
        "game_source": spec.game_source,
        "player": learner,
        "seed": spec.seed,
        "episodes": total,
        "generations": spec.generations,
        "alpha": spec.alpha,
        "gamma": spec.gamma,
        "epsilon": [spec.epsilon_start, spec.epsilon_end],
        "shaping": spec.shaping,
        "self_play": spec.self_play,
        "train_seed_stream": STREAM_TRAIN,
        "recent_win_rate": round(sum(r["outcome"] == "win" for r in last) / len(last), 4),
        "code_version": tron.__version__,
        "weights": dict(zip(FEATURE_NAMES, (round(float(v), 6) for v in w), strict=True)),
    }
    return TrainingResult(model=model, episodes=rows, train_seeds=seeds)


def evaluate_model(
    model: LinearModel,
    config: GameConfig,
    *,
    player: int = 0,
    references: Sequence[str] = REFERENCES,
    games: int = 20,
    depth: int = 2,
    seed: int = 0,
    exclude_seeds: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Compare le modèle aux agents de ``references`` sur des graines d'évaluation.

    Pour chaque référence R : ``games`` parties avec le modèle à la place
    ``player`` contre R aux places adverses (coéquipiers : ceux de la
    configuration), puis les mêmes parties avec R à la place ``player`` (base de
    comparaison). Graines du flux ``STREAM_EVAL``, premier joueur tournant ;
    ``ValueError`` si elles recoupent ``exclude_seeds`` (graines d'entraînement).
    """
    if not 0 <= player < config.n_players:
        raise ConfigError(f"joueur {player} hors de la configuration {config.name}")
    seeds = eval_seeds(seed, games)
    if set(seeds) & set(exclude_seeds):
        raise ValueError("graines d'évaluation non disjointes des graines d'entraînement")
    team = config.victory_team_index
    rows: list[dict[str, Any]] = []
    for ref in references:
        for side in ("learned", ref):
            score = {1: 0, 0: 0, -1: 0}
            moves = 0
            for g, s in enumerate(seeds):
                cfg = config.with_seed(s, g % config.n_players)
                agents = build_agents(cfg)
                for q in range(cfg.n_players):
                    if q == player and side == "learned":
                        agents[q] = LearnedAgent(q, cfg, model=model)
                    elif q == player or team[q] != team[player]:
                        agents[q] = make_agent(ref, q, cfg, depth=depth)
                result = Match(cfg, agents=agents, metrics=False).run()
                score[_learner_won(result, cfg, player)] += 1
                moves += result.n_moves
            rows.append(
                {
                    "config": config.name,
                    "reference": ref,
                    "learner": side,
                    "games": games,
                    "wins": score[1],
                    "draws": score[0],
                    "losses": score[-1],
                    "win_rate": round(score[1] / games, 4) if games else 0.0,
                    "mean_moves": round(moves / games, 2) if games else 0.0,
                }
            )
    return rows


def run_training(
    spec: TrainingSpec,
    out_dir: str | Path | None = None,
    *,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Entraîne, évalue sur des graines disjointes et écrit modèle et rapports dans ``out_dir``."""
    result = train(spec, on_episode=on_episode)
    if spec.eval_games > 0:
        configs = [(spec.game_source, spec.game), *spec.eval_configs]
        for _source, cfg in configs:
            result.evaluation += evaluate_model(
                result.model,
                cfg,
                player=spec.player,
                references=spec.eval_references,
                games=spec.eval_games,
                depth=spec.eval_depth,
                seed=spec.seed,
                exclude_seeds=result.train_seeds,
            )
        result.eval_seeds = eval_seeds(spec.seed, spec.eval_games)
    if out_dir is not None:
        write_reports(result, spec, out_dir)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_markdown(
    rows: Sequence[Mapping[str, Any]], n_train: int = 0, n_eval: int = 0
) -> str:
    """Tableau Markdown : taux de victoire du modèle et de la référence à la même place."""
    lines = [
        "| Configuration | Référence | Modèle appris | Référence à sa place | Écart |",
        "|---|---|---|---|---|",
    ]
    by_key = {(r["config"], r["reference"], r["learner"]): r for r in rows}
    for (cfg, ref, side), r in by_key.items():
        if side != "learned":
            continue
        base = by_key.get((cfg, ref, ref))
        own = float(r["win_rate"])
        other = float(base["win_rate"]) if base else float("nan")
        lines.append(
            f"| {cfg} | {ref} | {own:.0%} ({r['wins']}/{r['games']}) | "
            f"{other:.0%} ({base['wins'] if base else '?'}/{r['games']}) | {own - other:+.0%} |"
        )
    if n_train or n_eval:
        lines.append("")
        lines.append(
            f"Graines : {n_train} d'entraînement (flux {STREAM_TRAIN}), {n_eval} d'évaluation "
            f"(flux {STREAM_EVAL}), ensembles disjoints (vérifié)."
        )
    return "\n".join(lines) + "\n"


def write_reports(result: TrainingResult, spec: TrainingSpec, out_dir: str | Path) -> Path:
    """Écrit ``model.json``, ``training.csv``, ``evaluation.csv`` et ``evaluation.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.model.save(out / "model.json")
    write_csv(out / "training.csv", result.episodes)
    if result.evaluation:
        write_csv(out / "evaluation.csv", result.evaluation)
        (out / "evaluation.md").write_text(
            f"# Évaluation du modèle {spec.name}\n\n"
            + evaluation_markdown(
                result.evaluation, len(result.train_seeds), len(result.eval_seeds)
            ),
            encoding="utf-8",
        )
    return out


__all__ = [
    "REFERENCES",
    "TrainingResult",
    "TrainingSpec",
    "eval_seeds",
    "evaluate_model",
    "evaluation_markdown",
    "run_training",
    "train",
    "train_seeds",
    "write_csv",
    "write_reports",
]
