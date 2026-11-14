# API Python publique

`tron.api` est la façade stable du simulateur pour un notebook, un autre programme Python ou un outil externe, sans passer par la CLI. Les noms de `tron.api.__all__` forment l'API publique (version `API_VERSION = "1.0"`) ; ils sont aussi accessibles depuis le paquet (`tron.run_game`), chargés à la demande pour que `import tron` reste léger. Le reste du paquet (`tron.core`, `tron.sim`…) reste importable mais peut évoluer.

```python
from tron.api import load_game_config, load_replay, run_game, run_games

config = load_game_config("configs/games/duel_20x20.yaml")
result = run_game(config, seed=3, log="partie.jsonl")   # sans affichage
print(result.winning_team, result.n_moves, result.end_reason)

results = run_games(config, games=20, workers=4, out_dir="results/duel", logs=True)
final = load_replay("partie.jsonl").final_state()
```

## Fonctions

| Fonction | Rôle |
|---|---|
| `load_game_config(source)` | configuration depuis un fichier YAML ou JSON, un dictionnaire, ou un `GameConfig` (renvoyé tel quel) ; `ConfigError` si le fichier manque ou si la configuration est invalide |
| `create_match(config, observers=(), *, seed, first_player, agents, metrics)` | prépare une partie sans la jouer (`match.step()` / `match.run()`) ; `agents` remplace les agents de la configuration |
| `run_game(config, seed=None, observers=(), *, first_player, log, metrics)` | joue une partie sans affichage ; `log` écrit le journal JSONL rejouable |
| `run_games(config, games, workers=1, out_dir=None, *, logs, on_result)` | série de parties (graines dérivées et premier joueur tournant, comme `tron play --games`) ; avec `workers > 1`, des processus, mêmes résultats que la version séquentielle (hors durées) et dans l'ordre ; `out_dir` reçoit `results.csv` et, avec `logs=True`, `game_XXXX.jsonl` |
| `load_replay(path)` | journal `tron-log/1` ou `tron-log/2` : `states()`, `final_state()`, `details(i)`, `metrics()`… |
| `resume_game(path, observers=())` | reprend une sauvegarde `tron-checkpoint/1` (voir `docs/sauvegardes.md`) |

Le paramètre `config` accepte partout les mêmes sources que `load_game_config`.

## Types publics

`GameConfig`, `GameState`, `Outcome`, `Match`, `MatchObserver`, `MoveEvent`, `MatchResult`, `GameLog`, `Recorder`, `Agent`, `Decision`, `DecisionDetails`, `SearchStats`, `Heuristic`, ainsi que les registres et fabriques `AGENTS`, `register_agent`, `make_agent`, `build_agents`, `HEURISTICS`, `register_heuristic`, `make_heuristic`. Ce sont les classes internes historiques (mêmes objets : `tron.api.Match is tron.sim.Match`), aucune enveloppe.

## Contrats de mutabilité

- **`GameConfig` est immuable** (dataclass gelée) : `with_seed` et `from_dict` renvoient de nouveaux objets. Une configuration se partage sans risque entre parties, processus et fils.
- **`GameState` est mutable** : `apply` et `undo` le modifient en place. Un agent reçoit l'état de la partie et doit le rendre intact (`apply`/`undo` protégés par `try/finally`) ; `copy()` donne un état indépendant.
- **`Match` est à usage unique** : `run()` joue la partie jusqu'au bout et la consomme ; un nouvel appel renvoie le même `MatchResult` sans rejouer. Pour une autre partie, créer un autre `Match`.
- **`MatchResult` et `GameLog`** sont des valeurs à lire.

## Exceptions publiques

Toutes sous-classes de `ValueError` : `ConfigError` (configuration invalide ou introuvable), `LogError` (journal illisible, introuvable ou incohérent avec les règles), `CheckpointError` (sauvegarde), `ModelError` (modèle appris, `docs/learning.md`).

## Exemples (`examples/`)

| Script | Contenu |
|---|---|
| `partie_unique.py` | une partie, son journal, son état final en ASCII |
| `campagne.py` | une série de parties en parallèle et un résumé des victoires |
| `rejeu.py` | rejoue un journal état par état |
| `agent_personnalise.py` | un agent enregistré par `register_agent`, utilisé dans une configuration |

```bash
python examples/partie_unique.py configs/games/duel_20x20.yaml
python examples/campagne.py configs/games/teams_2v2_20x20.yaml 8 2
tron play configs/games/duel_20x20.yaml --no-gui --out /tmp/duel && python examples/rejeu.py /tmp/duel/game_0000.jsonl
python examples/agent_personnalise.py
```

Les quatre scripts sont exécutés par `tests/test_api.py`.