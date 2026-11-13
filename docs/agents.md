# Agents

`tron agents list` affiche les agents disponibles (intégrés et extensions) avec leurs paramètres ; pour distribuer un agent dans un paquet séparé, voir `docs/extensions.md`.

## 1. Contrat d'un agent

Un agent hérite de `tron.agents.base.Agent` et s'enregistre sous un nom avec `@register_agent`. Le nom est celui de la clé `agent` du YAML ; les autres clés du joueur (sauf `name`, `team`, `depth`) sont passées au constructeur.

```python
from tron.agents.base import Agent, SearchStats, register_agent

@register_agent("prudent")
class PrudentAgent(Agent):
    """Joue le coup qui laisse le plus de cases libres autour de la tête."""

    def __init__(self, player, config, *, marge: int = 1, **kwargs):
        super().__init__(player, config, **kwargs)   # depth, rounds, tie_break, seed, heuristic
        self.marge = marge

    def _decide(self, state, moves):
        scores = {}
        for m in moves:
            state.apply(m)
            scores[m] = len(state.legal_moves(self.player))
            state.undo()                              # l'état reçu doit être restauré
        best = max(scores.values())
        ties = [m for m in moves if scores[m] == best]
        move = self.pick(ties)                        # départage F3.7 (graine ou N, E, S, O)
        return move, SearchStats(nodes=len(moves), depth=1, ties=len(ties), engine=self.name)
```

```yaml
players:
  - {name: p0, agent: prudent, marge: 2}
```

Règles :

| Point | Exigence |
|---|---|
| état | `choose_move` reçoit une copie de l'état et doit la laisser identique (`apply`/`undo`, de préférence dans `try/finally`) |
| coups | `_decide(state, moves)` reçoit au moins un coup légal ; sans coup légal, `choose_move` renvoie `PASS` sans appeler `_decide` |
| aléa | uniquement `self.rng` (flux `make_rng(graine, STREAM_AGENT, joueur)`), jamais le générateur global |
| paramètres | un paramètre inconnu ou invalide lève `ConfigError` (`make_agent` convertit les `TypeError`) |
| statistiques | `SearchStats` : `nodes`, `leaves`, `prunes`, `time` (rempli par `choose_move`), `value`, `depth`, `ties`, `engine`, `pv` |

### Décision et crochets facultatifs

`choose_move(state, player=None, context=None) -> Decision(move, stats, diplomacy=None)`.

| Crochet | Par défaut | Usage |
|---|---|---|
| `_decide_with_context(state, moves, context)` | appelle `_decide` | agents qui exploitent le contexte (diplomatie) |
| `_diplomacy(state, move, context)` | `None` | intention diplomatique de fin de décision (P03) |
| `observe_move(player, direction, origin, target, eliminated, state)` | rien | observer chaque demi-coup de la partie (P05) ; `state` est l'état réel après le coup, à ne pas modifier |
| `reset()` | rien | réinitialiser la mémoire entre deux parties (appelé par `Match`) |

`DecisionContext` contient le demi-coup courant et, si la diplomatie est active, un `DiplomacySnapshot` (voir `docs/coalitions-dynamiques.md`). Un agent externe dont `choose_move` n'accepte pas `context` est appelé sans contexte.

## 2. Agents fournis

| Agent | Algorithme | Paramètres principaux |
|---|---|---|
| `random` | coup sûr tiré au hasard | – |
| `greedy` | profondeur 1, maximise sa part de l'heuristique | `tie_break` |
| `negamax` | negamax alpha-bêta, deux joueurs (référence) | `depth` |
| `paranoid` | alpha-bêta deux camps : tous contre la racine | `depth`, `rounds`, `ordering`, `iterative_deepening`, `tt`, `time_limit` |
| `maxn` | MaxN avec élagage superficiel de Korf | `depth`, `pruning` |
| `social` | matrice de préférences W (SOS) | `profile` (`maxn`, `paranoid`, `team`, `team_paranoid`, `custom`, `dynamic_team`), `alpha`, `matrix`, `engine`, `diplomacy`, `break_margin` |
| `mcts` | MCTS / UCT multijoueur à utilité vectorielle (P04) | `simulations`, `exploration`, `rollout_policy`, `rollout_depth`, `time_limit`, `profile`, `alpha` |
| `adaptive` | modèle d'adversaires et ajustement borné des poids (P05) | `adaptation_rate`, `observation_window`, `fallback_agent`, `fallback_profile`, `min_confidence`, `cooperation_threshold`, `diplomacy` |
| `learned` | politique linéaire apprise sur les états après coup (P10, `docs/learning.md`) | `model` (fichier `model.json`), `epsilon` |

Paramètres communs : `depth` (plies) ou `rounds` (tours complets : rounds × joueurs vivants), `tie_break` (`random` ou `fixed`).

### 2.1 MCTS (`mcts`)

```yaml
- name: mcts_0
  team: A
  agent: mcts
  simulations: 1500          # >= 1
  exploration: 1.41421356237 # constante C de UCT, >= 0
  rollout_policy: greedy     # random | greedy (mobilité de la case d'arrivée)
  rollout_depth: 80          # >= 1 demi-coups par simulation
  tie_break: random
```

Principe, statistiques et première comparaison : `docs/algorithmes.md`, §8. `time_limit` existe mais rend les parties dépendantes de la machine.

### 2.2 Agent adaptatif (`adaptive`)

```yaml
- name: adaptive
  team: A
  agent: adaptive
  depth: 3
  adaptation_rate: 0.25      # λ dans [0, 1] : amplitude maximale de l'ajustement
  observation_window: 20     # N dernières décisions de chaque adversaire
  fallback_agent: social     # social | maxn | paranoid | greedy
  fallback_profile: team     # profil de l'agent social de repli
```

**Modèle d'adversaire** (`tron.agents.opponents.OpponentModel`). Pour chaque coup observé d'un adversaire q, le coup joué est comparé aux autres coups légaux de q selon quatre critères (agressif : réduire les options des autres puis se rapprocher ; territorial : part de Voronoi ; survie : taille de la région accessible puis mobilité ; coopératif : parts des partenaires). L'alignement vaut 0 pour le pire coup, 1 pour le meilleur, et n'est pas compté si tous les coups se valent. Les caractéristiques brutes (variations de mobilité, de territoire et de composantes, distance aux autres têtes, impact sur la mobilité des autres) sont aussi conservées dans la fenêtre. Les scores moyens, tirés vers ½ par deux observations fictives, donnent par softmax les probabilités des profils `aggressive`, `territorial`, `survival`, `cooperative` et `random_like`. Le modèle estime des probabilités à partir de choix observables : il ne prétend pas deviner une intention.

**Ajustement.** w_t = (1 − λ)·w_base + λ·moyenne sur les adversaires de [w_base + Σ_k e_q(k)·(T_k − w_base)], où e_q(k) = max(0, p_q(k) − p⁰(k)) est l'excès de probabilité du profil k par rapport aux probabilités sans aucune observation p⁰ (un adversaire non observé ne déplace pas les poids) et T_k est la cible du profil. Comme Σ_k e_q(k) ≤ 1, w_t reste une combinaison convexe de w_base et des cibles :

| Profil estimé | Cible T_k | Intention |
|---|---|---|
| `aggressive` | plus de `safety` et de `mobility` | garder des échappatoires |
| `territorial` | plus de `bottleneck` | contrôler les passages, éviter les affrontements frontaux |
| `survival` | plus de `territory` et de `connectivity` | lui couper les zones ouvertes qu'il recherche |
| `cooperative` | poids de base | partenaire tactique potentiel (diplomatie) |
| `random_like` | poids de base | rien à exploiter |

Tant que les poids valent w_base (λ = 0, ou aucune confiance au-dessus de `min_confidence`), l'agent évalue avec l'heuristique de la configuration et reproduit exactement son agent de repli. Avec la diplomatie active, il propose une alliance à l'adversaire le plus probablement coopératif (probabilité ≥ `cooperation_threshold`), sinon il suit la politique « contre le meneur ».

Validation (`tests/test_opponents.py`, `tests/test_adaptive_agent.py`) : probabilités finies, dans [0, 1] et de somme 1 ; mêmes observations, mêmes probabilités ; remise à zéro entre deux parties ; joueur inexistant refusé ; adversaires synthétiques (agressif, territorial, survie, aléatoire) classés bien au-dessus du hasard (test binomial) ; résultats identiques en un ou deux processus.

## 3. Construire et inspecter les agents

```python
from tron.agents import build_agents, make_agent, AGENTS
agents = build_agents(config)                     # un agent par joueur
agent = make_agent("mcts", 0, config, simulations=200)
sorted(AGENTS)                                    # agents enregistrés
```

Les agents d'autres paquets s'ajoutent sans modifier ce dépôt par des points d'entrée `tron.agents` (voir `docs/extensions.md`).
