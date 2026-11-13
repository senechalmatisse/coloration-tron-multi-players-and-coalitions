# Apprentissage (P10)

Le paquet `tron.learning` fournit un environnement d'apprentissage par renforcement, un encodage des observations, une politique linéaire de référence entraînable et un rapport d'évaluation. **NumPy suffit** : aucun cadre d'apprentissage n'est requis, ni pour entraîner, ni pour recharger et jouer un modèle. L'extra facultatif `learning` (`pip install "tron[learning]"`) installe Gymnasium pour l'adaptateur `tron.learning.gym`.

```bash
tron train configs/training/linear_duel.yaml --out models/linear_duel
tron evaluate-model models/linear_duel/model.json configs/games/mcts_duel_12x12.yaml --games 40
tron play configs/games/learning_duel_10x10.yaml --no-gui   # avec agent: learned, voir plus bas
```

## 1. Environnement `TronEnv`

```python
import numpy as np
from tron.core import GameConfig
from tron.learning import ObservationSpec, TronEnv

env = TronEnv(
    GameConfig.from_yaml("configs/games/learning_duel_10x10.yaml"),
    player=0,                      # l'apprenant
    illegal="lose",                # lose | raise | random
    shaping="territory",           # none | territory
    shaping_scale=0.1,
    gamma=1.0,
    max_steps=200,                 # décisions de l'apprenant avant troncature
    observation=ObservationSpec(distances=True, territory=True),
)
obs = env.reset(seed=1)            # graine (et premier joueur) facultatifs
while not env.done:
    action = int(np.flatnonzero(obs["legal"])[0])
    obs, reward, terminated, truncated, info = env.step(action)
```

Un épisode est une partie vue par l'apprenant. Les autres joueurs sont joués par les agents de la configuration, reconstruits à chaque `reset` (donc reproductibles pour une graine donnée), ou par des agents fournis : `env.reset(seed=…, opponents={1: agent})`. Après `reset` et après chaque `step`, ils jouent jusqu'au prochain tour de l'apprenant ou jusqu'à la fin de la partie. L'environnement n'importe ni Arcade ni `tron.viz` (vérifié par un test).

**Actions.** Direction 0..3 (N, E, S, O). Une action hors de 0..3 (ou un booléen, un flottant) lève toujours `ValueError`.

**Action illégale** (direction bloquée alors qu'un coup légal existe), règle `illegal` :

| Règle | Effet |
|---|---|
| `lose` (défaut) | règle du jeu : le joueur heurte l'obstacle et est éliminé ; `info["illegal"]` vrai |
| `raise` | `IllegalActionError` (sous-classe de `ValueError`), rien n'est joué |
| `random` | remplacée par un coup légal tiré au hasard (flux `STREAM_LEARNING`, déterministe pour une graine) ; `info["played"]` donne le coup joué |

**Aucun coup légal.** L'environnement joue `PASS` à la place de l'apprenant (il est éliminé) : ce n'est jamais une action choisie, et une observation non terminale a toujours au moins une action légale. `info["auto_pass"]` le signale.

**Récompense.** Terminale : +1 si l'équipe de victoire de l'apprenant gagne, −1 si une autre gagne, 0 en cas d'égalité — exactement le résultat officiel (`GameState.winning_team`). Si l'apprenant est éliminé alors que son équipe joue encore, la partie est jouée jusqu'au bout avant de conclure l'épisode.

**Façonnage** (`shaping: territory`) : à chaque décision, F = γ·Φ(s′) − Φ(s), où Φ est la part de Voronoi de l'apprenant à ses tours de jeu et Φ(fin) = 0. Avec γ = 1, la somme des termes sur un épisode vaut −Φ(s₀), une constante : le façonnage ne crée aucune incitation à prolonger ou à écourter la partie (façonnage par potentiel, Ng, Harada et Russell, 1999). Un test vérifie que le retour vaut exactement récompense terminale − échelle · Φ(s₀).

**Fin d'épisode.** `terminated` : partie finie. `truncated` : `max_steps` décisions de l'apprenant atteintes, partie non finie. Un `step` après la fin lève `RuntimeError`.

**`info`** : `seed`, `first_player`, `ply`, `round`, `steps`, `illegal`, `played`, `auto_pass`, `learner_alive`, `alive`, puis `territory` (partie en cours) ou `winning_team`, `winner`, `end_reason`, `n_moves` (partie finie). `env.info` donne les mêmes champs hors de `step`.

### Observation (`tron-obs/1`)

Dictionnaire de tableaux NumPy, **relatif à l'apprenant** : le joueur relatif 0 est l'apprenant, puis les autres dans l'ordre de jeu (`relative_order`).

| Clé | Forme | Contenu |
|---|---|---|
| `planes` | (C, H+2, W+2) `float32` | plans `free`, `border`, `obstacle`, `walls_k`, `head_k` (k = joueur relatif), puis `distance` si demandé : 1 / (1 + d) depuis la tête de l'apprenant, 0 si inaccessible |
| `alive` | (n,) | joueurs vivants |
| `same_team` | (n,) | coéquipiers (équipe de victoire), l'apprenant compris |
| `current` | () `int64` | joueur relatif au trait |
| `legal` | (4,) `bool` | masque des actions légales |
| `territory` | (n,) | parts de Voronoi (`ObservationSpec(territory=True)`) |
| `preferences` | (n, n) | matrice de préférences sociales du profil de l'apprenant (`preferences=True`) |

`channel_names(n, distances)` donne les noms des plans.

## 2. Politique linéaire de référence

**Caractéristiques `afterstate-v1`** (`FeatureExtractor`) : décrivent l'état **après** un coup de l'apprenant, toutes dans [−1, 1] et indépendantes de la taille de la grille (un modèle entraîné sur 10×10 s'évalue sur 12×12). Elles reprennent les composantes de l'heuristique composite (P01) :

| Caractéristiques | Contenu |
|---|---|
| `bias` | 1 |
| `own_territory`, `own_mobility`, `own_connectivity`, `own_safety`, `own_bottleneck` | composantes composites de l'apprenant |
| `ally_territory` ; `opp_territory`, `opp_mobility`, `opp_safety` | territoire des coéquipiers ; territoire, mobilité moyenne et sécurité maximale des adversaires vivants |
| `allies_alive`, `opponents_alive`, `opp_distance` | fractions vivantes, distance de Manhattan à la tête adverse la plus proche / (W + H) |
| `zone_lead`, `in_zone` | contrôle de zone : avance au score / horizon, tête dans une zone |
| `survive_progress`, `opp_survive_progress` | objectifs `survive` : progression propre et adverse |
| `win`, `loss` | état terminal gagné ou perdu (ou apprenant éliminé) |

**Valeur** V(φ) = w · φ ; l'agent joue le coup dont l'état après coup a la plus grande valeur (égalités départagées par le générateur de l'agent).

**Entraînement** : SARSA(0) semi-gradient sur les états après coup, ε-glouton (ε décroît linéairement de `start` à `end`) :

    w ← w + α · (r + γ · V(φ_suivant) − V(φ_précédent)) · φ_précédent

et, en fin de partie, vers la récompense terminale (troncature : amorçage sur la meilleure valeur suivante). Les places adverses sont tirées à chaque épisode parmi les agents de référence ; en **auto-jeu**, le modèle est figé à la fin de chaque génération et rejoint les adversaires possibles (proportion `self_play.share`). Les coéquipiers restent les agents de la configuration. Le premier joueur tourne d'un épisode à l'autre. Tout l'aléa vient de `tron.core.rng` : deux entraînements de même graine donnent les mêmes poids (testé).

### Configuration d'entraînement (`configs/training/*.yaml`)

| Clé | Défaut | Rôle |
|---|---|---|
| `name` | `training` | nom du modèle (dossier `models/<name>` par défaut) |
| `game` | obligatoire | configuration de partie : chemin (relatif au fichier) ou dictionnaire |
| `player` | 0 | place de l'apprenant |
| `seed` | 0 | graine (épisodes : flux `STREAM_TRAIN` ; évaluation : `STREAM_EVAL`) |
| `episodes`, `generations` | 100, 1 | épisodes par génération, générations |
| `learning.alpha`, `learning.gamma` | 0,05 ; 1 | pas d'apprentissage, actualisation |
| `learning.epsilon` | `{start: 0.2, end: 0.02}` | exploration (un nombre : constante) |
| `env.shaping`, `env.shaping_scale`, `env.max_steps` | `none`, 0,1, aucun | façonnage et troncature |
| `opponents` | agents de la partie | agents de référence des places adverses (`greedy` ou `{agent: paranoid, depth: 2}`) |
| `self_play` | désactivé | `{enabled: true, share: 0.3}` (ou `true`) |
| `evaluation.games`, `.references`, `.depth`, `.configs` | 20, greedy/maxn/paranoid/social, 2, aucune | rapport d'évaluation, configurations jamais vues |

Toute clé inconnue, valeur hors bornes, ou agent inconnu lève une erreur explicite. `tron train` accepte `--episodes`, `--generations`, `--eval-games`, `--seed` et `--quiet`.

### Modèle (`tron-model/1`) et agent `learned`

`model.json` contient le format, le jeu de caractéristiques et leurs noms, les poids, la version du code et les métadonnées (configuration d'entraînement, graines, taux de victoire récent, poids nommés). Il se recharge avec `load_model` (erreur `ModelError` explicite si le format, les caractéristiques ou les poids ne conviennent pas) et se joue avec NumPy seul : charger l'agent n'importe ni l'environnement ni l'entraînement (vérifié par un test).

```yaml
players:
  - {name: appris, team: A, agent: learned, model: models/linear_duel/model.json}
  - {name: rival, team: B, agent: paranoid, depth: 2}
```

Le chemin est relatif au dossier courant ; `epsilon` (défaut 0) joue un coup légal au hasard avec cette probabilité.

## 3. Évaluation

`run_training` (et `tron train`) écrit `model.json`, `training.csv` (un épisode par ligne : graine, premier joueur, ε, décisions, retour, issue, adversaires), `evaluation.csv` et `evaluation.md`. Pour chaque référence R et chaque configuration (celle d'entraînement puis les configurations jamais vues) : `evaluation.games` parties avec le modèle à la place `player` contre R aux places adverses, puis les mêmes parties avec R à cette place (base de comparaison). Les graines d'évaluation viennent d'un autre flux que celles d'entraînement ; leur disjonction est vérifiée (`ValueError` sinon). `tron evaluate-model` refait cette évaluation pour un modèle et une configuration quelconques.

Résultat de `tron train configs/training/linear_duel.yaml` (300 épisodes en 2 générations, environ 11 s avec l'évaluation, Python pur) :

| Configuration | Référence | Modèle appris | Référence à sa place | Écart |
|---|---|---|---|---|
| learning_duel (entraînement, 10×10) | greedy | 45 % (9/20) | 55 % (11/20) | −10 % |
| learning_duel | maxn, paranoid, social (profondeur 2) | 35 % (7/20) | 50 % (10/20) | −15 % |
| mcts_duel (jamais vue, 12×12) | greedy | 65 % (13/20) | 55 % (11/20) | +10 % |
| mcts_duel | maxn, paranoid, social (profondeur 2) | 50 % (10/20) | 45 % (9/20) | +5 % |

En duel, MaxN, Paranoid et Social (profil `team`) prennent les mêmes décisions (réduction à deux camps), d'où des lignes identiques. Avec 20 parties, l'intervalle de confiance à 95 % d'un taux de victoire fait environ ± 20 points : aucun écart de ce tableau n'est significatif. Les poids appris sont interprétables : `own_territory` ≈ +0,89, `opp_territory` ≈ −0,88, `opp_safety` ≈ −0,43, `own_safety` ≈ +0,25.

**Coût mesuré** : 0,27 ms par `step` (10×10, adversaire greedy, observation complète) ; 300 épisodes d'entraînement en environ 4 s.

## 4. Adaptateur Gymnasium

```python
from tron.learning.gym import make_gym_env   # pip install "tron[learning]"

env = make_gym_env(config, player=0, shaping="territory")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Espaces : `Dict` (plans `Box`, `current` `Discrete(n)`, `legal` `MultiBinary(4)`, …) et `Discrete(4)`. L'adaptateur est testé contre un module Gymnasium simulé (Gymnasium n'est pas installé dans l'environnement de développement) : à vérifier avec une installation réelle.

## 5. Limites

- **Pas de DQN** : l'étape facultative « DQN dans un module dédié » n'est pas faite (aucun cadre de réseaux de neurones disponible) ; l'observation `planes` est prête pour un réseau convolutif. CFR et ses variantes ne sont pas prioritaires (plan, § P10).
- **Politique de profondeur 1** : sur la configuration d'entraînement, elle reste en dessous des agents de recherche de profondeur 2 ; l'utiliser comme heuristique de feuille d'une recherche est une extension naturelle, non faite.
- **Diplomatie** : l'environnement n'envoie pas de contexte de décision aux agents (alliances dynamiques inactives pendant l'entraînement).
- **Agents fournis** : `reset` les réinitialise mais conserve leur générateur aléatoire (comportement historique de `Agent.reset`) ; pour un épisode reproductible, fournir des agents neufs (ce que fait l'entraînement).
