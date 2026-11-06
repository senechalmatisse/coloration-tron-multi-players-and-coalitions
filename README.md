# Tron multi-joueur et coalitions

Simulateur de Tron sans joueur humain (2 à 8 joueurs, grille de 5 × 5 à 100 × 100, tour par tour) avec :

- des agents de recherche : aléatoire, glouton, Negamax, Paranoid, MaxN ;
- une extension aux équipes inspirée de SOS (*Socially Oriented Search*) ;
- une visualisation arcade en temps réel et en rejeu ;
- une campagne d'expériences reproductible et parallèle ;
- l'analyse statistique des résultats.

<p align="center">
  <img src="./assets/demo.gif" alt="Démonstration partie" width="700">
</p>

La pile cible est **Python 3.14** et **arcade 3.3.3**. NumPy et Numba servent au calcul, matplotlib et imageio aux figures et aux exports, pandas, SciPy et statsmodels à l'analyse.

## Installation

Une seule commande, au choix :

```bash
uv sync --extra dev                     # crée .venv avec Python 3.14 (fichier .python-version)
# ou
python3.14 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Numba compile le parcours en largeur au premier appel, puis le met en cache dans `__pycache__`. Pour déboguer en Python pur : `NUMBA_DISABLE_JIT=1`. Le code fonctionne
aussi sans Numba (même résultat, plus lentement).

## Démarrage rapide

```bash
(source .venv/bin/activate)

tron play configs/games/solo_vs_3_20x20.yaml                                        # fenêtre arcade, temps réel
tron play configs/games/ffa4_20x20.yaml --no-gui --games 10 --out results/ffa4
tron replay results/ffa4/game_0000.jsonl                                            # rejeu (avant / arrière)
tron replay results/ffa4/game_0000.jsonl --gif partie.gif --voronoi
tron bench --grid 20 --players 4 --agent paranoid --depth 10                        # performances
tron run configs/plans/pilote.yaml --workers 8                                      # campagne (reprise automatique)
tron analyze results/pilote/results.csv                                             # tableaux, tests, figures
```

Touches de la fenêtre :

| Touche | Action |
|---|---|
| `Espace` | pause / lecture |
| `→` / `←` | demi-coup suivant / précédent (le retour arrière n'existe qu'en rejeu) |
| `↑` / `↓` | vitesse |
| `V` | surcouche des territoires de Voronoi |
| `N` | nouvelle partie (temps réel) ou retour au début (rejeu) |
| `Début` / `Fin` | début / fin du rejeu |
| `S` | capture d'écran |
| `Échap` | quitter |

## Configuration d'une partie

```yaml
name: solo_vs_3
grid: 20                                                                                # ou [largeur, hauteur], ou {width, height}
seed: 3                                                                                 # tout l'aléa dérive de cette graine
first_player: 0
placement: {mode: regular, radius: 0.35, rotate: true}                                  # regular | random | manual
heuristic: {name: voronoi, tie: order, separation: false}
players:
  - {name: solo, team: solo, agent: paranoid, depth: 8}
  - {name: c1, team: coalition, agent: social, depth: 3, profile: team, alpha: 1.0}
  - {name: c2, team: coalition, agent: social, depth: 3, profile: team, alpha: 1.0}
  - {name: c3, team: coalition, agent: social, depth: 3, profile: team, alpha: 1.0}
```

Un joueur sans `team` forme sa propre équipe. Les clés autres que `name`, `team`, `agent` et `depth` sont transmises à l'agent. La clé `defaults` s'applique à tous les joueurs.

| Agent | Algorithme | Paramètres principaux |
|---|---|---|
| `random` | coup sûr tiré au hasard | – |
| `greedy` | profondeur 1, maximise sa part de territoire | `tie_break` |
| `negamax` | negamax alpha-bêta, deux joueurs (référence) | `depth` |
| `paranoid` | alpha-bêta deux camps : tous contre la racine | `depth`, `rounds`, `ordering`, `iterative_deepening`, `tt`, `time_limit` |
| `maxn` | MaxN avec élagage superficiel de Korf | `depth`, `pruning` |
| `social` | matrice de préférences W (SOS) | `profile` (`maxn`, `paranoid`, `team`, `team_paranoid`, `custom`), `alpha`, `matrix`, `engine` (`auto`, `alphabeta`, `maxn`) |

Paramètres communs : `depth` (plies) ou `rounds` (tours complets : rounds × joueurs vivants), et `tie_break` (`random`, tiré avec la graine, ou `fixed`, ordre N, E, S, O).

## Plans d'expériences

```yaml
name: etude_a
scenario: solo_vs_coalition                                                 # solo_vs_coalition | teams | custom
games: 100                                                                  # parties par condition
seed: 2026                                                                  # graines appariées entre conditions
seats: cyclic                                                               # chaque graine est jouée avec chaque premier joueur
base: {coalition_mode: team, alpha: 1.0, depth_unit: plies}
factors:
  solo_depth: [1, 2, 4, 6, 8, 10]
  coalition_depth: [1, 2, 3, 4]
  k: [1, 2, 3, 4, 5]
  grid: [10, 20, 40]
design: full                                                                # ou {type: subset, n: 60, seed: 1}, ou {type: list, ...}
key_conditions: {games: 200, where: [{grid: 20, coalition_depth: 2}]}
```

Fichiers fournis dans `configs/plans/` :

- `etude_a.yaml` : 360 conditions, soit 39 000 parties ;
- `etude_a_cooperation.yaml` : coordination de la coalition et algorithme du solitaire ;
- `etude_b.yaml` : équipes contre équipes ;
- `pilote.yaml` : 1 152 parties, environ une minute.

`tron run` écrit `results.csv` (une ligne par partie, avec la graine, le hash Git et les versions) et `manifest.json`. Une relance reprend là où l'exécution s'était arrêtée.
Options utiles : `--estimate N` (budget de calcul estimé sur N parties), `--limit`, `--logs` (journaux JSONL de chaque partie), `--dry-run`.

## Analyse

`tron analyze results.csv` produit :

- `summary.csv` : taux de victoire, intervalle de Wilson à 95 %, niveau du hasard |équipe|/n, rang moyen, durée, nœuds et temps par décision ;
- `paired.csv` : profondeurs consécutives comparées sur les mêmes parties (test exact de McNemar, correction de Holm) ;
- `regression.txt` : régression logistique de la victoire, erreurs standard groupées par ; statsmodels si disponible, sinon repli NumPy ;
- les figures : taux de victoire selon la profondeur (une courbe par k, un panneau par grille), cartes de chaleur des deux profondeurs, coût de calcul.

Le notebook `notebooks/analyse.ipynb` reprend les mêmes étapes de façon interactive.

## Architecture

```
src/tron/
  core/         configuration, état (apply / undo sans copie, Zobrist), placement, graines
  eval/         Voronoi (BFS compatible Numba), valeur des feuilles
  agents/       moteurs de recherche, agents, préférences sociales (SOS)
  sim/          Match + observateurs, journal JSONL, rejeu, arène
  experiments/  plans YAML, scénarios, exécution parallèle avec reprise, CSV
  viz/          palette, rendu NumPy, contrôle de lecture, exports, fenêtre arcade
  analysis/     statistiques, figures, régression
  cli/          commande tron
```

Chaque couche ne dépend que de celles du dessous : `cli` → `viz`, `experiments`, `analysis` → `sim` → `agents` → `eval` → `core`. Le moteur fonctionne sans affichage.

Documents détaillés :

- [`docs/architecture.md`](docs/architecture.md) : schémas d'architecture et de flux ;
- [`docs/algorithmes.md`](docs/algorithmes.md) : codage des valeurs, moteurs, élagages, réduction à deux camps, confrontation avec SOS ;
- [`docs/exigences.md`](docs/exigences.md) : traçabilité entre exigences, code et tests.

## Qualité

```bash
pytest                                                      # 82 tests (jalons J1 à J5 compris)
pytest --cov=tron                                           # couverture (cible : 80 % sur core, eval, agents)
ruff check src tests && ruff format --check src tests
mypy src
NUMBA_DISABLE_JIT=1 pytest                                  # mêmes tests en Python pur
```

Tests de validation des algorithmes :

- l'alpha-bêta donne la même valeur racine et les mêmes coups ex aequo qu'un minimax sans élagage ;
- MaxN donne le même résultat avec et sans élagage superficiel ;
- à deux joueurs, Negamax, Paranoid, MaxN et SOS sont d'accord ;
- `SocialAgent` reproduit Paranoid (profil Paranoid) et MaxN (profil Équipe, α = 0) ;
- la borne d'élagage généralisée est vérifiée contre `scipy.optimize.linprog`.

## Reproductibilité

À configuration et graine identiques, une partie est identique coup pour coup, quel que soit l'ordonnancement des processus.
Les temps de compilation JIT sont exclus des mesures (chaque processus préchauffe les noyaux avant de jouer).

Une limite : la limite de temps de réflexion optionnelle (`time_limit`) rend les parties dépendantes de la machine. Elle est désactivée par défaut et ne doit pas servir aux campagnes.
