# Performances

Toutes les mesures se reproduisent avec `tron bench`. La compilation JIT est exclue des chronométrages : chaque processus préchauffe les noyaux avant de mesurer.

## 1. Heuristiques (P01)

Commande, pour chaque taille de grille (`--grid`), nombre de joueurs (`--players`) et heuristique (`--heuristic`) du tableau, par exemple :

```bash
tron bench --grid 20 --players 4 --heuristic composite --agent maxn --depth 2 --evals 400 --decisions 5
```

Position mesurée : partie aléatoire remplie à 20 % (graine 0). Mesures en **Python pur** (sans Numba, Python 3.13, 2 cœurs), temps moyen d'une évaluation :

| Grille | Joueurs | `voronoi` | `composite` | Rapport |
|---|---:|---:|---:|---:|
| 10 × 10 | 2 | 37 µs | 101 µs | 2,7 |
| 10 × 10 | 4 | 52 µs | 105 µs | 2,0 |
| 10 × 10 | 8 | 46 µs | 114 µs | 2,5 |
| 20 × 20 | 2 | 146 µs | 407 µs | 2,8 |
| 20 × 20 | 4 | 159 µs | 421 µs | 2,6 |
| 20 × 20 | 8 | 180 µs | 436 µs | 2,4 |
| 40 × 40 | 2 | 809 µs | 2 065 µs | 2,6 |
| 40 × 40 | 4 | 818 µs | 2 148 µs | 2,6 |
| 40 × 40 | 8 | 949 µs | 2 096 µs | 2,2 |

L'heuristique composite coûte environ 2,5 fois Voronoi : elle refait le parcours de Voronoi puis un parcours en profondeur de toutes les cases libres (composantes et points d'articulation). Le coût est linéaire en nombre de cases et presque indépendant du nombre de joueurs, car le parcours est partagé.

Avec Numba, les deux noyaux sont compilés (le noyau composite retombe en Python pur avec un avertissement si la version de Numba installée refuse de le compiler). Les mesures JIT sont à produire avec la même commande sur une machine où Numba est installé.

## 2. Suite reproductible par composant (P13)

```bash
tron bench --suite                                    # 20×20, 4 joueurs, 5 positions par phase
tron bench --suite --format json --out mesures.json   # résultat tron-bench/1
tron bench --suite --reference benchmarks/reference/suite-python-20x20-4p.json
tron bench --suite --quick                            # 10×10, 2 joueurs, quelques dixièmes de seconde
```

**Positions.** `tron.benchmarks.position_set(grid, players, per_phase, seed)` produit un jeu de positions déterministe en quatre phases : ouverture (2–8 % de cases occupées, tous vivants), milieu de partie (35–50 % de la durée de la partie, joueurs en contact), grille cloisonnée (régions accessibles disjointes) et fin de partie (2·n demi-coups avant la fin). Les phases sont définies par rapport à des parties jouées jusqu'au bout, donc valables pour toutes les tailles de grille ; l'empreinte SHA-256 des positions figure dans chaque résultat.

**Mesures séparées**, par phase : génération des coups (`legal_moves`), `apply` + `undo`, Voronoi (`counts`), caractéristiques composites, décisions de recherche (Paranoid profondeur 3 par défaut : temps par décision et nœuds par seconde), simulations MCTS (200 par décision) ; puis écriture de lignes de résultats CSV et rendu d'image (`render_frame`, sans Arcade). Les nombres de positions, de répétitions et les graines sont fixes et enregistrés dans le résultat (`params`) ; les noyaux JIT sont préchauffés avant tout chronométrage, et le noyau utilisé (`environment.backend`) est enregistré : les résultats Python pur et JIT sont séparés.

**Sommes de contrôle.** Chaque mesure porte une somme de contrôle de ce qu'elle calcule (comptes de Voronoi, caractéristiques, coups choisis et nœuds explorés, images…). `--reference` affiche, mesure par mesure, le rapport des temps et signale `DIFFÉRENT` si un calcul a changé, et un avertissement si les paramètres ou les positions diffèrent. `tests/test_performance.py` vérifie à chaque exécution de la suite de tests que la suite réduite redonne exactement les sommes de `benchmarks/reference/suite-python-quick.json` : toute optimisation qui changerait un résultat est détectée, sans aucun seuil de temps.

Référence Python pur (`benchmarks/reference/suite-python-20x20-4p.json`, Python 3.13, 2 cœurs) :

| Mesure | Ouverture | Milieu | Cloisonnée | Fin |
|---|---:|---:|---:|---:|
| `legal_moves` (µs/appel) | 0,2 | 0,2 | 0,2 | 0,2 |
| `apply` + `undo` (µs) | 0,7 | 0,7 | 0,7 | 0,7 |
| Voronoi (µs/évaluation) | 201 | 178 | 68 | 75 |
| composite (µs) | 520 | 471 | 285 | 305 |
| Paranoid profondeur 3 (ms/décision) | 4,4 | 3,9 | 0,7 | 1,4 |
| MCTS (µs/simulation) | 301 | 217 | 14 | 23 |

Écriture des résultats : 17,5 µs par ligne CSV ; rendu d'une image 160×160 : 152 µs. Les mesures JIT sont à produire avec `tron bench --suite --out benchmarks/reference/suite-jit-20x20-4p.json` sur une machine où Numba est installé (non disponible dans l'environnement de développement de cette version).

## 3. Profilage

```bash
tron bench --suite --profile --top 30                 # profil cProfile, texte pstats
tron bench --suite --profile --format json            # fonctions en JSON (ncalls, tottime, cumtime)
tron bench --profile --profile-out decisions.prof     # mesure historique, fichier pstats
python -m pstats decisions.prof                       # ou : snakeviz decisions.prof
py-spy record -o profil.svg -- python -m tron bench --suite   # si py-spy est installé
```

`--profile` s'applique à la suite (`--suite`) ou à la mesure historique ; aucun outil externe n'est requis (`cProfile` et `pstats` font partie de Python). Sous profilage, les temps mesurés sont gonflés : ils servent à voir où part le temps, pas à comparer des versions.

## 4. Campagnes multiprocessus

```bash
tron bench --campaign configs/plans/pilote.yaml --workers 1 2 --limit 48 --out campagne.json
```

Chaque campagne (sans reprise, `--limit` parties) tourne dans un processus intermédiaire neuf qui mesure par `getrusage` le temps CPU de tous ses descendants et la mémoire résidente maximale du plus gros d'entre eux. Référence (`benchmarks/reference/campaign-pilote-python.json`, 2 cœurs) :

| Processus | Parties | Durée | CPU | Parties/s | Mémoire max |
|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 1,08 s | 1,04 s | 44,6 | 43,8 Mo |
| 2 | 48 | 0,68 s | 1,10 s | 70,9 | 43,3 Mo |

## 5. Tests de charge

Marqués `load` et exclus de la suite par défaut (`addopts` de `pyproject.toml`) ; aucun seuil de temps :

```bash
pytest -m load
```

Ils vérifient sur de grandes instances que les agents rendent l'état intact et jouent un coup légal (Paranoid profondeur 3 sur 40×40 à 8 joueurs et profondeur 4 sur 60×60 à 4 joueurs, MCTS à 3 000 simulations sur 30×30), jouent une mêlée complète à 8 joueurs sur 30×30 et une partie 50×50 à 8 joueurs, et mesurent une campagne à 1 et 2 processus (environ 20 s en Python pur).
