# Performances

Toutes les mesures se reproduisent avec `tron bench`. La compilation JIT est exclue des chronométrages : chaque processus préchauffe les noyaux avant de mesurer.

## 1. Heuristiques (P01)

Commande, pour chaque taille de grille G, nombre de joueurs N et heuristique H :

```bash
tron bench --grid G --players N --heuristic H --agent maxn --depth 2 --evals 400 --decisions 5
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
