# Heuristiques d'évaluation

`tron heuristics list` affiche les heuristiques disponibles (intégrées et extensions) avec leurs paramètres ; l'heuristique `zone` du contrôle de zone est décrite dans `docs/modes-de-jeu.md`.

Une heuristique évalue un état **non terminal** par un vecteur de parts h, une composante par joueur. Les états terminaux sont traités à part par `Valuation` (voir `docs/algorithmes.md`, §1).

## 1. Contrat `shares`

Toute heuristique hérite de `tron.eval.base.Heuristic` et implémente `shares(state) -> list[float]` :

| Propriété | Exigence |
|---|---|
| longueur | `n_players` |
| valeurs | finies et ≥ 0 |
| joueurs éliminés | exactement 0 |
| somme | 1 sur les joueurs vivants (à l'arrondi près) |
| joueur vivant | part strictement positive (sinon une défaite pourrait valoir autant qu'une survie) |
| état | `state` n'est pas modifié |

Ce contrat est celui que supposent les élagages : positivité (élagage superficiel de Korf), somme bornée (borne de somme et borne généralisée sur le simplexe), ordre victoire > heuristique > défaite.

Enregistrement et usage depuis le YAML :

```python
from tron.eval.base import Heuristic, register_heuristic

@register_heuristic("ma_heuristique")
class MaHeuristique(Heuristic):
    def __init__(self, width, height, n_players, *, poids: float = 1.0) -> None:
        super().__init__(width, height, n_players)
        self.poids = poids

    def shares(self, state):
        ...
```

```yaml
heuristic: {name: ma_heuristique, poids: 0.5}
```

Les clés autres que `name` sont passées au constructeur ; un paramètre inconnu lève une erreur explicite (`paramètres invalides pour l'heuristique …`).

## 2. Heuristiques fournies

| Nom | Principe | Paramètres |
|---|---|---|
| `uniform` | parts égales entre joueurs vivants (référence des tests de recherche) | – |
| `voronoi` | territoire de Voronoi (parcours en largeur multi-sources) | `tie` (`order`, `neutral`), `separation` |
| `composite` | territoire, mobilité, sorties, marge, contrôle des passages | les cinq poids, `tie`, `separation` |

### 2.1 Voronoi

Voir `docs/algorithmes.md`, §6. Le noyau est compatible Numba ; ses tableaux de travail (distances, masques des joueurs à distance minimale, marques) sont exposés par `VoronoiHeuristic.last_search()` pour être réutilisés sans refaire le parcours.

### 2.2 Composite (P01)

Pour chaque joueur vivant p, cinq caractéristiques bornées dans [0, 1] :

| Composante | Définition | Normalisation |
|---|---|---|
| `territory` | part de Voronoi (tête comprise) | déjà dans ]0, 1] |
| `mobility` | cases libres voisines de la tête | ÷ 4 |
| `connectivity` | composantes libres distinctes voisines de la tête (sorties vers des régions séparées) | min(c, 2) / 2 ∈ {0, ½, 1} |
| `safety` | taille de la plus grande composante libre voisine de la tête (marge avant enfermement) | ÷ nombre total de cases libres |
| `bottleneck` | points d'articulation du graphe des cases libres que p atteint le premier (Voronoi, même départage) | ÷ nombre de points d'articulation |

Un point d'articulation est une case libre dont la suppression déconnecte sa composante : contrôler un tel passage permet de couper l'accès d'un adversaire à une partie du plateau.

Score et parts :

```text
raw[p] = max(plancher, Σ_k w_k · x_k[p])        plancher = 1 / (4·W·H)
h[p]   = raw[p] / Σ_{q vivant} raw[q]           (0 pour un joueur éliminé)
```

Règles sur les poids :

- poids finis, positifs ou nuls, de somme strictement positive ; sinon `ValueError` (NaN, ±∞, négatif, non numérique, booléen, composante inconnue, tous nuls) ;
- **normalisation automatique** : les poids sont divisés par leur somme (multiplier tous les poids par une constante ne change rien) ;
- un poids absent prend sa valeur par défaut.

Valeurs par défaut :

```yaml
heuristic:
  name: composite
  territory: 0.55
  mobility: 0.15
  connectivity: 0.10
  safety: 0.15
  bottleneck: 0.05
  tie: order          # départage des cases à égale distance (comme voronoi)
```

Le plancher garantit une part strictement positive à un joueur vivant même si le poids du territoire est nul (ablation) : la plus petite part, au moins 1/(4·W·H·n), reste supérieure à ε, la valeur maximale d'une défaite dans `Valuation`. Les bornes d'élagage restent valides puisque Σh = 1 et h ≥ 0.

Exemple exécutable : `tron play configs/games/composite_20x20.yaml --no-gui`.

**Coût.** Un parcours en largeur (Voronoi) et un parcours en profondeur itératif de Tarjan (composantes et points d'articulation) par état, partagés par tous les joueurs. Le noyau est écrit dans le style Numba et décoré par `safe_njit` : si Numba refuse de le compiler, il s'exécute en Python pur avec un avertissement. Mesures dans `docs/performance.md`.

**Validation.** Les comptes du noyau sont comparés à une implémentation par force brute (composantes par parcours en largeur, points d'articulation par suppression de chaque case) dans les deux modes de départage ; les parts vérifient le contrat ci-dessus ; une position symétrique (miroir ou rotation, départage `neutral`) donne des parts symétriques ; alpha-bêta et MaxN restent exacts avec cette heuristique (`tests/test_composite_heuristic.py`).

**Limites.** Les caractéristiques sont des indicateurs locaux : `connectivity` et `safety` ignorent qui d'autre partage la composante, et `bottleneck` compte les points d'articulation sans pondérer la taille de la région qu'ils séparent.

### 2.3 Caractéristiques détaillées

`CompositeHeuristic.features(state)` renvoie un `CompositeFeatures` (une valeur par joueur et par composante, plus le nombre de cases libres, de points d'articulation et de composantes) ; `raw_scores(state)` renvoie les scores bruts. Ces données servent aux explications de décision et aux modèles d'adversaires.
