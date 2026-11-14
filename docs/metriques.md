# Métriques de coopération, de cohésion et de stabilité

Le gagnant, les rangs, le nombre de coups et le coût de recherche ne disent pas si une coalition coopère réellement. Chaque partie produit donc aussi des **métriques de partie**, calculées à partir des événements observables (coups, éliminations), du plateau et des équipes statiques. Elles ne modifient ni les règles ni les décisions : le collecteur (`tron.sim.metrics.MetricsCollector`) utilise ses propres instances d'heuristique.

Elles sont déterministes à configuration et graine identiques, et recalculables après coup depuis un journal JSONL (`GameLog.metrics()`), y compris un journal antérieur à leur introduction.

## 1. Définitions

Notations : t est une équipe statique, ses joueurs sont ses membres ; un demi-coup (ply) est le coup d'un joueur.

| Métrique | Définition opérationnelle | Unité, domaine | Non calculable |
|---|---|---|---|
| `survival_plies[p]` | demi-coups après lesquels p est encore vivant (indice du demi-coup de son élimination, ou durée totale s'il survit) | demi-coups, ≥ 0 | - |
| `team_survival_plies[t]` | max des `survival_plies` des membres : demi-coups pendant lesquels au moins un membre vit | demi-coups | - |
| `team_rank[t]` | meilleur rang final d'un membre (1 = équipe gagnante) | ≥ 1 | - |
| `intra_team_distance_mean[t]` | moyenne, sur les états observés (initial puis après chaque demi-coup) où au moins deux membres vivent, de la distance de Manhattan moyenne entre têtes de membres vivants | cases | `None` si jamais deux membres vivants ensemble |
| `ally_collisions[t]` | éliminations de membres dont la cause est **sans ambiguïté** un mur de coéquipier : collision directe, ou enfermement dont toutes les cases bloquantes (hors bordure et murs du joueur) sont à des coéquipiers | nombre | - |
| `blocking_moves[t]` | déplacements d'un membre qui l'amènent à côté de la tête d'un coéquipier vivant (le coéquipier perd une case libre voisine) | nombre | - |
| `team_moves[t]` | déplacements (hors éliminations) des membres | nombre | - |
| `cooperation_proxy[t]` | ½·(1 − distance / (W + H − 2)) + ½·(1 − blocking_moves / team_moves) | [0, 1] | `None` si la distance l'est |
| `territory_share_mean[t]` | moyenne de la somme des parts de Voronoi des membres aux points d'observation (état initial puis tous les k demi-coups, k = nombre de joueurs) | [0, 1] | `None` sans observation |
| `collective_vs_individual_gap[t]` | 1 − moyenne des `survival_plies` des membres / `team_survival_plies[t]` | [0, 1[ | `None` pour une équipe d'un joueur |

L'abandon d'un joueur qui avait encore un coup légal (`PASS` volontaire) n'est jamais attribué à un coéquipier : sa cause est indéterminée.

## 2. Limites des indicateurs

- `cooperation_proxy` est un **proxy de comportement observable** (rester proches sans se gêner), pas une mesure d'intention ou de « confiance ». Deux agents individualistes peuvent obtenir un bon score s'ils restent proches par hasard, et une coalition qui se répartit volontairement le plateau peut obtenir un score médiocre.
- `intra_team_distance_mean` dépend de la taille de la grille ; seule la version normalisée entre dans le proxy.
- `ally_collisions` sous-estime les gênes entre alliés : les enfermements de cause mixte (alliés et adversaires) ne sont pas comptés.
- `territory_share_mean` utilise Voronoi (départage `order`, ou `neutral` si la configuration l'indique), quelle que soit l'heuristique des agents.
- **Aucune métrique de « trahison »** n'est déduite d'un déplacement. Les ruptures d'alliance ne sont mesurables qu'avec la diplomatie explicite (voir `docs/coalitions-dynamiques.md`).

## 3. Colonnes du fichier de résultats

`MatchResult.to_row()` ajoute, après les colonnes historiques (inchangées) :

| Colonne | Contenu |
|---|---|
| `metrics_version` | version du schéma (1) ; vide si les métriques sont désactivées |
| `survival_p0` … `survival_p7` | survie de chaque joueur |
| `team_t0` … `team_t7` | nom de chaque équipe (`team_t0` est l'équipe du joueur 0, donc l'équipe suivie) |
| `tsurv_t<i>`, `trank_t<i>`, `tdist_t<i>`, `tcoll_t<i>`, `tcoop_t<i>`, `tterr_t<i>`, `tgap_t<i>` | métriques de l'équipe i (voir `tron.schema.TEAM_METRIC_COLUMNS`) |

Le jeu de colonnes est toujours le même, que les métriques soient actives ou non (`metrics: false` dans un plan laisse les cellules vides) : un fichier de résultats garde un en-tête unique.

**Compatibilité.** Un ancien `results.csv` reste analysable (les tableaux de coopération sont simplement absents). La reprise d'une campagne commencée avec une version antérieure continue avec l'ancien en-tête : les nouvelles colonnes sont ignorées pour cette campagne, avec un avertissement.

## 4. Analyse

```bash
tron run configs/plans/pilote.yaml --workers 2
tron analyze results/pilote/results.csv
```

En plus des sorties historiques, `tron analyze` écrit quand les métriques sont présentes :

- `cooperation.csv` : une ligne par (condition, équipe) avec l'effectif, la taille d'équipe, le taux de victoire et son intervalle de Wilson, la moyenne de chaque métrique et le total des collisions entre alliés ;
- `cooperation_conditions.csv` : une ligne par condition, sur les équipes d'au moins deux joueurs ;
- `cooperation_proxy_<facteur>.png` et `team_survival_<facteur>.png` : évolution de l'indice de coopération et de la survie des équipes selon `k`, `coalition_depth` ou `grid` (premier facteur qui varie).

Les fonctions correspondantes (`team_table`, `cooperation_summary`, `condition_summary`) sont dans `tron.analysis.cooperation` et ne dépendent ni du moteur, ni de Numba, ni d'Arcade.

Pour une campagne massive où seul le vainqueur compte, `metrics: false` dans le plan supprime le coût d'observation (une évaluation de Voronoi par tour complet).
