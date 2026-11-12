# Modes de jeu (P09)

Les variantes de Tron sont décrites par la clé `rules` de la configuration, jamais par des conditions codées dans les agents. Sans clé `rules`, la partie est classique et se déroule exactement comme avant.

L'architecture est dans `tron.core.rules` :

- les **déplacements** sont les mêmes dans tous les modes ;
- les **conditions de victoire** dépendent du mode et de la portée de la victoire ;
- les **objectifs secondaires** (classement par survie, points de zone…) sont décrits par le mode.

`GameState.over` (lu par `is_terminal()` et par les moteurs de recherche) est tenu à jour par `apply` et `undo`. Les **équipes de victoire** (`GameConfig.victory_team_index`) décident de la fin et du vainqueur ; les équipes de la configuration (`team_index`) restent celles des préférences sociales des agents.

## 1. Survie (`survival`)

```yaml
rules: {mode: survival, victory_scope: individual}   # ou team
```

| Portée | Fin de partie | Vainqueur |
|---|---|---|
| `team` | une seule équipe en vie (règle classique) | l'équipe survivante |
| `individual` | un seul joueur en vie, coéquipiers compris | le dernier survivant |

Résultats individuels explicites, ajoutés sans redéfinir le gagnant existant :

- `individual_winner` : le dernier survivant (ou le seul survivant) ;
- `survival_order` : joueurs du plus long au plus court temps de survie ;
- `winning_team` reste le nom d'une équipe de la configuration : en survie individuelle, celle du vainqueur individuel.

En survie individuelle, `Valuation` n'accorde la valeur de victoire qu'au survivant (ses coéquipiers éliminés reçoivent une valeur de défaite) ; les bornes d'élagage utilisent les équipes de victoire.

Exemple : `tron play configs/games/survival_individual_12x12.yaml --no-gui`.

Colonnes CSV ajoutées pour tous les modes : `rules_mode`, `individual_winner`, `survival_order`.

## 2. Contrôle de zone (`zone_control`)

```yaml
rules:
  mode: zone_control
  zones:                       # rectangles {x, y, width, height} ou listes [x, y] / [x, y, w, h]
    - {x: 2, y: 2, width: 3, height: 3}
    - [11, 11, 3, 3]
  zone_rule: head              # seule règle V1
  target_score: 16             # au moins une des deux clés suivantes
  max_rounds: 150
```

**Comptage.** Un *tour* est complet quand chaque joueur vivant a joué une fois (le tour se termine quand l'ordre de jeu revient au premier joueur). À la fin de chaque tour, une zone rapporte **un point** à l'équipe de victoire qui y a, seule, au moins une tête vivante (règle `head`). Une zone vide, ou disputée par des têtes de plusieurs équipes, ne rapporte rien.

**Fin de partie et vainqueur**, dans cet ordre :

1. une équipe atteint `target_score` : l'équipe au plus haut score gagne ;
2. une seule équipe reste en vie : elle gagne, même si elle est derrière au score ;
3. `max_rounds` tours ont été joués : l'équipe au plus haut score gagne.

En cas d'égalité au score, la seule équipe encore en vie parmi les meilleures gagne ; sinon la partie est nulle (`winning_team = -1`, `winners` vide). Les rangs (`ranks`) restent des rangs de survie ; le classement au score se lit dans `team_scores`.

**Validation.** Erreur explicite si aucune zone n'est donnée, si `target_score` et `max_rounds` manquent tous les deux, si une zone sort de la grille ou est vide, si `zone_rule` n'est pas `head`, si un compteur dépasse 255, ou si ces clés apparaissent dans un autre mode.

**Moteur.** Le tour et les scores font partie de l'état : `apply` les met à jour, `undo` les restaure exactement, et la clé Zobrist les inclut (tables `rounds` et `score`, tirées après les tables historiques : les clés des parties classiques ne changent pas). Les moteurs de recherche ne contiennent aucune règle de zone : ils lisent `GameState.over` et `Valuation`.

**Heuristique `zone`.** Pour chaque joueur vivant : part de Voronoi (`territory`), proximité moyenne des zones 1 / (1 + distance de Manhattan) (`proximity`) et avance de son équipe au score (`score`). Poids par défaut 0,5 / 0,3 / 0,2, validés et normalisés. Les parts respectent le contrat de `Valuation` (finies, positives, 0 pour un éliminé, somme 1). Hors du mode `zone_control`, l'heuristique vaut Voronoi. Coût mesuré (Python pur, 16×16, 4 joueurs, 60 positions) : environ 121 µs par évaluation, contre 111 µs pour Voronoi.

**Sorties.** Colonnes CSV `rounds` et `team_scores` (`"14;10"`, dans l'ordre des équipes de victoire). Dans le journal, un coup qui termine un tour et change le contrôle d'une zone porte `"zone": [[zone, équipe], ...]` (−1 : zone vide ou disputée). Les sauvegardes (`tron-checkpoint/1`) conservent le tour, les scores, l'historique des points et les contrôleurs du dernier tour : la reprise produit les mêmes événements de zone que la partie continue.

Dans la fenêtre Arcade et les exports d'images (PNG, GIF), les zones sont entourées à l'encre secondaire ; la ligne d'état du panneau ajoute le tour et les scores (`Lecture ×1 · tour 5 · A 3 – B 2`).

Exemple (zones dans les quatre coins, symétriques par quart de tour comme le placement `regular`, pour qu'aucune équipe ne soit avantagée par la géométrie) :

```bash
tron play configs/games/zone_control_16x16.yaml --no-gui --games 12 --out results/zone
```

Sur 12 parties (graine 13) : 7 victoires de A, 5 de B, 11 parties décidées au score cible.
