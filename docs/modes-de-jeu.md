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

Colonnes CSV ajoutées pour tous les modes : `rules_mode`, `individual_winner`, `survival_order`, puis `rounds` (tours complets, comptés seulement si les règles ont un état propre : zones ou obstacles), `team_scores` (contrôle de zone) et `obstacles` (§ 3).

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

**Validation.** Erreur explicite si aucune zone n'est donnée, si `target_score` et `max_rounds` manquent tous les deux, si une zone sort de la grille ou est vide, si `zone_rule` n'est pas `head`, si une valeur n'est pas un entier (booléens, décimaux et chaînes refusés) ou est < 1, ou si ces clés apparaissent dans un autre mode. Les compteurs (tours, scores) ne sont pas bornés.

**Moteur.** Le tour et les scores font partie de l'état : `apply` les met à jour, `undo` les restaure exactement, et la clé Zobrist les inclut (valeurs calculées par `ZobristTables.round_key` et `score_key`, hachage déterministe sans borne ; les clés des parties classiques ne changent pas, ce que vérifie un test de non-régression). Les moteurs de recherche ne contiennent aucune règle de zone : ils lisent `GameState.over` et `Valuation`.

**Heuristique `zone`.** Pour chaque joueur vivant : part de Voronoi (`territory`), proximité moyenne des zones 1 / (1 + distance de Manhattan) (`proximity`) et avance de son équipe au score (`score`). Poids par défaut 0,5 / 0,3 / 0,2, validés et normalisés. Les parts respectent le contrat de `Valuation` (finies, positives, 0 pour un éliminé, somme 1). Hors du mode `zone_control`, l'heuristique vaut Voronoi. Coût mesuré (Python pur, 16×16, 4 joueurs, 60 positions) : environ 121 µs par évaluation, contre 111 µs pour Voronoi.

**Sorties.** Colonnes CSV `rounds` et `team_scores` (`"14;10"`, dans l'ordre des équipes de victoire). Dans le journal, un coup qui termine un tour et change le contrôle d'une zone porte `"zone": [[zone, équipe], ...]` (−1 : zone vide ou disputée). Les sauvegardes (`tron-checkpoint/1`) conservent le tour, les scores, l'historique des points et les contrôleurs du dernier tour : la reprise produit les mêmes événements de zone que la partie continue.

Dans la fenêtre Arcade et les exports d'images (PNG, GIF), les zones sont entourées à l'encre secondaire ; la ligne d'état du panneau ajoute le tour et les scores (`Lecture ×1 · tour 5 · A 3 – B 2`).

Exemple (zones dans les quatre coins, symétriques par quart de tour comme le placement `regular`, pour qu'aucune équipe ne soit avantagée par la géométrie) :

```bash
tron play configs/games/zone_control_16x16.yaml --no-gui --games 12 --out results/zone
```

Sur 12 parties (graine 13) : 7 victoires de A, 5 de B, 11 parties décidées au score cible.

## 3. Obstacles dynamiques (clé `obstacles`)

```yaml
rules:
  mode: classic                # combinable avec tous les modes
  obstacles: {every: 3, count: 4, start: 3, max: 48, margin: 1}
```

| Clé | Défaut | Rôle |
|---|---|---|
| `every` | obligatoire | intervalle, en tours complets, entre deux apparitions |
| `count` | 1 | obstacles posés à chaque apparition |
| `start` | `every` | tour complet de la première apparition |
| `max` | aucun | nombre total maximal d'obstacles |
| `margin` | 1 | distance de Manhattan minimale + 1 aux têtes vivantes (0 : toute case libre) |

**Calendrier.** À la fin des tours complets `start`, `start + every`, `start + 2·every`…, `count` obstacles apparaissent (moins si `max` est atteint ou s'il n'y a plus assez de cases admissibles). Aucun obstacle n'apparaît une fois la partie finie.

**Placement.** Un obstacle n'apparaît que sur une case libre : jamais sur une tête, un mur ou un autre obstacle, ni à distance ≤ `margin` d'une tête vivante. Les cases candidates sont tirées par un hachage déterministe (`tron.core.rng.hash_index`, flux `STREAM_OBSTACLES`) de la graine de la partie, du numéro de tour et d'un compteur ; si la grille est trop encombrée pour que ces tirages suffisent, toutes les cases admissibles sont parcourues dans un ordre haché. La même position donne donc toujours les mêmes obstacles : la recherche les anticipe (règle connue de tous les agents), et une autre graine donne d'autres obstacles.

**Moteur.** Un obstacle est une case de valeur `OBSTACLE` (−3), traitée comme un mur par `legal_moves`, Voronoi, l'heuristique composite et les déroulements MCTS. Les cases posées font partie du jeton d'annulation de la fin de tour : `undo` les libère exactement. La clé Zobrist les inclut (table `obst`, tirée après les tables historiques), de même que `copy`, les instantanés (caractère `/`, cohérence vérifiée avec l'historique et avec `max`) et la carte ASCII (`#`). Les tables de transposition restent exactes : sur 48 positions avec obstacles, Paranoid et MaxN à profondeur 4 jouent les mêmes coups avec et sans table.

**Coût mesuré** (Python pur, 14×14, 4 joueurs, 60 demi-coups aléatoires puis retour) : 2,7 µs par paire `apply` + `undo` en mode classique, 5,0 µs avec obstacles (dont le suivi des tours) ; c'est négligeable devant une évaluation heuristique (≈ 100 µs).

**Sorties.** Dans le journal, le coup qui termine un tour avec apparition porte `"obstacles": [[x, y], ...]` ; au rejeu, ces cases sont recalculées et comparées au journal (`LogError` en cas d'écart). Colonne CSV `obstacles` (total posé). Les obstacles sont gris (couleur `muted` de la palette) dans la fenêtre Arcade et les exports d'images.

Exemple : `tron play configs/games/obstacles_14x14.yaml --no-gui --games 8 --out results/obstacles`.

