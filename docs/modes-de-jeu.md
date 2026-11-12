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
