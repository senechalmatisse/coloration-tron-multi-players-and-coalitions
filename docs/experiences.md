# Expériences : créer un plan, lancer, reprendre, analyser

## 1. Créer un plan

Un plan YAML décrit un **scénario** (qui transforme les niveaux des facteurs en configuration de partie), des **facteurs** et le nombre de parties par condition.

```yaml
name: heuristic_comparison
scenario: solo_vs_coalition       # solo_vs_coalition | teams | custom
games: 40                         # parties par condition
seed: 2030                        # graine du plan : graines appariées entre conditions
seats: cyclic                     # chaque graine est jouée avec chaque premier joueur
base: {solo_agent: paranoid, coalition_mode: team, coalition_depth: 2}
factors:
  solo_heuristic: [voronoi, composite]
  solo_depth: [2, 4]
  k: [1, 2]
  grid: [12]
design: full                      # ou {type: subset, n: 24, seed: 17}, ou {type: list, conditions: [...]}
exclude: []                       # combinaisons à retirer
pilot: {games: 4}                 # variante pilote (tron run --pilot)
metrics: true                     # métriques de coopération (P02)
details: true                     # explications de décision dans les journaux (P07)
```

**Graines appariées.** La graine de la partie g ne dépend que de la graine du plan et de g, jamais de la condition : deux conditions de même nombre de joueurs jouent exactement les mêmes placements. Avec `seats: cyclic`, la graine d'indice g div n est jouée par n parties consécutives, chacune avec un premier joueur différent.

### Facteurs reconnus par les scénarios

| Facteur | Effet |
|---|---|
| `k`, `grid`, `composition` | taille de la coalition, grille, composition des équipes (`teams`) |
| `solo_depth`, `coalition_depth`, `depth`, `depth_t<i>`, `depth_unit` | profondeurs (plies ou tours complets) |
| `solo_agent` (alias `agent_solo`), `coalition_mode` (alias `agent_coalition`), `mode`, `mode_t<i>` | algorithmes ; modes de coalition : `team`, `team_paranoid`, `maxn`, `paranoid`, `dynamic_team`, `greedy`, `random`, `mcts`, `adaptive` |
| `simulations`, `exploration`, `rollout_policy`, `rollout_depth` | paramètres des joueurs `mcts` seulement |
| `adaptation_rate`, `observation_window`, `fallback_agent`, `fallback_profile` | paramètres des joueurs `adaptive` seulement |
| `heuristic_name`, `w_territory`, `w_mobility`, `w_connectivity`, `w_safety`, `w_bottleneck` | heuristique de la partie et poids de l'heuristique composite |
| `solo_heuristic` | heuristique propre au solitaire (la coalition garde celle de la partie) |
| `ablate` | `none` ou une composante de l'heuristique composite mise à zéro |
| `diplomacy_enabled`, `proposal_ttl_plies`, `alliance_weight`, `max_allies_per_player`, `betrayal_penalty`, `break_cooldown_plies` | diplomatie tactique (P03) |

Le scénario `custom` accepte une configuration complète (`config`) modifiée par des facteurs à chemin pointé (`players.0.depth`).

### Plans fournis

| Plan | Question | Taille (pilote) |
|---|---|---|
| `etude_a.yaml`, `etude_a_cooperation.yaml`, `etude_b.yaml`, `pilote.yaml` | études historiques (profondeur, coopération, équipes) | - |
| `heuristic_comparison.yaml` | Voronoi ou composite pour le solitaire, mêmes agents, graines et profondeurs | 320 parties (32) |
| `ablation.yaml` | contribution marginale de chaque composante de l'heuristique composite | 480 (48) |
| `mcts_comparison.yaml` | MCTS contre MaxN, Paranoid, Social et Greedy | 800 (40) |
| `mcts_simulations.yaml` | taux de victoire de MCTS selon le nombre de simulations | 400 (40) |
| `dynamic_alliances.yaml` | diplomatie inactive ou active, durée de vie, poids d'alliance, taille de coalition | 480 (32) |
| `adaptive_agents.yaml` | agent fixe ou adaptatif face à chaque profil d'adversaire | 576 (36) |

## 2. Lancer et reprendre

```bash
tron run configs/plans/ablation.yaml --dry-run           # taille du plan
tron run configs/plans/ablation.yaml --pilot --workers 2 # variante pilote (results/ablation_pilot)
tron run configs/plans/ablation.yaml --estimate 10       # budget estimé sur 10 parties
tron run configs/plans/ablation.yaml --workers 8         # campagne complète
tron run configs/plans/ablation.yaml --games 10          # autre effectif par condition
```

Toujours commencer par le pilote : il valide la chaîne et calibre le temps. Ses parties sont un préfixe de la campagne complète (mêmes graines).

`results.csv` est écrit au fil de l'eau ; une relance reprend là où l'exécution s'était arrêtée, sans doublon. Les résultats ne dépendent pas de l'ordonnancement des processus. Une campagne commencée avec une version antérieure se poursuit avec son ancien en-tête (colonnes nouvelles ignorées, avec un avertissement).

`manifest.json` contient le plan complet, les facteurs et leurs niveaux, le nombre de conditions, l'environnement (versions de Python, NumPy, moteur de calcul, hash Git), les versions du code, du format de journal et des métriques, et les **limites méthodologiques** propres aux configurations jouées (par exemple la V1 de la diplomatie, ou l'usage de `time_limit`).

## 3. Analyser

```bash
tron analyze results/ablation/results.csv
```

| Sortie | Contenu |
|---|---|
| `summary.csv` | par condition : effectif, victoires, taux et intervalle de Wilson à 95 %, niveau du hasard, rang moyen, durée, nœuds, temps, simulations ou feuilles, profondeur moyenne par décision du joueur suivi |
| `paired.csv` | comparaisons appariées (mêmes graines, mêmes sièges, mêmes autres facteurs) : McNemar exact et correction de Holm par facteur, plus les écarts moyens de temps et de nœuds par décision avec le test de Wilcoxon |
| `regression.txt`, `regression.csv` | régression logistique de la victoire, erreurs standard groupées par graine |
| `cooperation*.csv`, `diplomacy.csv` | métriques de coopération et de diplomatie (`docs/metriques.md`) |
| figures PNG | courbes et cartes de chaleur historiques, taux de victoire par niveau, courbes de dose |

Comparaisons appariées produites automatiquement :

| Facteur | Comparaison |
|---|---|
| `solo_depth`, `depth_t0` | niveaux consécutifs (historique) |
| `solo_heuristic`, `heuristic_name` | chaque niveau contre `voronoi` |
| `solo_agent` | chaque algorithme contre `paranoid` |
| `ablate` | chaque ablation contre l'heuristique complète (`none`) : la colonne `diff` est la contribution marginale de la composante retirée (négative si elle aide) |
| `diplomacy_enabled` | active contre inactive |
| `simulations`, `adaptation_rate`, `alliance_weight`, `proposal_ttl_plies`, `w_*` | niveaux consécutifs, avec une courbe de taux de victoire |

La régression ajoute aux termes historiques ceux des nouveaux facteurs qui varient : `log2(simulations)`, les facteurs numériques tels quels, `diplomacy_enabled` codé 0/1, et une indicatrice par niveau non de référence des facteurs catégoriels (`solo_heuristic__composite`, `ablate__safety`…), compatible avec le repli NumPy quand statsmodels est absent.

## 4. Limites

- Un pilote (4 parties par condition) sert à valider et calibrer : ses intervalles de confiance sont trop larges pour conclure.
- Les comparaisons appariées n'utilisent que les parties présentes dans les deux niveaux ; avec un plan partiel (`subset`), les paires peuvent manquer.
- La puissance dépend du nombre de paires discordantes : avec 40 parties, seuls de gros écarts sont détectables.
