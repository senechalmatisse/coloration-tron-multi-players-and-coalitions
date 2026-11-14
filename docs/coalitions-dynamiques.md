# Coalitions dynamiques : diplomatie tactique

Les équipes de la configuration sont **permanentes** : elles déterminent la victoire, les rangs, la fin de partie et le hachage de Zobrist. La diplomatie ajoute des **alliances tactiques souples** qui modifient seulement les préférences sociales (matrice W) des agents qui les exploitent. Le résultat reste attribué à l'équipe permanente définie en YAML.

Une version future pourra introduire des équipes dynamiques comme règle de jeu distincte ; ce n'est pas le cas ici.

## 1. Configuration

```yaml
diplomacy:
  enabled: true                # un bloc diplomacy sans enabled est actif ; absent = inactif
  proposal_ttl_plies: 8        # durée de vie d'une proposition (demi-coups), >= 1
  max_allies_per_player: 2     # plafond d'alliés simultanés, >= 1
  alliance_weight: 0.65        # poids minimal entre alliés dans W_t, dans [0, 1]
  betrayal_penalty: 0.15       # baisse de W[p][q] par trahison de q envers p, dans [0, 1] (0 par défaut)
  break_cooldown_plies: 4      # délai avant une nouvelle proposition entre ex-alliés, >= 0
  free_action: true            # hypothèse V1 : proposer ou rompre ne coûte pas de demi-coup
```

`free_action: true` est la seule valeur acceptée : elle rend explicite, dans la configuration et donc dans l'en-tête des journaux et le manifeste des campagnes, l'hypothèse selon laquelle une action diplomatique est gratuite (en Tron, renoncer à un déplacement élimine le joueur).

Exemple exécutable : `tron play configs/games/diplomacy_4p_20x20.yaml --no-gui --out results/dipl`.

## 2. Protocole

| Règle | Détail |
|---|---|
| intention | au plus une par décision, formulée à la fin de celle-ci (`Decision.diplomacy`) : `propose` ou `break` vers une cible |
| proposition | reste en attente `proposal_ttl_plies` demi-coups (faite au demi-coup r, elle expire au demi-coup r + TTL) |
| alliance | formée quand deux joueurs se sont proposé mutuellement avant expiration ; symétrique |
| rupture | unilatérale, à son tour ; la paire ne peut plus se proposer d'alliance pendant `break_cooldown_plies` demi-coups |
| élimination | le joueur perd ses alliances (`dissolved`) et ses propositions (`expired`) ; il ne peut plus agir |
| refus | une intention invalide (soi-même, joueur inconnu ou éliminé, doublon, plafond, délai) est tracée `rejected` sans interrompre la partie |
| victoire | inchangée : une alliance n'empêche jamais de gagner contre un allié |

**Trahison (règle explicite).** La rupture par q de son alliance avec p est une trahison si p n'avait jamais rompu d'alliance avec q auparavant : q fait défection le premier dans cette relation. La confiance de p envers q vaut 1 / (1 + trahisons de q envers p).

Ordre dans un demi-coup : décision de l'agent (avec un instantané de l'état diplomatique), application de l'intention au demi-coup courant, application du coup, puis nettoyage (éliminations et expirations au demi-coup suivant).

## 3. Agents

- `Agent.choose_move(state, player=None, context=None)` reçoit un `DecisionContext` (demi-coup et `DiplomacySnapshot`). Les agents historiques l'ignorent ; un agent externe dont `choose_move` n'accepte pas `context` est appelé sans contexte.
- Crochets : `_decide_with_context(state, moves, context)` (par défaut `_decide`) et `_diplomacy(state, move, context)` (par défaut aucune intention).
- `SocialAgent` exploite la diplomatie avec le profil `dynamic_team` (ou le paramètre `diplomacy: true`) quand la diplomatie de la partie est active. Sinon il garde la matrice statique historique.

**Matrice dynamique W_t** (`tron.agents.social.dynamic_matrix`), dans cet ordre :

1. W_t = W_base (profil `team` pour `dynamic_team`, paramètre `alpha`) ;
2. pour chaque alliance active (a, b) : W_t[a][b] = max(W_base[a][b], alliance_weight), et symétriquement ;
3. si `betrayal_penalty` > 0 : W_t[p][q] −= betrayal_penalty × trahisons de q envers p ;
4. bornage de chaque poids dans [−1, 1].

Le statut d'alliance est symétrique ; les préférences peuvent rester asymétriques (pénalités). Le moteur est choisi comme avant (`auto`, `alphabeta`, `maxn`) à partir de W_t, avec la réduction à deux camps.

**Politique diplomatique** (`tactical_intent`, déterministe, « contre le meneur ») : l'équipe meneuse est celle dont la somme des parts de territoire des membres vivants est la plus grande. Dans l'ordre : rompre avec un allié de l'équipe meneuse si elle devance l'équipe du joueur de plus de `break_margin` (0,15 par défaut) ; accepter une proposition en attente d'un joueur hors de l'équipe meneuse ; si l'équipe du joueur ne mène pas, proposer une alliance au joueur le plus fort hors de l'équipe meneuse et de sa propre équipe. Les coéquipiers permanents ne sont jamais candidats.

## 4. Journal, résultats et métriques

- Journal `tron-log/2` : l'en-tête contient la configuration diplomatique complète ; une ligne `move` porte, si besoin, `diplomacy: [{ply, player, target, kind}, …]`. Les journaux `tron-log/1` restent lisibles ; un format inconnu est refusé explicitement.
- `GameLog.diplomacy_events()` relit les événements ; `GameLog.metrics()` recalcule les métriques, diplomatie comprise, avec les mêmes valeurs qu'en direct.
- Colonnes CSV (`metrics_version` 2) : `dipl_enabled`, `dipl_proposals`, `dipl_formed`, `dipl_breaks`, `dipl_betrayals`, `dipl_rejected`, `dipl_duration` (durée moyenne des alliances en demi-coups ; une alliance encore active dure jusqu'à la fin).
- `tron analyze` écrit `diplomacy.csv` (moyennes par condition) quand des parties ont joué avec la diplomatie.

## 5. Limites de la V1

- **W_t est constante pendant une recherche** : les agents ne simulent pas les propositions ou ruptures futures dans leur arbre. Les alliances évoluent seulement entre les décisions réelles. Cette limite est aussi inscrite dans le manifeste des campagnes (configuration complète des parties).
- La politique diplomatique est volontairement simple et lisible ; elle sert de référence aux expériences, pas d'optimum.
- La gratuité des actions diplomatiques (`free_action`) est une hypothèse de modélisation.
- Les métriques de trahison reposent sur la règle explicite ci-dessus, jamais sur l'interprétation d'un déplacement.
