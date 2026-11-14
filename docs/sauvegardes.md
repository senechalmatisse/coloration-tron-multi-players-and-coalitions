# Sauvegarde et reprise exacte d'une partie

Un journal JSONL permet de **rejouer** une partie coup par coup. Une sauvegarde permet de la **reprendre** : l'état courant et la mémoire des agents sont restaurés sans rejouer les décisions, et la suite de la partie est identique à celle d'une partie jouée d'un seul tenant. Les deux formats sont distincts.

## 1. Commandes

```bash
# jouer jusqu'au demi-coup 40 puis sauvegarder
tron play configs/games/diplomacy_4p_20x20.yaml --no-gui --checkpoint c1.json --stop-after 40
# reprendre jusqu'au demi-coup 90 et sauvegarder à nouveau
tron resume-checkpoint c1.json --stop-after 90 --checkpoint c2.json
# reprendre jusqu'à la fin en écrivant un journal complet (rejouable dès le premier coup)
tron resume-checkpoint c2.json --log partie.jsonl
tron replay partie.jsonl
```

En Python :

```python
from tron.sim import Match
match = Match(config)
for _ in range(40):
    match.step()
match.save_checkpoint("c1.json")
resumed = Match.from_checkpoint("c1.json")   # observers=… facultatif
result = resumed.run()
```

## 2. Contenu (`tron-checkpoint/1`)

| Champ | Contenu |
|---|---|
| `format`, `code_version`, `git_hash` | version du format, du code et du dépôt |
| `config` | configuration complète de la partie |
| `state` | grille (un caractère par case, bordure comprise), têtes, joueurs vivants, demi-coups d'élimination, joueur au trait, demi-coup, coups, ordre d'élimination, compteurs d'équipes, positions initiales, historique `apply`/`undo`, clé de Zobrist |
| `agents` | par joueur : nom de l'agent et état interne (`Agent.snapshot_state()`) |
| `diplomacy` | propositions, alliances, délais, ruptures, trahisons, historique d'événements |
| `metrics` | accumulateurs des métriques de partie |
| `counters`, `started` | statistiques cumulées de la partie (décisions, nœuds, temps, profondeurs, feuilles) |
| `moves` | lignes `move` déjà jouées, au format du journal |
| `result` | résultat si la partie est finie (pour information) |

État interne des agents :

| Agent | État sauvegardé |
|---|---|
| tous | état du générateur aléatoire (`bit_generator.state`) |
| `paranoid`, `maxn`, `social`, `negamax` | table de transposition et heuristique de l'historique (tri des coups) |
| `adaptive` | état de l'agent de repli, fenêtres du modèle d'adversaires, poids courants, dernier état diplomatique vu |

Un agent externe qui garde une mémoire entre ses décisions surcharge `snapshot_state()` et `restore_state(data)` (voir `docs/agents.md`).

## 3. Vérifications à la lecture

Une sauvegarde est refusée avec un message explicite (`CheckpointError`) si :

- son format n'est pas `tron-checkpoint/1` (version incompatible) ;
- le fichier est tronqué ou corrompu (JSON invalide) ;
- un champ obligatoire manque ;
- la grille, les têtes, les joueurs vivants, l'ordre d'élimination ou les compteurs d'équipes sont incohérents ;
- **la clé de Zobrist ne correspond pas à la grille** ;
- les agents ne correspondent pas à la configuration, ou le nombre de coups enregistrés ne correspond pas au demi-coup.

Une sauvegarde écrite par une autre version du code est acceptée avec un avertissement. L'écriture est atomique (fichier temporaire puis renommage) : une interruption ne remplace jamais une sauvegarde valide par un fichier partiel.

## 4. Garanties et limites

Les tests (`tests/test_checkpoint.py`) sauvegardent une partie à cinq joueurs (Paranoid avec table de transposition, social à diplomatie dynamique, adaptatif, MCTS, aléatoire) après 0, 9 et 31 demi-coups, la reprennent et vérifient que la suite des événements (coups, nœuds, valeurs, variantes principales, événements diplomatiques, explications), le résultat, les métriques et les statistiques cumulées sont identiques à ceux de la partie d'un seul tenant.

Limites : la durée de la partie et les temps de réflexion sont mesurés et dépendent de la machine ; ils ne sont pas comparables entre une partie reprise et une partie d'un seul tenant. La sauvegarde grossit avec la table de transposition des agents de recherche (quelques dizaines de kilo-octets sur une grille 11 × 11).
