# Rapport de mise en œuvre du plan d'améliorations (P01 à P17)

Branche `feat/plan-ameliorations`, partie du commit `5db062c` de `main`. Chaque tâche a son commit (P09 : un commit par mode), avec ses tests, sa documentation et un exemple de configuration ou de commande.

## 1. Tableau final

| ID | Statut | Commit | Tests | Limites |
|---|---|---|---|---|
| P01 Heuristique composite | fait | `706602d` | `test_composite_heuristic.py` (référence par force brute, contrat des parts, élagage exact) | noyau Numba non exécuté ici ; coût ≈ 2,5 × Voronoi en Python pur |
| P02 Métriques de coopération | fait | `2e8c75a` | `test_cooperation.py` | schéma CSV v2 (reprise des anciens fichiers avec avertissement) |
| P03 Diplomatie tactique | fait | `8c58531` | `test_diplomacy.py` | V1 : actions diplomatiques gratuites ; alliances tactiques sans effet sur les équipes gagnantes |
| P04 MCTS | fait | `e3ec88e` | `test_mcts.py` | `time_limit` dépend de la machine ; première comparaison sur un pilote |
| P05 Modèle d'adversaires, agent adaptatif | fait | `7328628` | `test_opponents.py`, `test_adaptive_agent.py` | adaptation bornée, quatre profils fixes |
| P06 Campagnes comparatives | fait | `2e3019b` | `test_campaigns.py` | seuls les pilotes ont été exécutés ; régression : repli NumPy (statsmodels absent) |
| P07 Explications de décision | fait | `91035f0` | `test_decision_details.py` | panneau Arcade validé avec un bouchon d'API |
| P08 Sauvegardes reprenables | fait | `cda0f28` | `test_checkpoint.py` | un agent externe à mémoire doit implémenter `snapshot_state` |
| P09 Modes de jeu | fait | `3aaaf1b`, `c3fed9b`, `7ead30e`, `584502d` | `test_game_modes.py` | zones : règle `head` seule ; objectifs hors contrôle de zone et survie individuelle |
| P10 Apprentissage | fait, sans DQN | `04d6f49` | `test_learning_env.py` | pas de DQN ; politique linéaire de profondeur 1 sous les agents de profondeur 2 ; adaptateur Gymnasium testé contre un bouchon |
| P11 API publique | fait | `4573381` | `test_api.py` | — |
| P12 Extensions | fait | `f0628e3` | `test_plugins.py` | installation réelle du paquet d'exemple non testée (distributions simulées) |
| P13 Profilage, charge | fait | `7f892cd` | `test_performance.py`, `pytest -m load` | références JIT à produire sur une machine avec Numba |
| P14 CI, rendu hors écran | fait, CI non exécutée | `f7f2fc6` | `test_render_headless.py` | flux GitHub jamais lancé ; couverture estimée sans pytest-cov |
| P15 Outils de configuration | fait | `d2a68b8` | `test_cli_config.py` | commandes graphiques couvertes par la procédure manuelle |
| P16 Optimisation de la recherche | fait | `f8c3bed` | `test_search_optimizations.py` | aucune nouvelle option par défaut (gain non démontré) ; `reduction` inexacte |
| P17 Comparaison de rejeux | fait | `591ea6a` | `test_viz_comparison.py` | fenêtre à deux plateaux validée avec un bouchon d'API |

## 2. Validation

Environnement de développement : Python 3.13, sans accès réseau ; NumPy, SciPy, pandas, matplotlib, imageio, Pillow, PyYAML installés ; **absents** : Numba, Arcade, statsmodels, pytest-cov / coverage, tqdm, Gymnasium.

| Commande | Résultat |
|---|---|
| `pytest` | 405 réussis, 1 ignoré (statsmodels absent), 4 tests de charge exclus par défaut |
| `NUMBA_DISABLE_JIT=1 pytest` | identique (Numba absent : tous les tests tournent déjà en Python pur) |
| `pytest -m load` | 4 réussis |
| `ruff check src tests`, `ruff format --check src tests` | conformes |
| `mypy src` | aucune erreur (72 fichiers) |
| `pytest --cov=tron` | non exécutable (pytest-cov absent) ; couverture de lignes estimée par un traceur `sys.monitoring` : 95,5 % sur `core` + `eval` + `agents`, 92 à 97 % par paquet |

Les résultats par défaut sont inchangés : clés de Zobrist d'une partie classique identiques à celles du code d'origine (test dédié), décisions de recherche identiques à la référence de mesures (sommes de contrôle), `NUMBA_DISABLE_JIT` sans effet sur les résultats.

## 3. Écarts avec la spécification

- **Ordre des tâches** : P07 a été livré avant P06 et P10 avant P11–P14 (le plan recommandait l'inverse) ; les dépendances réelles sont respectées (P10 s'appuie sur les contrats de simulation déjà stabilisés).
- **Messages de commit** : ceux du plan, à quelques mots près (P09 découpé en `P09.1` à `P09.4`, comme le plan l'exige).
- **Non réalisé** : DQN (étape facultative de P10), mesures JIT (P01, P13), exécution du flux CI (P14), validation des fenêtres Arcade sur écran (P07, P14, P17 : procédure dans `docs/validation-gui.md`).
- **Choix de conception notables** : la table de transposition ne stocke que des coups (jamais des valeurs) ; obstacles déterministes par hachage sans état (`tron.core.rng.hash_index`) ; compteurs de tours et de scores non bornés ; extensions : les noms intégrés restent prioritaires et aucun chemin de module n'est importé depuis un YAML.
