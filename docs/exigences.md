# Traçabilité des exigences

Chaque exigence du cahier des charges est reliée à son implémentation et aux tests qui la vérifient.
Priorités : [M] obligatoire, [S] recommandé, [C] optionnel.

## Exigences fonctionnelles

| Exigence | Implémentation | Tests |
|---|---|---|
| F1.1 [M] grille W×H 5–100, nord = y − 1 | `core/config.py` (validation), `core/state.py` | `test_core` : validation, bords, coups légaux |
| F1.2 [M] 2 à 8 joueurs, ordre cyclique, premier joueur tournant | `GameState._next_alive`, `sim/arena.game_config`, `Plan.tasks` (`seats`) | `test_core::test_rotation…`, `test_sim::test_arena…`, `test_experiments::test_paired…` |
| F1.3 [M] coup = direction N/E/S/O, sans inertie | `core/directions.py`, `GameState.apply` | `test_core` |
| F1.4 [M] élimination : sortie, case occupée, aucun coup légal | `GameState.apply` (`PASS`), `Match.step` | `test_core::test_collision…`, `test_sim::test_pass_events…` |
| F1.5 [M] fin à une équipe ; vainqueur, rangs, coups | `GameState.outcome`, `MatchResult` | `test_core::test_team_game…` |
| F1.6 [M] placement régulier, aléatoire ou manuel | `core/placement.py` | `test_core::test_regular…`, `test_placement…` |
| F1.7 [M] tout l'aléa dérive d'une graine enregistrée | `core/rng.py` (flux indépendants), `MatchResult.seed` | `test_sim::test_match_is_reproducible` |
| F1.8 [M] `legal_moves`, `apply` / `undo` sans copie, `is_terminal`, `outcome` | `core/state.py` | `test_core::test_apply_undo_restores_everything` |
| F2.1 [M] Voronoi, départage par ordre de jeu ou cases neutres | `eval/voronoi.py` | `test_eval::test_kernel_matches_naive_reference` |
| F2.2 [M] vecteur h de somme 1 | `VoronoiHeuristic.shares` | `test_eval` |
| F2.3 [M] terminaux dominants ; [S] victoires rapides, défaites tardives | `eval/valuation.py` | `test_eval::test_valuation…` |
| F2.4 [C] séparation, taille de région (borne de parité) | option `separation` | `test_eval::test_separation…` |
| F3.1 [M] `choose_move`, profondeur par joueur, statistiques | `agents/base.py` (`Decision`, `SearchStats`) | `test_agents::test_agents_restore_state…` |
| F3.2 [M] aléatoire, glouton | `agents/simple.py` | `test_agents::test_greedy_beats_random_in_duels` (J3) |
| F3.3 [M] Paranoid, alpha-bêta complet | `ParanoidAgent`, moteur `alphabeta` | `test_agents::test_alphabeta_equals_full_minimax` (J4) |
| F3.4 [S] MaxN, élagage superficiel de Korf | `MaxNAgent`, moteur `maxn` | `test_agents::test_maxn_shallow_pruning_is_exact` (J4) |
| F3.5 [S] Negamax alpha-bêta de référence | `NegamaxAgent`, moteur `negamax` | `test_agents::test_two_players_all_algorithms_agree` (J4) |
| F3.6 [S] tri, approfondissement itératif ; [C] table de Zobrist | `agents/search.py`, `core/zobrist.py` | `test_agents` (options `history`, `tt`, `iterative`) |
| F3.7 [M] départage aléatoire (graine) ou N, E, S, O | `Agent.pick` | `test_agents::test_tie_break_modes` |
| F4.1 [M] équipes statiques dans la configuration | `GameConfig.team_index` | `test_core` |
| F4.2 [M] matrice W, choix par W[p]·h, h brut remonté | `agents/social.py`, moteur `maxn` | `test_agents::test_generalized_pruning…` |
| F4.3 [M] profils MaxN, Paranoid, Équipe (α), Équipe paranoïaque | `social_matrix` | `test_agents::test_social_matrices_and_reductions` |
| F4.4 [M] alpha-bêta si deux camps, MaxN sinon | `two_camp_reduction`, `SocialAgent.engine_name` | `test_agents::test_two_team_reduction…` |
| F4.5 [M] recherche propre à chaque membre | un agent par joueur | `test_sim` |
| F4.6 [S] matrice W libre | profil `custom` | `test_agents::test_generalized_pruning…` |
| F5.1 [M] configuration YAML | `GameConfig.from_yaml` | `test_core::test_config_roundtrip` |
| F5.2 [M] arène, résultat et journal JSONL rejouable | `sim/arena.py`, `sim/recorder.py` | `test_sim::test_recorder_roundtrip` |
| F5.3 [M] plan YAML, plan complet ou partiel, graines appariées | `experiments/plan.py` | `test_experiments::test_designs_and_filters` |
| F5.4 [M] parallèle, déterministe, écriture incrémentale, reprise | `experiments/runner.py`, `store.py` | `test_experiments::test_run_plan_resume_and_determinism` |
| F5.5 [M] CSV, une ligne par partie | `MatchResult.to_row`, `run_task` | `test_experiments` |
| F6.1 [M] couleurs par équipe, têtes, éliminés ; [S] palette daltonien | `viz/palette.py`, `viz/render.py` | `test_viz::test_palette`, `test_render_frame` |
| F6.2 [M] temps réel et rejeu, pause, pas à pas avant / arrière, vitesse, nouvelle partie, capture | `viz/controller.py`, `viz/arcade_app.py` | `test_viz::test_live_controller`, `test_replay_controller`, `test_arcade_window_with_stub` |
| F6.3 [S] panneau d'information, surcouche de Voronoi | `PlaybackController.panel`, `owners` | `test_viz` |
| F6.4 [S] export GIF / MP4 | `viz/export.py` | `test_viz::test_exports` |
| F6.5 [M] mode sans affichage | `sim` indépendant de `viz`, `--no-gui` | `test_analysis_cli::test_cli_end_to_end` |
| F7.1 [M] taux de victoire, Wilson, rang, durée, coût | `analysis/stats.py` | `test_analysis_cli::test_wilson_interval`, `test_summary…` |
| F7.2 [M] courbes et cartes de chaleur régénérées depuis les CSV | `analysis/figures.py`, `tron analyze`, notebook | `test_analysis_cli::test_cli_end_to_end` |
| F7.3 [S] régression logistique avec interactions | `analysis/regression.py` | `test_analysis_cli::test_summary_paired_and_regression` |
| F7.4 [M] commande `tron` : play, replay, bench, run, analyze | `cli/main.py` | `test_analysis_cli::test_cli_end_to_end` |

## Exigences non fonctionnelles

| Exigence | Réponse |
|---|---|
| Performance | `tron bench`. Mesuré en Python pur (sans Numba) sur 20 × 20 à 4 joueurs : 4 800 évaluations de Voronoi par seconde, décision Paranoid à 10 plies de 550 ms (médiane). Numba compile le même noyau ; l'objectif de 100 000 évaluations par seconde est à mesurer avec `tron bench` une fois Numba installé. |
| Reproductibilité | graines dérivées, départages canoniques, tables de Zobrist à graine fixe, résultats indépendants de l'ordonnancement (test dédié) |
| Portabilité | Python pur et NumPy, chemins `pathlib`, encodage UTF-8 explicite, processus compatibles `spawn` et `forkserver` |
| Qualité | annotations de types (mypy sans erreur), ruff, 82 tests, couverture de lignes de 94 % sur `core`, `eval` et `agents` |
| Extensibilité | registres d'agents, d'heuristiques et de scénarios (`register_*`) |
| Documentation | README, docstrings, `docs/architecture.md`, `docs/algorithmes.md`, ce document |

## Jalons

| Jalon | Critère | État |
|---|---|---|
| J1 Moteur | règles testées, parties identiques à graine égale | fait |
| J2 Visualisation | temps réel et rejeu d'une partie à 4 joueurs | contrôleur et rendu testés ; fenêtre arcade à valider sur un poste graphique |
| J3 Heuristique et agents de base | Voronoi validé, glouton ≥ 90 % contre l'aléatoire, benchmark, calibrage | fait (`tron bench`, `tron run --estimate`, plan pilote) |
| J4 Paranoid et MaxN | élagage = sans élagage ; accord à deux joueurs | fait |
| J5 Équipes | SOS Paranoid = Paranoid ; SOS Équipe α = 0 = MaxN | fait |
| J6 Campagne | études A et B exécutées | plans prêts (`etude_a.yaml`, `etude_b.yaml`) ; pilote exécuté |
| J7 Analyse et rapport | figures, discussion | chaîne d'analyse prête ; rapport à rédiger à partir des résultats |
