# Visualisation, rejeu et explications

La visualisation n'est qu'un observateur : le moteur fonctionne sans elle (`--no-gui`). La logique de lecture (`tron.viz.controller`) et le rendu (`tron.viz.render`) sont indépendants d'Arcade et testés sans écran ; seule `tron.viz.arcade_app` dessine dans une fenêtre.

## 1. Commandes et touches

```bash
tron play configs/games/solo_vs_3_20x20.yaml          # temps réel
tron replay results/ffa4/game_0000.jsonl              # rejeu, avant et arrière
tron replay results/ffa4/game_0000.jsonl --gif p.gif  # export GIF / MP4
```

| Touche | Action |
|---|---|
| `Espace` | pause / lecture |
| `→` / `←` | demi-coup suivant / précédent (retour arrière en rejeu seulement) |
| `↑` / `↓` | vitesse |
| `V` | surcouche des territoires de Voronoi |
| `D` | panneau de la dernière décision (P07) |
| `N` | nouvelle partie (temps réel) ou retour au début (rejeu) |
| `Début` / `Fin` | début / fin du rejeu |
| `S` | capture d'écran |
| `Échap` / `Q` | quitter |

## 2. Explication des décisions (P07)

Chaque décision porte un `DecisionDetails` typé et sérialisable (`tron.agents.base`), rempli uniquement avec ce que l'agent a réellement calculé :

| Agent | Informations |
|---|---|
| `greedy` | valeur de chaque coup légal |
| `paranoid`, `maxn`, `social`, `negamax` | valeur exacte des coups racine entièrement évalués ; « non disponible » pour un coup seulement borné par l'élagage ; poids W[p] et alliés tactiques (`social`) |
| `mcts` | utilité moyenne et visites de chaque coup racine |
| `adaptive` | ce que fournit l'agent de repli, plus les poids d'heuristique utilisés et le profil le plus probable de chaque adversaire |
| `random` | « non disponible » pour tous les coups (coup tiré au hasard) |

Une valeur qui n'a pas été calculée n'est **jamais reconstruite** : elle s'affiche « non disponible ».

Chaîne de transmission : `Decision.details` → `MoveEvent.details` → ligne `move` du journal (`details`, plus `pv`, `leaves`, `prunes`) → `GameLog.details(i)` → `PlaybackController.decision_panel()` → panneau Arcade.

Le panneau (`D`) affiche : joueur et agent, coup, valeur, moteur, profondeur, nœuds, feuilles ou simulations, coupes, variante principale, coups classés avec leurs valeurs, puis les lignes propres à l'agent (visites, poids W, alliés, poids d'heuristique, profils d'adversaires). La suite de la variante principale est dessinée sur la grille (petits carrés à la couleur du joueur) ; `render_frame(..., path=pv_path(état, coups))` produit la même surcouche sans Arcade.

Un rejeu restitue exactement les mêmes informations qu'en direct. Un journal sans détails (version antérieure ou `Recorder(path, details=False)`) se rejoue normalement : le panneau indique alors « non disponible ». Pour une campagne massive, `details: false` dans le plan supprime les détails des journaux (les journaux eux-mêmes sont facultatifs, option `--logs`).

## 3. Validation de l'interface graphique

Les tests utilisent un bouchon d'API Arcade : ils vérifient la logique de la fenêtre (touches, appels de dessin, textes), pas le rendu réel. La fenêtre doit être validée sur un poste graphique avec la procédure de `docs/validation-gui.md`.
