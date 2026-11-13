# Validation de l'interface graphique (P14)

Les tests automatiques ne valident **pas** la fenêtre Arcade : `tests/test_render_headless.py` vérifie l'image NumPy produite par `render_frame` (dimensions, type `uint8`, couleurs des murs, des têtes, des éliminés et des territoires) et `tests/test_viz.py` la logique de la fenêtre avec un bouchon de l'API Arcade (touches, appels de dessin, textes). Le rendu réel (OpenGL, polices, fenêtre, clavier) se valide à la main, sur un poste avec écran, avec la procédure ci-dessous, à chaque version qui touche `tron.viz` ou la version d'Arcade.

## 1. Préparation

- poste avec affichage (Linux X11/Wayland, macOS ou Windows), Python 3.14 et Arcade 3.3.3 (versions de `pyproject.toml`) ;
- `uv sync --extra dev`, puis `uv run tron --version` ;
- noter : date, système, carte graphique si connue, `python --version`, `python -c "import arcade; print(arcade.version.VERSION)"`, commit testé.

## 2. Essai de fumée

```bash
tron play configs/games/solo_vs_3_20x20.yaml
```

La fenêtre s'ouvre en moins de quelques secondes (première compilation Numba comprise), la partie se joue en temps réel, le panneau latéral affiche le titre, la ligne d'état, le demi-coup, les joueurs et les équipes, et la partie se termine par « Victoire de l'équipe … ».

## 3. Touches (temps réel)

| Touche | Attendu | OK |
|---|---|---|
| `Espace` | pause, puis reprise ; la ligne d'état indique « Pause » | ☐ |
| `→` | un seul demi-coup joué, la lecture reste en pause | ☐ |
| `←` | met la lecture en pause ; pas de retour arrière en temps réel | ☐ |
| `↑` / `↓` | la vitesse augmente / diminue (ligne d'état « Lecture ×… ») | ☐ |
| `V` | surcouche des territoires de Voronoi affichée, puis masquée | ☐ |
| `D` | panneau de la dernière décision (coup, valeur, moteur, nœuds, variante principale dessinée sur la grille), puis masqué | ☐ |
| `N` | nouvelle partie (demi-coup 0) | ☐ |
| `S` | capture PNG écrite dans `screenshots/` (horodatée), image conforme à la fenêtre | ☐ |
| `Échap` puis `Q` (relancer entre les deux) | la fenêtre se ferme proprement, sans trace d'erreur | ☐ |

## 4. Rejeu

```bash
tron play configs/games/ffa4_20x20.yaml --no-gui --out /tmp/ffa4
tron replay /tmp/ffa4/game_0000.jsonl
```

| Touche | Attendu | OK |
|---|---|---|
| `Espace` | lecture / pause | ☐ |
| `→` / `←` | demi-coup suivant / précédent, l'état affiché est cohérent (murs ajoutés puis retirés) | ☐ |
| `Début` / `Fin` | premier / dernier demi-coup | ☐ |
| `N` | retour au début du rejeu | ☐ |
| `D` | mêmes informations de décision qu'en direct ; « non disponible » pour ce qui n'a pas été calculé | ☐ |

## 5. Modes de jeu et exports

| Commande | Attendu | OK |
|---|---|---|
| `tron play configs/games/zone_control_16x16.yaml` | quatre zones entourées ; ligne d'état « … · tour N · A x – B y » ; fin « (score cible atteint) » ou par élimination | ☐ |
| `tron play configs/games/obstacles_14x14.yaml` | obstacles gris apparaissant en fin de tour | ☐ |
| `tron play configs/games/hunters_prey_12x12.yaml` | fin « (objectif de survie atteint) » ou victoire des chasseurs | ☐ |
| `tron replay /tmp/ffa4/game_0000.jsonl --gif /tmp/p.gif --mp4 /tmp/p.mp4` | fichiers lisibles, mêmes images que la fenêtre | ☐ |

## 6. Comparaison de rejeux

```bash
tron play configs/games/ffa4_20x20.yaml --no-gui --out /tmp/a
tron play configs/games/ffa4_20x20.yaml --no-gui --out /tmp/b --seed 1
tron compare /tmp/a/game_0000.jsonl /tmp/b/game_0000.jsonl
```

| Vérification | Attendu | OK |
|---|---|---|
| ouverture | deux plateaux côte à côte, panneau de comparaison, avertissement « comparaison non appariée : graines différentes » | ☐ |
| `→` / `←`, `Début` / `Fin`, `Espace`, `↑` / `↓` | les deux plateaux avancent et reculent ensemble ; la partie la plus courte se fige (cadre gris) | ☐ |
| lignes de temps | marques d'événements sous chaque plateau, curseur qui suit le demi-coup | ☐ |
| `V` | territoires sur les deux plateaux | ☐ |
| `Échap` | fermeture propre | ☐ |

## 7. Compte rendu

Consigner, dans la demande de fusion ou la note de version : l'environnement noté au § 1, les cases cochées, et pour chaque écart une capture (`S`) et la commande exacte. Un écart bloquant (fenêtre qui ne s'ouvre pas, touche sans effet, plantage) empêche la publication d'une version qui modifie `tron.viz`.
