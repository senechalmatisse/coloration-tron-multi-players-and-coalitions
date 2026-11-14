# Validation de l'interface graphique (P14)

Les tests automatiques ne valident **pas** la fenêtre Arcade : `tests/test_render_headless.py` vérifie l'image NumPy produite par `render_frame` (dimensions, type `uint8`, couleurs des murs, des têtes, des éliminés et des territoires), `tests/test_viz_neon.py` le rendu NumPy néon, et `tests/test_viz.py` et `tests/test_viz_neon_window.py` la logique des fenêtres avec un bouchon de l'API Arcade (touches, appels de dessin, textes, sprites, bloom simulé). L'aperçu simulé de `tests/arcade_preview.py` contrôle la mise en page, pas le rendu OpenGL. Le rendu réel (OpenGL, polices, fenêtre, clavier) se valide à la main, sur un poste avec écran, avec la procédure ci-dessous, à chaque version qui touche `tron.viz` ou la version d'Arcade.

## 1. Préparation

- poste avec affichage (Linux X11/Wayland, macOS ou Windows), Python 3.14 et Arcade 3.3.3 (versions de `pyproject.toml`) ;
- `uv sync --extra dev`, puis `uv run tron --version` ;
- noter : date, système, carte graphique si connue, `python --version`, `python -c "import arcade; print(arcade.version.VERSION)"`, commit testé.

## 2. Essai de fumée

```bash
tron play configs/games/solo_vs_3_20x20.yaml --theme classic
```

Les § 2 à 6 se valident avec le thème classique (`--theme classic`), le § 7 avec le thème néon (défaut). La fenêtre s'ouvre en moins de quelques secondes (première compilation Numba comprise), la partie se joue en temps réel, le panneau latéral affiche le titre, la ligne d'état, le demi-coup, les joueurs et les équipes, et la partie se termine par « Victoire de l'équipe … ».

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

## 7. Thème néon

```bash
tron play configs/games/ffa4_20x20.yaml                       # néon, qualité medium
tron play configs/games/melee_8_30x30.yaml --quality ultra    # 8 joueurs, bloom
tron play configs/games/diplomacy_4p_20x20.yaml               # alliances tactiques
tron replay /tmp/ffa4/game_0000.jsonl
```

| Vérification | Attendu | OK |
|---|---|---|
| ouverture | arène sombre centrée verticalement, grille fine, lignes majeures, liseré cyan lumineux ; HUD à droite (titre `TRON // COALITIONS`, cartes des joueurs, statistiques, événements, aide sur trois lignes) sans texte qui déborde | ☐ |
| traces | lueur continue, sans couture ni point plus clair aux jonctions des cases ni aux virages ; cœur blanc net | ☐ |
| motos | orientées dans le sens du déplacement, phare à l'avant, halo qui pulse doucement ; numéro du joueur lisible à côté | ☐ |
| élimination | explosion, onde de choc, éclair bref, léger tremblement (≤ 4 px), trace qui s'éteint en ~0,6 s, croix rouge, ligne `JOUEUR … ÉLIMINÉ · cause` dans le journal | ☐ |
| fin de partie | gerbe de particules et bandeau `VICTOIRE · …` ; badges `VAINQUEUR` / `VAINCU` ou rang | ☐ |
| alliances (diplomatie) | lien pointillé animé entre alliés, qui ne masque pas les traces ; ligne `ALLIANCE a + b` ; `A` ajoute les liens entre coéquipiers | ☐ |
| `V`, `D` | territoires teintés à frontières lumineuses ; losanges de la variante principale et panneau de décision | ☐ |
| `G` | `low` (halo réduit, pas d'animation de fond), `medium`, `ultra` (bloom) ; ligne « rendu » du HUD à jour | ☐ |
| `B` en `ultra` | le bloom s'allume et s'éteint ; s'il est indisponible, la ligne « rendu » n'indique pas « bloom » et un avertissement est journalisé, sans plantage | ☐ |
| `T` | bascule vers le thème classique et retour, partie conservée | ☐ |
| rejeu : `←`, `Début`, `Fin` | traces retirées et rajoutées sans artefact ; journal élagué au retour arrière ; effets effacés | ☐ |
| fluidité | ≥ 60 images/s à 8 joueurs (`melee_8_30x30`) en `medium` ; noter le taux en `ultra` | ☐ |
| zones et obstacles | `zone_control_16x16` : zones teintées de l'équipe qui les tient ; `obstacles_14x14` : blocs sombres au liseré orangé, flash à l'apparition | ☐ |
| comparaison | `tron compare a.jsonl b.jsonl` : deux arènes néon synchronisées, titres sans chevauchement | ☐ |
| écran haute densité (Retina, 4K à 200 %) | image nette, bloom et tremblement correctement alignés | ☐ |

## 8. Compte rendu

Consigner, dans la demande de fusion ou la note de version : l'environnement noté au § 1, les cases cochées, et pour chaque écart une capture (`S`) et la commande exacte. Un écart bloquant (fenêtre qui ne s'ouvre pas, touche sans effet, plantage) empêche la publication d'une version qui modifie `tron.viz`.
