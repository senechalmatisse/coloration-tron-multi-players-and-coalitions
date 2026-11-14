# Visualisation, rejeu et explications

La visualisation n'est qu'un observateur : le moteur fonctionne sans elle (`--no-gui`). La logique de lecture (`tron.viz.controller`), la scène (`tron.viz.scene`) et les rendus NumPy (`tron.viz.render`, `tron.viz.neon`) sont indépendants d'Arcade et testés sans écran ; seules `tron.viz.arcade_app`, `tron.viz.comparison_app` et `tron.viz.neon_arcade` dessinent dans une fenêtre.

Deux thèmes : **néon** (« Neon Arena », défaut de la ligne de commande) et **classique** (palette validée pour les daltoniens, défaut des fonctions de la bibliothèque). La touche `T` bascule de l'un à l'autre en cours de partie.

## 1. Commandes et touches

```bash
tron play configs/games/solo_vs_3_20x20.yaml                    # temps réel (thème néon)
tron play configs/games/solo_vs_3_20x20.yaml --quality ultra    # néon avec bloom
tron play configs/games/solo_vs_3_20x20.yaml --theme classic    # palette d'origine
tron replay results/ffa4/game_0000.jsonl                        # rejeu, avant et arrière
tron replay results/ffa4/game_0000.jsonl --gif p.gif            # export GIF / MP4 (néon)
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
| `T` | thème néon / classique |
| `G` | qualité graphique du thème néon : `low` → `medium` → `ultra` |
| `B` | post-traitement *bloom* (thème néon) |
| `A` | liens entre coéquipiers vivants (les alliances tactiques sont toujours tracées) |
| `S` | capture d'écran |
| `Échap` / `Q` | quitter |

## 2. Explication des décisions

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

## 4. Comparaison de deux rejeux

```bash
tron play configs/games/ffa4_20x20.yaml --no-gui --out /tmp/a
tron play configs/games/ffa4_20x20.yaml --no-gui --out /tmp/b --seed 1
tron compare /tmp/a/game_0000.jsonl /tmp/b/game_0000.jsonl                       # fenêtre à deux plateaux
tron compare /tmp/a/game_0000.jsonl /tmp/b/game_0000.jsonl --no-gui --ply 40     # panneau en texte
tron compare /tmp/a/game_0000.jsonl /tmp/b/game_0000.jsonl --no-gui --format json
tron compare /tmp/a/game_0000.jsonl /tmp/b/game_0000.jsonl --png c.png --gif c.gif --no-gui
```

Deux journaux (`tron-log/1` ou `tron-log/2`) sont rejoués côte à côte, **synchronisés par un curseur commun de demi-coup**. Rien n'est recalculé par les agents : seuls les coups enregistrés sont rejoués ; les parts de territoire sont recalculées par Voronoi sur l'état affiché. Un journal plus court reste figé sur son état final (cadre gris dans la fenêtre, « figé : partie terminée » dans le panneau).

**Panneau de comparaison** : configuration, graine et premier joueur, sièges (équipes), agents, gagnant, rangs, durée, demi-coup affiché, parts territoriales par équipe, décisions, temps et nœuds cumulés par joueur jusqu'au curseur, nombre d'événements, **première divergence** (premier demi-coup où le joueur ou la direction diffèrent). Sous chaque plateau, une ligne de temps marque les événements du journal (éliminations, diplomatie, changements de contrôle de zone, obstacles) et le curseur ; les trois derniers événements atteints sont écrits en clair.

**Appariement.** La comparaison n'est dite appariée que si les deux parties ont la même grille, le même nombre de joueurs, les mêmes équipes par siège, les mêmes règles, la même graine, le même placement initial et le même premier joueur. Sinon, un avertissement liste les écarts (fenêtre, panneau texte, sortie d'erreur de la commande) : la comparaison reste affichée, mais ne doit pas être lue comme un appariement.

Touches de la fenêtre : `Espace` lecture / pause, `→` / `←` demi-coup suivant / précédent, `↑` / `↓` vitesse, `Début` / `Fin`, `V` territoires, `Échap` / `Q` quitter. La logique (`tron.viz.comparison.ComparisonSession`) est testée sans fenêtre (`tests/test_viz_comparison.py`, journaux courts, dont un `tron-log/1`) ; la fenêtre (`tron.viz.comparison_app`) se valide avec la procédure de `docs/validation-gui.md`.

## 5. Thème néon « Neon Arena »

Ambiance Tron Legacy / cyberpunk en 2D : fond très sombre, grille fine, traces lumineuses, motos stylisées, particules et HUD holographique. Le jeu reste lisible : la caméra est fixe et montre toute l'arène ; les effets sont courts et ne masquent jamais les traces.

```bash
tron play configs/games/melee_8_30x30.yaml --quality ultra      # 8 joueurs, bloom
tron replay /tmp/ffa4/game_0000.jsonl --gif p.gif --quality low # export néon léger
tron compare a.jsonl b.jsonl --png c.png --no-gui               # image côte à côte néon
```

Options communes à `play`, `replay` et `compare` : `--theme {neon,classic}` (défaut `neon`) et `--quality {low,medium,ultra}` (défaut `medium`). Elles valent pour la fenêtre comme pour les exports PNG, GIF et MP4.

### 5.1 Ce qui est dessiné

| Élément | Rendu |
|---|---|
| arène | fond `#030712` → `#071525` en dégradé radial, grille fine `#123047`, lignes majeures lumineuses toutes les 5 cases (10 au-delà de 30 cases), liseré cyan lumineux, bande de balayage lente (`medium`, `ultra`) |
| traces | trois couches : halo large, corps coloré, cœur blanc bleuté `(235, 250, 255)` ; virages à angle droit nets ; une trace d'éliminé s'éteint en 0,6 s vers une teinte grise |
| motos | silhouette vue de dessus orientée selon le dernier déplacement : deux roues cerclées de lumière, carénage, cockpit, phare, feu arrière ; halo pulsant `1 + 0,2 · sin(5 t)` ; moto éteinte barrée d'une croix rouge après élimination |
| couleurs | une teinte néon par équipe (cyan, magenta, orange, vert, violet, rouge, jaune, bleu), des nuances pour les membres ; une équipe nommée par une couleur (« Rouge », « Blue »…) en reçoit la teinte. Le numéro du joueur est inscrit près de sa moto : l'identité ne repose pas sur la seule couleur |
| événements | virage : étincelles ; élimination : explosion aux couleurs mêlées si la trace d'un autre joueur est heurtée, onde de choc, désintégration de la fin de trace, bref éclair et léger tremblement ; alliance : lien qui se tend entre les deux têtes ; rupture : étincelles rouges ; obstacles : flash orangé ; victoire : gerbe de particules et bandeau « VICTOIRE · équipe » |
| coalitions | alliances tactiques actives : liens pointillés animés entre les têtes, sous les motos et sans masquer les traces ; `A` ajoute les liens entre coéquipiers |
| territoires (`V`) | cases teintées de la couleur du propriétaire de Voronoi, frontières lumineuses |
| décision (`D`) | variante principale en losanges lumineux, panneau holographique en bas de l'arène |
| modes de jeu | zones de contrôle teintées de l'équipe qui les tient ; obstacles en blocs de métal sombre au liseré orangé |
| HUD | titre, état, une carte par joueur (badge `ACTIF`, `ÉLIMINÉ`, `VAINQUEUR`, `VAINCU` ou rang ; barre de territoire ; profondeur, nœuds, valeur, alliés, agent), statistiques (joueurs actifs, cases occupées, éliminations, alliances, demi-coup et tour, temps écoulé en direct, vitesse, rendu) et journal horodaté par demi-coup (`[042] JOUEUR 3 (hasard) ÉLIMINÉ · trace de 1`) |

### 5.2 Niveaux de qualité

| Niveau | Traces | Animations | Particules | Tremblement, éclair | Bloom |
|---|---|---|---|---|---|
| `low` | deux couches (corps et lueur serrée), grille simple | non | 80 au plus, peu émises | non | non |
| `medium` (défaut) | trois couches, lignes majeures | oui | 500 au plus | oui, à vitesse ≤ 20 demi-coups/s | non |
| `ultra` | trois couches, lignes majeures | oui | 1 500 au plus | oui | oui (`B` le coupe) |

Le *bloom* (extraction des pixels lumineux, flou gaussien séparable à demi-résolution, composition additive) passe par `arcade.gl` et des shaders GLSL 3.30 (`tron/viz/shaders`). À la moindre erreur (pilote, contexte OpenGL, API), il se désactive, un avertissement est journalisé et le rendu continue sans lui ; la ligne « rendu » du HUD indique s'il est actif. Les particules sont calculées sur le CPU (NumPy) ; il n'y a pas de particules GPU.

### 5.3 Architecture et performances

| Module | Rôle | Arcade |
|---|---|---|
| `tron.viz.theme` | couleurs, tailles relatives à la case, paramètres d'animation, niveaux de qualité | non |
| `tron.viz.scene` | traces reconstruites de `GameState.history` (incrémental, retour arrière compris), orientations, connexions de chaque case, événements visuels et leur cause, alliances actives, frontières de territoire | non |
| `tron.viz.neon` | images partagées (tuiles de traces, motos, fond, obstacles) et rendu NumPy des exports | non |
| `tron.viz.effects` | particules (flux d'aléa `STREAM_EFFECTS`, déterministe), mise en scène des événements ; `bloom` (Arcade) | non, sauf `bloom` |
| `tron.viz.hud` | cartes des joueurs, statistiques, journal d'événements | non |
| `tron.viz.neon_arcade` | arène et HUD en sprites et primitives Arcade | oui |

- **Une trace = des tuiles.** Chaque case d'une trace est une tuile choisie parmi 16 selon ses connexions (nord, est, sud, ouest). Le corps est rastérisé exactement (rectangles disjoints) et la lueur est une somme de flous gaussiens : le flou étant linéaire, la somme des tuiles en mélange additif *est* la lueur de la trace entière. La fenêtre dessine ces tuiles comme des sprites (un demi-coup ajoute deux sprites et change la texture de la case précédente : coût constant) ; le rendu NumPy les cumule dans un calque par joueur. Fenêtre et exports produisent donc la même image, au quantifiage 8 bits près.
- Les textures sont créées une fois par clé (`TextureBank`) : changer de qualité ou de thème ne fait pas grossir l'atlas. Le fond (dégradé, grille, liseré) est une seule texture.
- Les objets `arcade.Text` sont créés une fois ; leur contenu n'est réaffecté que s'il change. Toutes les animations suivent `delta_time`.
- En avance rapide (plus de 12 demi-coups traités d'un coup), aucune particule n'est émise ; au-delà de 20 demi-coups/s, ni tremblement ni éclair. Un retour arrière efface les effets en cours.
- Le moteur ne dépend pas du rendu : la scène observe l'historique de l'état ; la direction jouée, la diplomatie, les zones et les obstacles viennent de `MoveEvent` (direct) ou des lignes du journal (rejeu) via `PlaybackController.ply_extras`.

### 5.4 Exports et aperçu sans écran

`log_frames(log, theme="neon", quality=..., fps=...)` et `neon_frames` produisent les images néon d'un journal, effets animés compris (le temps des effets avance de `1/fps` par image ; les dernières images laissent la victoire se dérouler). `render_neon(état)` rend un état isolé. Les images ont une marge de `0,75 · case` pixels pour le liseré.

Les tests (`tests/test_viz_neon.py`, `tests/test_viz_neon_window.py`) vérifient la logique avec un bouchon d'Arcade (`tests/arcade_stub.py`). `tests/arcade_preview.py` va plus loin : un « Arcade logiciel » qui exécute réellement les appels de dessin des fenêtres (sprites en mélanges alpha et additif, rectangles, lignes, textes) et en produit une image, pour contrôler la mise en page sans écran. C'est un **aperçu simulé** : polices, anticrénelage et bloom diffèrent d'Arcade ; la validation sur un poste graphique (`docs/validation-gui.md`, § 7) reste nécessaire.

### 5.5 Accessibilité

Les teintes néon ne sont pas validées pour les daltoniens. Le thème classique (`--theme classic` ou `T`) garde la palette validée (séparation CVD et contraste) ; dans les deux thèmes, le numéro de chaque joueur est affiché près de sa tête et le panneau le nomme. Le tremblement de caméra est léger (4 pixels au plus, 0,22 s) et absent en qualité `low`.
