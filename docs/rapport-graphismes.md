# Refonte graphique « Neon Arena » : rapport

Ce rapport fait correspondre les propositions de refonte graphique (ambiance Tron Legacy / cyberpunk, sans passer en 3D) à ce qui a été réalisé. La description complète du thème est dans `docs/visualisation.md` (§ 5) ; la procédure de validation sur poste graphique dans `docs/validation-gui.md` (§ 7).

## 1. Correspondance avec les propositions

| Proposition | Réalisation | Fichiers |
|---|---|---|
| palette néon centralisée | `theme.py` : couleurs d'environnement (`#030712`, `#071525`, `#123047`), 8 teintes néon, couleurs d'interface, épaisseurs relatives à la case, paramètres d'animation ; une équipe nommée par une couleur en reçoit la teinte | `viz/theme.py` |
| traces néon à trois couches | halo large, corps coloré, cœur blanc `(235, 250, 255)` ; tuiles par case dont la somme reconstitue exactement la lueur de la trace (flou linéaire) | `viz/neon.py`, `viz/neon_arcade.py` |
| motos stylisées orientées | silhouette vue de dessus (roues lumineuses, carénage, cockpit, phare, feu arrière), une texture par orientation, halo pulsant `1 + 0,2 · sin(5 t)` | `viz/neon.py` |
| arène : fond, grille, lignes majeures, bordure lumineuse, fond animé | une texture de fond (dégradé radial, grille fine, lignes majeures toutes les 5 ou 10 cases, liseré cyan lumineux) ; bande de balayage lente | `viz/neon.py` |
| bloom | extraction, flou gaussien séparable horizontal puis vertical à demi-résolution, composition additive ; shaders GLSL 3.30 ; qualité `ultra` ou touche `B` ; repli automatique en cas d'erreur | `viz/effects/bloom.py`, `viz/shaders/` |
| particules CPU | système vectorisé NumPy, déterministe (`STREAM_EFFECTS`) : traînée, étincelles de virage, explosion (couleurs mêlées en cas de collision avec une autre trace), désintégration, gerbe de victoire, formation d'alliance, apparition d'obstacles | `viz/effects/particles.py`, `viz/effects/director.py` |
| HUD holographique | cartes joueurs (ACTIF / ÉLIMINÉ / VAINQUEUR / VAINCU / rang, barre de territoire, alliés), statistiques, journal horodaté ; objets `arcade.Text` persistants | `viz/hud.py`, `viz/neon_arcade.py` |
| caméra | caméra fixe et globale ; tremblement court et léger (≤ 4 px, 0,22 s) lors d'une élimination, coupé en `low` et au-delà de 20 demi-coups/s | `viz/arcade_app.py` |
| visualisation des coalitions | liens pointillés animés entre alliés (sans masquer les traces), lien qui se tend à la formation ; liens entre coéquipiers (`A`) ; territoires de Voronoi teintés à frontières lumineuses ; variante principale en losanges | `viz/scene.py`, `viz/neon_arcade.py` |
| performances | traces en sprites mis à jour en temps constant par demi-coup, textures créées une fois (`TextureBank`), fond en une seule texture, `delta_time` partout, effets coupés en avance rapide | `viz/neon_arcade.py` |
| niveaux de qualité | `low`, `medium` (défaut), `ultra` ; `--quality` et touche `G` | `viz/theme.py` |
| moteur séparé du rendu | la scène observe `GameState.history` (nouvelle propriété en lecture seule) ; direction jouée, diplomatie, zones et obstacles viennent de `MoveEvent` ou du journal | `viz/scene.py`, `viz/controller.py` |

En plus : exports GIF / MP4 / PNG en néon identiques à la fenêtre (effets compris), comparaison de rejeux en néon, bascule de thème `T`, et correction d'un chevauchement existant des titres de la fenêtre de comparaison (bandeau de titres réservé, titres tronqués).

## 2. Écarts assumés

- **Particules GPU** : non réalisées ; les particules CPU tiennent le budget (1 500 au plus en `ultra`).
- **Fond par shader** : le fond animé est une texture et une bande de balayage dessinée en primitives ; seul le bloom utilise des shaders.
- **Zoom de caméra** : non réalisé, pour garder la vue globale fixe.
- **Teintes néon et daltonisme** : non validées ; le thème classique (`--theme classic`, touche `T`) garde la palette validée.

## 3. Compatibilité

- Défauts : la ligne de commande passe au thème néon (`--theme neon --quality medium`) ; les fonctions et classes de la bibliothèque (`TronWindow`, `run_live`, `log_frames`, `ComparisonSession.frame`…) gardent `theme="classic"` par défaut, donc leurs sorties sont inchangées.
- Un test existant vérifiait les dimensions d'une image de `tron compare --png` : il fixe désormais `--theme classic` (le néon ajoute un liseré). Aucun test n'a été supprimé.
- Fenêtre classique : marge portée à `max(14, 0,75 · case)` pixels et hauteur minimale à 640 pixels, pour que la bascule de thème n'exige pas de redimensionner la fenêtre.

## 4. Validation

- 441 tests réussis, 1 ignoré (statsmodels absent), 4 tests de charge désélectionnés (réussis à part) ; ruff, format et mypy sans erreur.
- Couverture estimée des nouveaux modules : 95 à 99 % des lignes.
- La fenêtre n'a **pas** pu être ouverte ici (Arcade non installé, pas d'écran). Elle est testée avec un bouchon d'API (`tests/arcade_stub.py`) et un « Arcade logiciel » qui dessine réellement ses appels (`tests/arcade_preview.py`, aperçus simulés). Le bloom n'est testé qu'avec un contexte OpenGL simulé. La procédure de `docs/validation-gui.md` (§ 7) est à dérouler sur un poste graphique.
