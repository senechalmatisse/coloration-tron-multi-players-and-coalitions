# Exemples produits par la chaîne complète

Les fichiers de ce dossier ont été générés en Python pur (sans Numba), sans retouche :

| Fichier | Commande |
|---|---|
| `solo_vs_3.gif` | `tron play configs/games/solo_vs_3_20x20.yaml --no-gui --gif docs/exemples/solo_vs_3.gif --voronoi --cell 14 --fps 12` |
| `pilote_*.png`, `pilote_*.csv`, `pilote_regression.txt` | `tron run configs/plans/pilote.yaml --workers 2`, puis `tron analyze results/pilote/results.csv` |

Le plan pilote compte 1 152 parties et a été exécuté en une minute sur deux cœurs. C'est une version réduite de l'étude A :

- solitaire Paranoid de profondeur 1 à 6 ;
- coalition coordonnée (profil Équipe, α = 1) de profondeur 1 ou 2 ;
- k de 1 à 3 ;
- grilles 10 × 10 et 14 × 14 ;
- 24 parties par condition.

Ce plan sert à valider la chaîne et à calibrer la campagne. Ses effectifs sont trop faibles pour conclure. On y lit toutefois déjà :

- des gains décroissants de la profondeur en duel : comparaisons appariées significatives de 1 à 2 et de 2 à 4 plies, pas de 4 à 6 (`pilote_paired.csv`) ;
- une coalition coordonnée de deux joueurs ou plus qui l'emporte presque toujours sur ces petites grilles ;
- un coût de calcul multiplié par environ 3 à chaque pli supplémentaire, au plus.
