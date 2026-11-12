# Algorithmes et choix de conception

Notations : n joueurs, v le vecteur de valeur d'une feuille, W la matrice de préférences sociales, T(p) l'équipe du joueur p.

## 1. Valeur des feuilles

Tous les moteurs partagent le même codage (`tron.eval.valuation.Valuation`) :

| Situation | v(i) |
|---|---|
| joueur vivant, état non terminal | h(i), part de territoire de Voronoi (tête comprise, donc > 0) |
| joueur éliminé au demi-coup r (compté depuis la racine) | ε·r/T, dans [0, ε[ |
| état terminal, membre de l'équipe gagnante (même éliminé) | 1 + μ·(1 − r/T), où r est le demi-coup de la fin |

Constantes : μ = 10⁻³, T = 1024 et ε = min(μ/(4nT), 1/(4WH)).

Propriétés garanties :

- **Positivité** : toutes les composantes sont ≥ 0, ce qui est la borne inférieure exigée par l'élagage superficiel.
- **Dominance** :
  - une victoire vaut plus que 1, donc plus que toute part de territoire ;
  - une part de territoire vaut au moins 1/(WH) ;
  - une élimination vaut moins que ε, donc moins que toute part de territoire.
- **Préférences temporelles** : une victoire rapide vaut plus, une défaite tardive vaut plus. Les échelles sont emboîtées : les écarts entre défaites sont
  très inférieurs au bonus de victoire, lui-même inférieur aux écarts entre parts de territoire.
- **Bornes de somme** :
  - deux joueurs d'équipes différentes : v(p) + v(q) ≤ 1 + μ + ε ;
  - total des composantes : Σv ≤ S = T_max·(1 + μ) + n·ε.

Pourquoi pas ±∞ ou ±1000 ? Une composante négative casse la borne inférieure de l'élagage de Korf, et une somme non bornée rend la borne de somme inopérante.
Le codage ci-dessus garde l'élagage à la fois valide et efficace.

## 2. Moteurs de recherche (§6.3)

| Moteur | Utilisé par | Principe |
|---|---|---|
| `alphabeta` | Paranoid, SOS réductible à deux camps | f = Σ c_j·v(j). Le camp de la racine maximise, l'autre minimise. Fail-soft, coupure si α ≥ β |
| `maxn` | MaxN, SOS général | le joueur p retient le fils qui maximise W[p]·v ; le vecteur brut v est remonté |
| `negamax` | référence à deux joueurs | f_p = v(p) − v(q), avec négation à chaque niveau |

- Un joueur sans coup légal joue `PASS` : il est éliminé et ce demi-coup compte dans la profondeur.
- Dans Paranoid, la recherche s'arrête dès que le camp racine est entièrement éliminé : sa valeur est alors figée, la coupure est donc exacte.
- **Départage dans l'arbre** : à valeur égale, la direction de plus petit indice (N < E < S < O) l'emporte. En MaxN, deux fils de même valeur pour le joueur au trait
  peuvent différer pour les autres joueurs et changer la valeur racine. Une règle canonique rend donc le résultat indépendant de l'ordre d'exploration et de l'élagage.
- **Départage à la racine** : la recherche renvoie tous les coups ex æquo à 10⁻¹² près. L'alpha-bêta les détecte exactement grâce à une fenêtre racine (meilleur − 10⁻¹², +∞). L'agent choisit ensuite au hasard (graine) ou dans l'ordre N, E, S, O.
- **SOS** : Wilson, Zuckerman et Nau appliquent la transformation linéaire aux feuilles et remontent le vecteur transformé. Remonter le vecteur brut et calculer W[p]·v à la volée sélectionne la même feuille. Le vecteur brut permet en plus de journaliser h et de borner l'élagage sur le simplexe.

## 3. Élagages

**Alpha-bêta (deux camps).** Il est exact : les tests vérifient la même valeur racine et les mêmes coups ex æquo qu'un minimax sans élagage, sur des positions aléatoires.

**Élagage superficiel de Korf (MaxN, W = I).** Au nœud N du joueur p, de parent q :

- x est la meilleure valeur déjà acquise par q chez les frères de N ;
- y est la meilleure valeur de p parmi les fils déjà explorés de N.

Comme v(p) + v(q) ≤ B_pq, le parent n'obtiendra pas mieux que B_pq − y en N. On coupe les fils restants de N si B_pq − y < x. L'inégalité est stricte, pour ne jamais couper un
fils qui pourrait être ex æquo.

L'élagage profond est invalide en MaxN : la valeur d'un petit-enfant ne peut pas devenir la valeur racine, mais elle peut changer le choix d'un joueur intermédiaire, et donc la
valeur racine (Korf 1991 ; Sturtevant et Korf 2000).

**Élagage superficiel généralisé (W quelconque, dérivation propre au projet).** Toutes les feuilles vérifient v ≥ 0 et Σv ≤ S. On calcule donc

U(q, p, y) = max { W[q]·v : v ≥ 0, Σv ≤ S, W[p]·v ≥ y }

et l'on coupe si U < x.

- Calcul : l'optimum est atteint en un sommet du polytope, soit 0, S·e_j, ou l'intersection de l'hyperplan W[p]·v = y avec une arête du simplexe, d'où une
  énumération en O(n²).
- U(y) est concave et linéaire par morceaux, avec des points de rupture en 0 et en S·W[p][j]. Il est précalculé une fois par couple (q, p), puis évalué par interpolation.
- Avec W = I, on retrouve la règle de Korf.
- Validation : la borne est comparée à `scipy.optimize.linprog`, et la recherche élaguée est comparée à la recherche complète.

**Efficacité.** La règle de Korf exige x + y > 1 + μ. Elle coupe souvent à deux joueurs,
rarement au-delà, ce qui est cohérent avec l'absence de gain asymptotique en cas moyen
établie par Korf. Avec une matrice de type Paranoid parcourue par le moteur vectoriel, la
borne généralisée retrouve l'essentiel des coupes alpha-bêta superficielles.

## 4. Choix automatique du moteur

1. **W de rang 1 sur les joueurs vivants** : chaque ligne vaut ±a_p fois la ligne racine. Tous maximisent ou minimisent le même scalaire, et l'alpha-bêta est exact. Ce cas couvre Paranoid (c = e_r) et l'équipe paranoïaque (c = 1_T).
2. **Deux équipes avec α = 1** : les lignes valent 1_A et 1_B. La réduction utilise f = V_A − V_B.
   - Elle est exacte quand V_A + V_B est constant, c'est-à-dire sur les feuilles heuristiques sans joueur éliminé (Σh = 1, y compris en mode neutre, puisque h est normalisé sur les cases attribuées).
   - Ailleurs, elle préserve l'ordre victoire > heuristique > défaite, mais les écarts entre éliminations peuvent pondérer légèrement autrement qu'en SOS strict.
   - `engine: maxn` force la sémantique SOS exacte.
   - Un test vérifie l'égalité des choix dans le cas exact.
3. **Sinon** : MaxN vectoriel avec W.

À deux joueurs, W = I correspond aussi à deux équipes d'un joueur : l'alpha-bêta est retenu, avec les mêmes décisions que MaxN.

## 5. Confrontation avec SOS (Wilson, Zuckerman et Nau, AAMAS 2011)

| Point | SOS (article) | Ce projet |
|---|---|---|
| Poids | c_ij ∈ [−1, 1] | idem ; le profil `custom` le vérifie |
| Poids sur soi | libre (toute diagonale > 0 redonne Max-n) | 1, sauf profil Paranoid (0 pour p ≠ r) |
| Normalisation | aucune | aucune ; h est déjà sur le simplexe |
| Remontée | vecteur transformé c·eval | vecteur brut, même choix de feuille |
| Égalités | non traitées (premier fils implicite) | règle canonique N < E < S < O ; F3.7 à la racine |
| Élagage | immédiat et superficiel (profond si une seule colonne non nulle) | Korf (W = I), borne généralisée (W quelconque), alpha-bêta pour les cas à deux camps |
| Apprentissage de la matrice | variante avec apprentissage | hors périmètre (alliances statiques) |
| Domaine | Quoridor à 4 joueurs | Tron, territoires de Voronoi |

Le profil Paranoid correspond exactement à la propriété 2 de l'article (c_ii = 1, c_ji = −1, 0 ailleurs), et la matrice identité donne Max-n (propriété 1).

## 6. Territoire de Voronoi

- **Parcours.** Un seul parcours en largeur multi-sources, les têtes étant insérées dans l'ordre de jeu à partir du joueur au trait. Chaque case reçoit le masque binaire des
  joueurs qui l'atteignent à distance minimale : les masques d'une couche sont définitifs avant que la couche ne soit dépilée.
- **Attribution.**
  - Mode `order` : en cas d'égalité, la case revient au premier joueur du masque dans l'ordre de jeu, ce qui est cohérent avec le tour par tour.
  - Mode `neutral` : une case disputée ne revient à personne.
  - Dans les deux modes, les parts sont normalisées sur les cases attribuées.
- **Séparation** (option `separation`). Les régions de deux joueurs sont reliées si une case partagée ou une arête frontière les joint ; une fermeture transitive est calculée sur au plus huit joueurs. Un joueur isolé est évalué par la borne de parité en damier : au plus 2·min(r, s) + [r > s] coups, où r compte les cases de la couleur opposée à sa tête et s celles de sa couleur.
- **Validation.** Le noyau est comparé à une implémentation naïve (un parcours en largeur par joueur) sur des positions aléatoires, dans les deux modes et avec la séparation.

## 7. Optimisations

- **Tri des coups** : coup mémorisé par la table de transposition, puis heuristique de l'historique (bonus profondeur² sur les coupures), puis N, E, S, O.
- **Approfondissement itératif** : les coups racine sont triés selon l'itération précédente. Avec une limite de temps, l'itération interrompue est abandonnée et l'itération complète précédente est conservée.
- **Table de transposition** :
  - clé de Zobrist : cases occupées, têtes des joueurs vivants, ensemble des vivants, joueur au trait ;
  - une table par agent, donc par matrice W ;
  - seul le meilleur coup est mémorisé, pour le tri, ce qui préserve l'exactitude. Les transpositions vraies sont rares en Tron : chaque trajet laisse un mur différent.

## 8. MCTS multijoueur à utilité vectorielle (P04)

L'agent `mcts` explore un grand nombre de continuations au lieu d'une recherche exhaustive à profondeur fixe.

- **Nœud** : coup entrant, joueur au trait, nombre de visites N, somme des vecteurs de récompense (une composante par joueur).
- **Sélection** : au nœud où p a le trait, l'enfant c maximise U_p(c) + C·√(ln N(nœud)/N(c)), avec U_p(c) = W[p]·(somme(c)/N(c)). Par défaut W = I (profil `maxn`) ; tout profil social est accepté (`profile`, `alpha`, `matrix`). Comme pour MaxN, le vecteur brut est remonté et l'utilité calculée à la sélection.
- **Expansion** : un coup non essayé, dans un ordre tiré avec la graine. Un joueur sans coup légal joue `PASS` et est éliminé, exactement comme dans les moteurs exacts.
- **Simulation** : `random` (coup sûr au hasard) ou `greedy` (coup qui maximise le nombre de cases libres voisines de la case d'arrivée, départage au hasard ; c'est une politique de mobilité, beaucoup moins coûteuse que l'agent glouton à Voronoi). Arrêt en fin de partie ou après `rollout_depth` demi-coups.
- **Évaluation** : `Valuation.vector(état, demi-coup racine)`, donc le même codage que les moteurs exacts (victoire > part de territoire > défaite, toutes composantes ≥ 0).
- **Choix** : l'enfant de la racine le plus visité ; à égalité, la règle de départage de l'agent.
- **Restauration** : la recherche travaille sur une copie de l'état ; chaque simulation défait ses coups dans un bloc `try/finally`.
- **Reproductibilité** : tout l'aléa vient du flux de l'agent (`make_rng(graine, STREAM_AGENT, joueur)`), une graine Python dérivée par décision. `time_limit` est facultatif et rend les parties dépendantes de la machine : à proscrire dans les campagnes.

Ce n'est pas un UCT à deux joueurs : chaque nœud est évalué avec l'utilité du joueur qui y a le trait. Aucun élagage alpha-bêta ou de Korf n'est appliqué (aucune borne n'est prouvée pour ce parcours stochastique).

Statistiques : `nodes` = nœuds créés (racine comprise), `leaves` = simulations, `depth` = profondeur maximale de l'arbre, `ties` = coups racine à égalité de visites, `value` = utilité moyenne du coup choisi, `pv` = suite des enfants les plus visités. Les résultats CSV exposent `leaves_p<i>` (simulations ou feuilles cumulées) et `meandepth_p<i>`.

**Première comparaison** (Python pur, indicative : effectifs faibles). Duels sur 10 × 10, 12 parties : MCTS à 200 simulations gagne 12/12 contre l'agent aléatoire et 6/12 (simulations `greedy`) ou 8/12 (`random`) contre le glouton à Voronoi. Plan `mcts_comparison.yaml` réduit à 4 parties par condition (160 parties, grille 12 × 12, solitaire contre 1 ou 2 adversaires) :

| Solitaire | Taux de victoire moyen | Temps par décision | Nœuds par décision | Simulations ou feuilles par décision | Profondeur moyenne |
|---|---:|---:|---:|---:|---:|
| `mcts` (300 simulations) | 25 % | 14 ms | 203 | 220 | 7,3 |
| `maxn` (profondeur 3) | 56 % | 1 ms | 18 | 11 | 2,3 |
| `paranoid` (profondeur 3) | 44 % | 1 ms | 20 | 12 | 2,3 |
| `social` (profondeur 3) | 44 % | 1 ms | 20 | 12 | 2,3 |
| `greedy` | 31 % | < 1 ms | 3 | 2 | 1,0 |

Avec 300 simulations et une politique de simulation simple, MCTS reste ici en retrait des recherches exactes de profondeur 3, pour un coût dix fois supérieur. `social` (profil Équipe, α = 1) donne exactement les décisions de `paranoid` pour un joueur seul, comme prévu par la réduction à deux équipes (§4). Les plans `mcts_comparison.yaml` (20 parties par condition) et `mcts_simulations.yaml` (courbe selon le nombre de simulations) mesurent ces écarts avec des graines appariées ; ces chiffres sont à confirmer à pleine taille.

## 9. Méthodologie expérimentale (§7)

- **Graines appariées.** La graine de la partie g dépend de la graine du plan et de g, pas de la condition. Avec `seats: cyclic`, la graine d'indice g div n est jouée n fois, une
  fois avec chaque premier joueur. Les sièges tournent ainsi complètement, ce qui neutralise l'avantage du premier coup sur les cases disputées.
- **Profondeur.** Elle s'exprime en plies (cahier des charges) ou en tours complets (`depth_unit: rounds`, profondeur = tours × joueurs vivants). En plies, un solitaire
  face à 5 adversaires à profondeur 6 ne voit qu'un seul de ses propres coups. L'analyse fournit aussi `solo_depth_rounds`.
- **Niveau du hasard.** Il vaut |T|/n, soit 1/(k+1) pour le solitaire, et non 50 %.
- **Statistiques.**
  - Intervalle de Wilson à 95 %.
  - Test exact de McNemar entre profondeurs consécutives sur les mêmes parties, avec correction de Holm.
  - Régression logistique, erreurs standard groupées par graine.
- **Puissance.** Avec 100 parties, l'intervalle mesure environ ±10 points dans le pire cas. Détecter un écart de 10 points avec une puissance de 80 % demande environ 400 parties par condition indépendante ; l'appariement réduit ce besoin.

## Références

- Negamax et optimisations de l'alpha-bêta : <https://www.lamsade.dauphine.fr/~cazenave/papers/berder00.pdf>
- MaxN et Paranoid (Sturtevant et Korf, AAAI 2000) : <https://cdn.aaai.org/AAAI/2000/AAAI00-031.pdf>
- Élagage alpha-bêta multi-joueurs (Korf 1991) : <https://faculty.cc.gatech.edu/~thad/6601-gradAI-fall2015/Korf_Multi-player-Alpha-beta-Pruning.pdf>
- SOS (Wilson, Zuckerman et Nau, AAMAS 2011) : <https://www.cs.umd.edu/~nau/papers/wilson2011modeling.pdf>
- Post-mortem du Google AI Challenge 2010 (Tron) : <https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html>
