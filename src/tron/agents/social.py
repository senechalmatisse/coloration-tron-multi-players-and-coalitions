"""Préférences sociales (extension inspirée de SOS, F4.1 à F4.6).

Matrice W (n × n) : ``W[p][j]`` est le poids que le joueur p accorde à la
composante v(j). À chaque nœud, p retient le fils qui maximise W[p]·v. Wilson,
Zuckerman et Nau (AAMAS 2011) appliquent c·eval aux feuilles et remontent le
vecteur transformé ; remonter le vecteur brut et calculer W[p]·v à la volée
choisit la même feuille (transformation linéaire), et permet de calculer les
bornes d'élagage sur le simplexe.

Profils générés à partir des équipes (F4.3) :

* ``maxn`` : identité ;
* ``paranoid`` de racine r : W[r][r] = 1, W[p][r] = −1 pour p ≠ r, 0 ailleurs
  (propriété 2 de SOS) ;
* ``team`` : W[p][p] = 1, W[p][j] = α pour chaque coéquipier j, 0 ailleurs ;
* ``team_paranoid`` : les membres de l'équipe racine T ont la ligne 1_T, les
  autres la ligne −1_T ;
* ``dynamic_team`` (P03) : comme ``team`` pour la matrice de base W_base, plus
  les alliances tactiques de la diplomatie (:func:`dynamic_matrix`) et la
  politique diplomatique :func:`tactical_intent`.

Choix du moteur (F4.4) : alpha-bêta complet quand W se ramène à deux camps, MaxN
vectoriel sinon. Deux cas sont reconnus :

1. W de rang 1 sur les joueurs vivants, chaque ligne valant ±a_p fois la ligne
   racine. Chacun maximise ou minimise le même scalaire, ce qui couvre Paranoid
   et l'équipe paranoïaque. La réduction est exacte.
2. Deux équipes avec α = 1 : les lignes valent 1_A et 1_B. La réduction utilise
   f = V_A − V_B et suppose V_A + V_B constant. C'est exact sur les feuilles
   heuristiques sans joueur éliminé (Σ h = 1) ; ailleurs, l'ordre
   victoire > heuristique > défaite reste préservé.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

from tron.agents.context import DiplomacyIntent, DiplomacySnapshot

PROFILES = ("maxn", "paranoid", "team", "team_paranoid", "custom", "dynamic_team")
Matrix = list[list[float]]


def social_matrix(
    profile: str,
    team: Sequence[int],
    root: int,
    alpha: float = 1.0,
    matrix: Sequence[Sequence[float]] | None = None,
) -> Matrix:
    """Construit W pour ``profile`` à partir des équipes (indice d'équipe par joueur)."""
    n = len(team)
    if profile == "maxn":
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if profile == "paranoid":
        w = [[0.0] * n for _ in range(n)]
        for p in range(n):
            w[p][root] = 1.0 if p == root else -1.0
        return w
    if profile in ("team", "dynamic_team"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha doit être dans [0, 1]")
        return [
            [1.0 if i == j else (alpha if team[i] == team[j] else 0.0) for j in range(n)]
            for i in range(n)
        ]
    if profile == "team_paranoid":
        t = team[root]
        return [
            [(1.0 if team[i] == t else -1.0) if team[j] == t else 0.0 for j in range(n)]
            for i in range(n)
        ]
    if profile == "custom":
        if matrix is None:
            raise ValueError("le profil 'custom' exige une matrice")
        w = [[float(x) for x in row] for row in matrix]
        if len(w) != n or any(len(row) != n for row in w):
            raise ValueError(f"la matrice doit être {n}x{n}")
        if any(abs(x) > 1.0 for row in w for x in row):
            raise ValueError("les poids doivent être dans [-1, 1] (SOS)")
        return w
    raise ValueError(f"profil inconnu : {profile} (attendu : {PROFILES})")


def dynamic_matrix(
    base: Sequence[Sequence[float]],
    snapshot: DiplomacySnapshot,
    alliance_weight: float,
    betrayal_penalty: float = 0.0,
) -> Matrix:
    """Matrice W_t d'une décision, à partir de W_base et de l'état diplomatique (P03).

    Ordre d'application, sans valeur cachée :

    1. W_t = W_base ;
    2. pour chaque alliance active (a, b) : W_t[a][b] = max(W_t[a][b], alliance_weight),
       et de même pour W_t[b][a] (statut d'alliance symétrique) ;
    3. si ``betrayal_penalty`` > 0 : W_t[p][q] -= betrayal_penalty × (trahisons de q
       envers p) ;
    4. bornage de chaque poids dans [−1, 1] (contrainte de SOS).

    W_t est constante pendant une recherche : les agents ne simulent pas les
    actions diplomatiques futures dans leur arbre (limite de la V1).
    """
    if not 0.0 <= alliance_weight <= 1.0:
        raise ValueError("alliance_weight doit être dans [0, 1]")
    if not 0.0 <= betrayal_penalty <= 1.0:
        raise ValueError("betrayal_penalty doit être dans [0, 1]")
    w = [[float(x) for x in row] for row in base]
    for a, b in snapshot.active_pairs:
        w[a][b] = max(w[a][b], alliance_weight)
        w[b][a] = max(w[b][a], alliance_weight)
    if betrayal_penalty > 0.0:
        for p, row in enumerate(w):
            for q in range(len(row)):
                k = snapshot.betrayals[q][p]
                if k and q != p:
                    row[q] -= betrayal_penalty * k
    return [[min(1.0, max(-1.0, x)) for x in row] for row in w]


def tactical_intent(
    player: int,
    team: Sequence[int],
    alive: Sequence[bool],
    shares: Sequence[float],
    snapshot: DiplomacySnapshot,
    max_allies: int,
    break_margin: float = 0.15,
) -> DiplomacyIntent | None:
    """Politique diplomatique simple et déterministe de la V1 (« contre le meneur »).

    L'équipe meneuse est celle dont la somme des parts de territoire des membres
    vivants est la plus grande (à égalité, le plus petit indice). Dans l'ordre :

    1. **rompre** avec un allié de l'équipe meneuse si celle-ci devance l'équipe
       du joueur de plus de ``break_margin`` ;
    2. **accepter** (proposer en retour) la proposition en attente du proposant
       de plus petit indice, si ce proposant n'est pas dans l'équipe meneuse
       (sauf si c'est l'équipe du joueur qui mène) ;
    3. **proposer**, si l'équipe du joueur ne mène pas, au joueur le plus fort
       hors de l'équipe meneuse et de sa propre équipe.

    Les coéquipiers permanents ne sont jamais candidats (ils sont déjà valorisés
    par le profil). Aucune intention si le plafond d'alliés est atteint.
    """
    n = len(team)
    my_team = team[player]
    team_share: dict[int, float] = {}
    for q in range(n):
        if alive[q]:
            team_share[team[q]] = team_share.get(team[q], 0.0) + shares[q]
    if not team_share:
        return None
    leader = min(team_share, key=lambda t: (-team_share[t], t))
    allies = snapshot.allies(player)
    my_share = team_share.get(my_team, 0.0)
    for q in allies:
        tq = team[q]
        if tq == leader and tq != my_team and team_share[tq] - my_share > break_margin:
            return DiplomacyIntent("break", q)
    if len(allies) >= max_allies:
        return None

    def eligible(q: int) -> bool:
        return (
            q != player
            and alive[q]
            and team[q] != my_team
            and not snapshot.are_allied(player, q)
            and not snapshot.in_cooldown(player, q)
            and not snapshot.has_pending(player, q)
        )

    for q in snapshot.proposals_to(player):
        if eligible(q) and (team[q] != leader or leader == my_team):
            return DiplomacyIntent("propose", q)
    if leader == my_team:
        return None
    candidates = [q for q in range(n) if eligible(q) and team[q] != leader]
    if not candidates:
        return None
    return DiplomacyIntent("propose", max(candidates, key=lambda q: (shares[q], -q)))


def sparse_rows(w: Matrix) -> list[list[tuple[int, float]] | None]:
    """Lignes creuses de W ; ``None`` pour une ligne égale à e_p (utilité = v(p))."""
    rows: list[list[tuple[int, float]] | None] = []
    for p, row in enumerate(w):
        nz = [(j, x) for j, x in enumerate(row) if x != 0.0]
        rows.append(None if nz == [(p, 1.0)] else nz)
    return rows


def is_identity(w: Matrix) -> bool:
    return all(x == (1.0 if i == j else 0.0) for i, row in enumerate(w) for j, x in enumerate(row))


@dataclass(frozen=True, slots=True)
class TwoCamp:
    """Réduction à deux camps : f = Σ c_j·v(j), maximisé si sign = +1, minimisé sinon."""

    coeffs: tuple[tuple[int, float], ...]
    signs: tuple[int, ...]
    camp: tuple[int, ...] | None
    """Joueurs dont dépend f : si tous sont éliminés, f est figé (None : jamais)."""
    kind: str


def two_camp_reduction(
    w: Matrix, team: Sequence[int], root: int, alive: Sequence[bool]
) -> TwoCamp | None:
    """Détecte si W se ramène à deux camps pour les joueurs vivants (F4.4)."""
    n = len(w)
    r = w[root]
    if any(x != 0.0 for x in r):
        signs = [1] * n
        ok = True
        for p in range(n):
            if not alive[p]:
                continue
            row = w[p]
            j0 = next(j for j in range(n) if r[j] != 0.0)
            lam = row[j0] / r[j0]
            if lam == 0.0 or any(abs(row[j] - lam * r[j]) > 1e-12 for j in range(n)):
                ok = False
                break
            signs[p] = 1 if lam > 0 else -1
        if ok and signs[root] > 0:
            coeffs = tuple((j, r[j]) for j in range(n) if r[j] != 0.0)
            players = tuple(j for j, _ in coeffs)
            kind = "paranoid" if len(coeffs) == 1 else "rank1"
            return TwoCamp(coeffs, tuple(signs), players if len(players) < n else None, kind)
    if len(set(team)) == 2:
        expected = [[1.0 if team[i] == team[j] else 0.0 for j in range(n)] for i in range(n)]
        if all(w[p] == expected[p] for p in range(n) if alive[p]):
            a = team[root]
            team_coeffs = tuple((j, 1.0 if team[j] == a else -1.0) for j in range(n))
            team_signs = tuple(1 if team[p] == a else -1 for p in range(n))
            return TwoCamp(team_coeffs, team_signs, None, "two_teams")
    return None


# ------------------------------------------------- élagage superficiel généralisé
def lp_upper(a: Sequence[float], b: Sequence[float], s: float, y: float) -> float:
    """max b·v  s.c.  a·v ≥ y,  v ≥ 0,  Σv ≤ s  (énumération des sommets, O(n²)).

    Les sommets du polytope sont les sommets 0 et s·e_j du simplexe qui vérifient
    la contrainte, plus les intersections de l'hyperplan a·v = y avec les arêtes
    [0, s·e_j] et [s·e_j, s·e_k].
    """
    tol = 1e-15
    best = -math.inf
    n = len(a)
    if y <= tol:
        best = 0.0
    for j in range(n):
        if s * a[j] >= y - tol:
            best = max(best, s * b[j])
        if a[j] != 0.0:
            t = y / (s * a[j])
            if -tol <= t <= 1.0 + tol:
                best = max(best, t * s * b[j])
        for k in range(j + 1, n):
            da = a[k] - a[j]
            if da != 0.0:
                t = (y / s - a[j]) / da
                if -tol <= t <= 1.0 + tol:
                    best = max(best, s * (b[j] + t * (b[k] - b[j])))
    return best


class _PiecewiseBound:
    """U(y) concave et linéaire par morceaux, de points de rupture a·sommets."""

    __slots__ = ("us", "ys")

    def __init__(self, a: Sequence[float], b: Sequence[float], s: float) -> None:
        self.ys = sorted({0.0, *(s * x for x in a)})
        self.us = [lp_upper(a, b, s, y) for y in self.ys]

    def __call__(self, y: float) -> float:
        ys = self.ys
        if y <= ys[0]:
            return self.us[0]
        if y > ys[-1]:
            return -math.inf
        i = bisect.bisect_left(ys, y)
        if ys[i] == y:
            return self.us[i]
        y0, y1 = ys[i - 1], ys[i]
        u0, u1 = self.us[i - 1], self.us[i]
        return u0 + (u1 - u0) * (y - y0) / (y1 - y0)


class LinearBound:
    """Borne U(q, p, y) = max{W[q]·v : v ≥ 0, Σv ≤ S, W[p]·v ≥ y}.

    Valide dès que toutes les feuilles vérifient v ≥ 0 et Σv ≤ S, ce que
    garantit le codage de :class:`~tron.eval.valuation.Valuation`. Avec W = I,
    on retrouve la règle de Korf U = S − y.
    """

    def __init__(self, w: Matrix, sum_bound: float) -> None:
        self.w = w
        self.s = sum_bound
        self._cache: dict[tuple[int, int], _PiecewiseBound] = {}

    def __call__(self, q: int, p: int, y: float) -> float:
        f = self._cache.get((q, p))
        if f is None:
            f = _PiecewiseBound(self.w[p], self.w[q], self.s)
            self._cache[(q, p)] = f
        return f(y)
