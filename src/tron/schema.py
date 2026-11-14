"""Schéma des fichiers de résultats, partagé par ``sim`` (écriture) et ``analysis`` (lecture).

Module neutre, sans dépendance : l'analyse d'un CSV n'importe ni le moteur, ni Numba, ni l'affichage.
"""

MAX_PLAYERS = 8
MAX_TEAMS = 8

METRICS_VERSION = 2
"""Version du schéma des métriques de partie (colonne ``metrics_version`` du CSV).

1 : métriques de coopération par joueur et par équipe ;
2 : ajout des colonnes ``dipl_*`` de la diplomatie tactique.
"""

TEAM_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("tsurv", "team_survival_plies"),
    ("trank", "team_rank"),
    ("tdist", "intra_team_distance_mean"),
    ("tcoll", "ally_collisions"),
    ("tcoop", "cooperation_proxy"),
    ("tterr", "territory_share_mean"),
    ("tgap", "collective_vs_individual_gap"),
)
"""Préfixe de colonne CSV (``<préfixe>_t<i>``) et champ de ``MatchMetrics`` par équipe."""

DIPLOMACY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dipl_enabled", "diplomacy_enabled"),
    ("dipl_proposals", "proposals"),
    ("dipl_formed", "alliances_formed"),
    ("dipl_breaks", "alliance_breaks"),
    ("dipl_betrayals", "betrayals"),
    ("dipl_rejected", "rejected_intents"),
    ("dipl_duration", "alliance_duration_mean"),
)
"""Colonnes CSV de la diplomatie et champ correspondant de ``MatchMetrics``."""
