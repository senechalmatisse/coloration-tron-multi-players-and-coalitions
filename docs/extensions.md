# Extensions : agents et heuristiques externes (P12)

Un autre paquet Python peut ajouter un agent ou une heuristique **sans modifier ce dépôt** : il déclare ses classes par des *entry points*, Tron les découvre au chargement et elles s'utilisent par leur nom dans n'importe quelle configuration YAML. Un exemple complet se trouve dans `examples/plugin_exemple/`.

```bash
pip install -e examples/plugin_exemple
tron plugins list
tron play examples/plugin_exemple/partie.yaml --no-gui
```

## 1. Contrat d'un agent

Sous-classe de `tron.agents.Agent` (ou de `tron.api.Agent`) :

```python
from tron.agents import Agent, DecisionDetails, SearchStats
from tron.core.config import ConfigError

class PrudentAgent(Agent):
    def __init__(self, player, config, *, bonus_centre: float = 0.0, **kw):
        super().__init__(player, config, **kw)        # depth, seed, heuristic, tie_break…
        if bonus_centre < 0:
            raise ConfigError(f"bonus_centre doit être ≥ 0 : {bonus_centre}")
        self.bonus_centre = bonus_centre

    def _decide(self, state, moves):                  # moves : coups légaux, jamais vide
        scores = {}
        for m in moves:
            state.apply(m)
            try:
                scores[m] = len(state.legal_moves(self.player))
            finally:
                state.undo()                          # l'état reçu est rendu intact
        best = max(scores.values())
        move = self.pick([m for m in moves if scores[m] == best])
        self._details = DecisionDetails(scores=tuple(scores.items()))   # facultatif (P07)
        return move, SearchStats(nodes=len(moves) + 1, depth=1, engine=self.name)
```

- **Constructeur** : `(player, config, **paramètres)` ; les paramètres viennent de la ligne du joueur dans le YAML (`{agent: prudent, bonus_centre: 0.1}`). Transmettre `**kw` à `super().__init__`. Une valeur invalide lève `ConfigError` avec un message explicite ; un paramètre inconnu (`TypeError`) est converti en `ConfigError` par `make_agent`.
- **`_decide(state, moves)`** renvoie `(coup, SearchStats)`. `state` est l'état de la partie : le rendre **intact** (`apply`/`undo` protégés par `try/finally`), ou travailler sur `state.copy()`.
- **Aléa** : uniquement `self.rng` (dérivé de la graine de la partie par `tron.core.rng`) et `self.pick(égalités)` ; jamais `random` ou `time`, pour que les parties restent reproductibles.
- **Crochets facultatifs** : `observe_move` (voir les coups joués, P05), `snapshot_state`/`restore_state` (sauvegardes, P08 : par défaut l'état du générateur aléatoire ; un agent qui garde une mémoire entre ses décisions doit l'y ajouter pour qu'une reprise soit exacte), `_decide_with_context` et `_diplomacy` (diplomatie, P03), `describe()` (nom affiché).

## 2. Contrat d'une heuristique

Sous-classe de `tron.eval.Heuristic`, constructeur `(width, height, n_players, **paramètres)` :

```python
from tron.eval import Heuristic

class CenterHeuristic(Heuristic):
    def shares(self, state):
        ...  # une part par joueur
```

`shares(state)` renvoie une part par joueur : **finie, ≥ 0, nulle pour un joueur éliminé, de somme 1 sur les joueurs vivants** (contrat de `Valuation`). Elle ne modifie pas `state` et ne dépend que de lui (pas d'aléa). Les paramètres viennent du YAML (`heuristic: {name: centre, …}`) ; un paramètre inconnu donne une erreur explicite.

## 3. Déclaration et découverte

Dans le `pyproject.toml` du paquet d'extension :

```toml
[project.entry-points."tron.agents"]
prudent = "tron_plugin_exemple:PrudentAgent"

[project.entry-points."tron.heuristics"]
centre = "tron_plugin_exemple:CenterHeuristic"
```

Le nom de l'entry point est le nom utilisé dans les configurations ; les classes n'ont pas besoin du décorateur `register_agent`. Les extensions sont chargées à l'import de `tron.agents` (groupe `tron.agents`) et de `tron.eval` (groupe `tron.heuristics`), **une seule fois** : le chargement est idempotent (`tron.plugins.load_plugins`, `force=True` pour recharger).

| Situation | État | Effet |
|---|---|---|
| chargée et enregistrée | `loaded` | utilisable par son nom |
| nom d'un élément intégré (ou déjà enregistré par le programme) | `conflict` | ignorée : l'élément intégré reste prioritaire |
| classe qui, à l'import de son module, remplace un nom existant | `conflict` | ignorée, le nom d'origine est restauré |
| même nom déclaré par deux paquets | `conflict` pour le second | la première dans l'ordre (nom, paquet, cible) est chargée |
| erreur à l'import | `error` | ignorée ; type et message de l'exception conservés |
| cible qui n'est pas une sous-classe de `Agent` / `Heuristic` | `invalid` | ignorée |

Chaque extension ignorée émet un `tron.plugins.PluginWarning` sans empêcher l'import de Tron. `tron plugins list` affiche le groupe, le nom, le paquet (nom et version), la classe, l'état et le détail de chaque extension (`--all` : éléments intégrés compris) ; son code de sortie vaut 1 si une extension a été ignorée.

**Reproductibilité.** `TRON_NO_PLUGINS=1` désactive la découverte (campagnes de référence). Le manifeste d'une campagne (`manifest.json`, `tron run`) liste les extensions chargées avec leur paquet, leur version et leur cible. Une configuration qui nomme un agent d'extension non installé échoue avec « agent inconnu ».

**Sécurité.** Aucun code n'est jamais importé depuis un chemin écrit dans une configuration : `agent: mon_paquet.module:Classe` est refusé avec un message qui renvoie vers la déclaration d'un entry point. Seuls les noms enregistrés (intégrés ou déclarés par un paquet installé) sont acceptés.

## 4. Registre historique

`register_agent` et `register_heuristic` restent utilisables pour un agent défini dans le programme lui-même (voir `examples/agent_personnalise.py`) ; les noms ainsi enregistrés sont prioritaires sur les extensions du même nom.

Les tests (`tests/test_plugins.py`) installent des paquets simulés (dossiers `.dist-info` sur `sys.path`) et vérifient le chargement, l'idempotence, les conflits, les erreurs d'import, les cibles invalides, la désactivation, le refus des chemins de module et la compatibilité du registre historique.
