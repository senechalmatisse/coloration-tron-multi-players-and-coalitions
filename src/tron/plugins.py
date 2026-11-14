"""Extensions installées : agents et heuristiques déclarés par *entry points*.

Un paquet Python ajoute un agent ou une heuristique sans modifier ce dépôt en
déclarant, dans son ``pyproject.toml`` :

.. code-block:: toml

    [project.entry-points."tron.agents"]
    prudent = "tron_plugin_exemple:PrudentAgent"

    [project.entry-points."tron.heuristics"]
    centre = "tron_plugin_exemple:CenterHeuristic"

Les extensions sont découvertes au chargement de ``tron.agents`` et ``tron.eval``
(une seule fois : le chargement est idempotent) puis utilisables par leur nom
dans une configuration YAML (``agent: prudent``, ``heuristic: centre``).

Règles :

* un nom d'agent ou d'heuristique **intégré** reste prioritaire : une extension
  qui le réclame est ignorée (état ``conflict``), de même qu'une extension qui
  redéfinirait un nom intégré à l'import ;
* deux extensions du même nom : la première dans l'ordre (nom, paquet, cible)
  est chargée, l'autre est ignorée (``conflict``) ;
* une erreur d'import ou une cible qui n'est pas une sous-classe de ``Agent``
  ou ``Heuristic`` est signalée (``error`` / ``invalid``) par un
  :class:`PluginWarning`, sans empêcher l'import de Tron ;
* ``TRON_NO_PLUGINS=1`` désactive la découverte (campagnes reproductibles) ;
* aucun code n'est jamais chargé depuis un chemin écrit dans une configuration :
  seuls les noms enregistrés (intégrés ou déclarés par un paquet installé) sont acceptés.

``tron plugins list`` affiche les extensions, leur paquet, leur cible et leur état.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import warnings
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from typing import Any

AGENT_GROUP = "tron.agents"
HEURISTIC_GROUP = "tron.heuristics"
GROUPS = (AGENT_GROUP, HEURISTIC_GROUP)
DISABLE_ENV = "TRON_NO_PLUGINS"
STATUSES = ("loaded", "conflict", "error", "invalid")


class PluginWarning(UserWarning):
    """Extension ignorée : conflit de nom, erreur d'import ou cible invalide."""


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Une extension déclarée et le résultat de son chargement."""

    group: str
    name: str
    target: str
    """Cible ``module:objet`` de l'entry point."""
    distribution: str
    """Paquet qui la déclare (``nom version``), ``inconnu`` si indéterminé."""
    status: str
    detail: str = ""

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"

    def to_dict(self) -> dict[str, str]:
        return {
            "group": self.group,
            "name": self.name,
            "target": self.target,
            "distribution": self.distribution,
            "status": self.status,
            "detail": self.detail,
        }


_LOADED: dict[str, list[PluginInfo]] = {}
"""Résultat du dernier chargement par groupe (idempotence)."""
_OWNED: dict[str, dict[str, type]] = {group: {} for group in GROUPS}
"""Noms enregistrés par des extensions (les autres noms du registre sont intégrés)."""
_HIJACKERS: dict[type, list[str]] = {}
"""Classes qui ont tenté, à l'import de leur module, de remplacer un nom existant."""


def _registry(group: str) -> tuple[MutableMapping[str, Any], type]:
    if group == AGENT_GROUP:
        from tron.agents.base import AGENTS, Agent  # noqa: PLC0415 - évite un cycle

        return AGENTS, Agent
    from tron.eval.base import HEURISTICS, Heuristic  # noqa: PLC0415

    return HEURISTICS, Heuristic


def _distribution(ep: md.EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return "inconnu"
    try:
        return f"{dist.metadata['Name']} {dist.version}"
    except Exception:
        return "inconnu"


def installed_entry_points(group: str) -> list[md.EntryPoint]:
    """Entry points installés du groupe, dans l'ordre (nom, paquet, cible)."""
    eps = md.entry_points(group=group)
    return sorted(eps, key=lambda ep: (ep.name, _distribution(ep), ep.value))


def _warn(info: PluginInfo) -> None:
    warnings.warn(
        f"extension {info.group} « {info.name} » ({info.target}, {info.distribution}) "
        f"ignorée : {info.detail}",
        PluginWarning,
        stacklevel=3,
    )


def load_plugins(
    group: str,
    *,
    entry_points: Iterable[md.EntryPoint] | None = None,
    force: bool = False,
) -> list[PluginInfo]:
    """Charge les extensions du groupe et les enregistre ; renvoie leur bilan.

    Idempotent : un deuxième appel renvoie le bilan du premier sans rien
    réimporter, sauf avec ``force=True`` (les noms enregistrés par des extensions
    sont alors retirés puis rechargés). ``entry_points`` remplace la découverte
    (tests, outils) et implique un rechargement.
    """
    if group not in GROUPS:
        raise ValueError(f"groupe d'extensions inconnu : {group} (attendu : {GROUPS})")
    if group in _LOADED and not force and entry_points is None:
        return _LOADED[group]
    registry, base = _registry(group)
    owned = _OWNED[group]
    for name, cls in list(owned.items()):  # rechargement : retirer les extensions
        if registry.get(name) is cls:
            del registry[name]
    owned.clear()
    if entry_points is None and os.environ.get(DISABLE_ENV, "") not in ("", "0"):
        _LOADED[group] = []
        return []
    eps = installed_entry_points(group) if entry_points is None else list(entry_points)
    infos: list[PluginInfo] = []
    for ep in eps:
        dist = _distribution(ep)

        def record(
            status: str, detail: str = "", *, ep: md.EntryPoint = ep, dist: str = dist
        ) -> None:
            info = PluginInfo(group, ep.name, ep.value, dist, status, detail)
            infos.append(info)
            if status != "loaded":
                _warn(info)

        if ep.name in owned:
            record("conflict", f"nom déjà pris par l'extension {owned[ep.name].__module__}")
            continue
        if ep.name in registry:
            record("conflict", "nom déjà enregistré (élément intégré à Tron ou au programme)")
            continue
        before = dict(registry)
        try:
            obj = ep.load()
        except Exception as exc:
            registry.clear()
            registry.update(before)
            record("error", f"{type(exc).__name__} : {exc}")
            continue
        for n, cls in before.items():
            new = registry.get(n)
            if new is not cls and isinstance(new, type):
                _HIJACKERS.setdefault(new, []).append(n)
        registry.clear()
        registry.update(before)  # seuls les noms d'entry points sont enregistrés
        if isinstance(obj, type) and obj in _HIJACKERS:
            record("conflict", f"la classe redéfinit à son import : {sorted(_HIJACKERS[obj])}")
            continue
        if not (isinstance(obj, type) and issubclass(obj, base)):
            record("invalid", f"{ep.value} n'est pas une sous-classe de {base.__name__}")
            continue
        setattr(obj, "name", ep.name)  # noqa: B010 - attribut de classe des registres
        registry[ep.name] = obj
        owned[ep.name] = obj
        record("loaded")
    _LOADED[group] = infos
    return infos


def plugin_report() -> list[PluginInfo]:
    """Bilan des deux groupes (chargés si ce n'est pas déjà fait)."""
    import tron.agents  # noqa: F401, PLC0415 - déclenche le chargement des deux groupes

    return [info for group in GROUPS for info in load_plugins(group)]


__all__ = [
    "AGENT_GROUP",
    "DISABLE_ENV",
    "GROUPS",
    "HEURISTIC_GROUP",
    "STATUSES",
    "PluginInfo",
    "PluginWarning",
    "installed_entry_points",
    "load_plugins",
    "plugin_report",
]
