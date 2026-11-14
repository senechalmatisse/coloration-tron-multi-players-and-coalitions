"""Mise en scène des événements visuels : particules, ondes, liens, éclair, tremblement.

:class:`EffectDirector` reçoit les changements constatés par
:class:`~tron.viz.scene.SceneTracker` et en déduit les effets, puis les fait
vivre au rythme du temps écoulé (``delta_time``). Il ne dessine rien : le rendu
NumPy et la fenêtre Arcade lisent son état. Règles de sobriété :

* en avance rapide (plus de ``MAX_EFFECT_PLIES`` demi-coups d'un coup) ou à
  grande vitesse de lecture, pas de particules ni de tremblement ;
* un retour arrière efface les effets en cours (ils appartiennent au futur) ;
* le tremblement est court et léger, et seulement si la qualité le permet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tron.core.directions import DX, DY
from tron.core.state import GameState
from tron.viz.effects.particles import ParticleSystem
from tron.viz.palette import RGB, mix
from tron.viz.scene import SceneTracker, SyncResult, VisualEvent
from tron.viz.theme import (
    DEATH_FADE,
    FLASH_DURATION,
    LINK_GROW,
    MAX_EFFECT_PLIES,
    RING_DURATION,
    SHAKE_AMPLITUDE,
    SHAKE_DURATION,
    SHAKE_MAX_SPEED,
    NeonTheme,
    Quality,
    resolve_quality,
)

_WHITE = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class Ring:
    """Onde de choc circulaire (centre en cases)."""

    x: float
    y: float
    color: RGB
    start: float
    life: float
    r0: float
    r1: float

    def radius(self, now: float) -> float:
        k = min(1.0, max(0.0, (now - self.start) / self.life))
        return self.r0 + (self.r1 - self.r0) * (1.0 - (1.0 - k) ** 2)

    def alpha(self, now: float) -> float:
        k = min(1.0, max(0.0, (now - self.start) / self.life))
        return (1.0 - k) ** 1.5


@dataclass(frozen=True, slots=True)
class LinkFlash:
    """Lien d'alliance qui se tend de ``a`` vers ``b`` (formation) puis s'efface."""

    a: int
    b: int
    color: RGB
    start: float
    life: float

    def progress(self, now: float) -> float:
        return min(1.0, max(0.0, (now - self.start) / LINK_GROW))

    def alpha(self, now: float) -> float:
        k = (now - self.start) / self.life
        return max(0.0, 1.0 - max(0.0, k - 0.5) * 2.0)


class EffectDirector:
    """Effets visuels déclenchés par les événements d'une scène."""

    def __init__(self, theme: NeonTheme, quality: Quality | str, seed: int = 0) -> None:
        self.theme = theme
        self.quality = resolve_quality(quality)
        self.seed = seed
        self.time = 0.0
        self.particles = ParticleSystem(self.quality.max_particles, seed)
        self.rings: list[Ring] = []
        self.links: list[LinkFlash] = []
        self.deaths: dict[int, float] = {}
        """Joueur éliminé → instant de l'élimination (−inf : sans animation)."""
        self._flash_start = -math.inf
        self._shake_start = -math.inf
        self._shake_amp = 0.0
        self._move_acc = 0.0

    # réglages
    def set_quality(self, quality: Quality | str) -> None:
        self.quality = resolve_quality(quality)
        old = self.particles
        self.particles = ParticleSystem(self.quality.max_particles, self.seed)
        self.particles.rng = old.rng

    def clear(self) -> None:
        """Efface les effets en cours (particules, ondes, liens, éclair, tremblement)."""
        self.particles.clear()
        self.rings.clear()
        self.links.clear()
        self._flash_start = -math.inf
        self._shake_start = -math.inf

    # lecture
    def fade(self, player: int) -> float:
        """Avancement (0 → 1) de l'extinction de la trace d'un joueur éliminé."""
        start = self.deaths.get(player)
        if start is None:
            return 0.0
        if start == -math.inf:
            return 1.0
        return min(1.0, max(0.0, (self.time - start) / DEATH_FADE))

    def flash_alpha(self) -> float:
        k = (self.time - self._flash_start) / FLASH_DURATION
        return 0.0 if k >= 1.0 or k < 0.0 else (1.0 - k) ** 2

    def shake_offset(self) -> tuple[float, float]:
        """Décalage de caméra courant, en cases (nul hors tremblement)."""
        k = (self.time - self._shake_start) / SHAKE_DURATION
        if k >= 1.0 or k < 0.0:
            return 0.0, 0.0
        amp = self._shake_amp * (1.0 - k) ** 2
        t = self.time * 60.0
        return amp * math.sin(t * 1.7), amp * math.cos(t * 1.3)

    def active_rings(self) -> list[Ring]:
        return [r for r in self.rings if self.time - r.start < r.life]

    def active_links(self) -> list[LinkFlash]:
        return [lk for lk in self.links if self.time - lk.start < lk.life]

    @property
    def busy(self) -> bool:
        """Vrai tant qu'un effet est en cours (le rendu doit continuer d'évoluer)."""
        return bool(
            len(self.particles)
            or self.active_rings()
            or self.active_links()
            or self.flash_alpha() > 0
            or any(self.time - s < DEATH_FADE for s in self.deaths.values())
        )

    # évolution
    def update(self, dt: float) -> None:
        self.time += max(0.0, dt)
        self.particles.update(dt)
        self.rings = self.active_rings()
        self.links = self.active_links()

    # événements
    def handle(
        self,
        result: SyncResult,
        tracker: SceneTracker,
        state: GameState,
        speed: float = 10.0,
    ) -> None:
        """Déclenche les effets des événements de ``result`` (lecture à ``speed`` demi-coups/s)."""
        if result.reset or result.removed:
            self.clear()
            self.deaths = {
                p: self.deaths.get(p, -math.inf) if not result.reset else -math.inf
                for p in range(state.n)
                if not state.is_alive(p)
            }
        burst = len(result.added) <= MAX_EFFECT_PLIES and not result.reset
        calm = speed <= SHAKE_MAX_SPEED
        for ev in result.events:
            if ev.kind == "elimination":
                self.deaths[ev.player] = self.time if burst else -math.inf
            if not burst:
                continue
            handler = _HANDLERS.get(ev.kind)
            if handler is not None:
                handler(self, ev, tracker, state, calm)

    # effets
    def _scale(self, n: float) -> int:
        return max(0, round(n * self.quality.particle_scale))

    def _move(self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool) -> None:
        self._move_acc += 0.8 * self.quality.particle_scale
        k = int(self._move_acc)
        self._move_acc -= k
        if k <= 0:
            return
        color = self.theme.player[ev.player]
        x, y = ev.point
        back = math.atan2(-DY[ev.heading], -DX[ev.heading]) if ev.heading >= 0 else 0.0
        self.particles.emit(
            k,
            x,
            y,
            [color, mix(color, _WHITE, 0.5)],
            speed=(0.3, 1.1),
            life=(0.25, 0.5),
            size=(0.05, 0.10),
            angle=back,
            spread=1.6,
            drag=3.0,
            jitter=0.15,
        )

    def _turn(self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool) -> None:
        self._move(ev, tracker, state, calm)
        color = self.theme.player[ev.player]
        x, y = ev.point
        self.particles.emit(
            self._scale(7),
            x,
            y,
            [mix(color, _WHITE, 0.55), color],
            speed=(1.5, 4.0),
            life=(0.2, 0.45),
            size=(0.04, 0.09),
            drag=4.0,
            shrink=0.8,
        )

    def _elimination(
        self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool
    ) -> None:
        theme = self.theme
        p = ev.player
        color = theme.player[p]
        colors: list[RGB] = [color, mix(color, _WHITE, 0.6)]
        if ev.other >= 0:  # collision avec la trace d'un autre joueur : couleurs mêlées
            colors += [theme.player[ev.other], mix(color, theme.player[ev.other], 0.5)]
        elif ev.cause in ("bordure", "obstacle"):
            colors.append(theme.warning)
        x, y = ev.point
        self.particles.emit(
            self._scale(40),
            x,
            y,
            colors,
            speed=(1.5, 6.5),
            life=(0.5, 1.2),
            size=(0.08, 0.22),
            drag=2.6,
        )
        self.rings.append(Ring(x, y, color, self.time, RING_DURATION, 0.25, 2.4))
        # désintégration : la fin de la trace se défait en poussière lumineuse
        path = tracker.paths[p] if p < tracker.n else []
        for cx, cy in path[-6:]:
            self.particles.emit(
                self._scale(3),
                cx + 0.5,
                cy + 0.5,
                [color, theme.dead[p]],
                speed=(0.1, 0.6),
                life=(0.6, 1.3),
                size=(0.05, 0.12),
                angle=-math.pi / 2,
                spread=2.4,
                drag=1.0,
                jitter=0.35,
            )
        if calm and self.quality.flash:
            self._flash_start = self.time
        if calm and self.quality.shake:
            self._shake_start = self.time
            self._shake_amp = SHAKE_AMPLITUDE

    def _alliance(
        self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool
    ) -> None:
        if not (0 <= ev.player < state.n and 0 <= ev.other < state.n):
            return
        color = mix(self.theme.player[ev.player], self.theme.player[ev.other], 0.5)
        self.links.append(LinkFlash(ev.player, ev.other, color, self.time, 1.4))
        for q in (ev.player, ev.other):
            x, y = state.head(q)
            self.particles.emit(
                self._scale(10),
                x + 0.5,
                y + 0.5,
                [self.theme.accent, self.theme.player[q]],
                speed=(0.8, 2.4),
                life=(0.4, 0.8),
                size=(0.05, 0.11),
            )

    def _rupture(
        self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool
    ) -> None:
        for q in (ev.player, ev.other):
            if not 0 <= q < state.n:
                continue
            x, y = state.head(q)
            self.particles.emit(
                self._scale(10),
                x + 0.5,
                y + 0.5,
                [self.theme.danger, self.theme.player[q]],
                speed=(1.0, 3.0),
                life=(0.3, 0.7),
                size=(0.05, 0.11),
                drag=3.0,
            )

    def _zone(self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool) -> None:
        zones = state.config.rules.zones
        if not 0 <= ev.index < len(zones):
            return
        z = zones[ev.index]
        color = zone_color(state, self.theme, ev.other)
        cx, cy = z.x + z.width / 2, z.y + z.height / 2
        self.rings.append(
            Ring(cx, cy, color, self.time, RING_DURATION * 1.4, 0.3, 0.6 * max(z.width, z.height))
        )

    def _obstacles(
        self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool
    ) -> None:
        for x, y in ev.cells:
            self.particles.emit(
                self._scale(6),
                x + 0.5,
                y + 0.5,
                [self.theme.obstacle_edge, self.theme.warning],
                speed=(0.6, 2.0),
                life=(0.3, 0.6),
                size=(0.05, 0.1),
            )
            self.rings.append(Ring(x + 0.5, y + 0.5, self.theme.warning, self.time, 0.35, 0.2, 0.9))

    def _victory(
        self, ev: VisualEvent, tracker: SceneTracker, state: GameState, calm: bool
    ) -> None:
        out = state.outcome()
        for p in out.winners if out is not None else ():
            x, y = state.head(p)
            color = self.theme.player[p]
            self.particles.emit(
                self._scale(60),
                x + 0.5,
                y + 0.5,
                [color, mix(color, _WHITE, 0.7), self.theme.accent],
                speed=(1.0, 7.0),
                life=(0.7, 1.6),
                size=(0.08, 0.24),
                drag=1.8,
            )
            self.rings.append(Ring(x + 0.5, y + 0.5, color, self.time, RING_DURATION * 2, 0.3, 3.2))
        if self.quality.flash:
            self._flash_start = self.time


def zone_color(state: GameState, theme: NeonTheme, team: int) -> RGB:
    """Couleur d'une équipe de victoire (zone contrôlée), ou couleur neutre si ``team`` < 0."""
    if team < 0:
        return theme.text_dim
    victory = state.config.victory_team_index
    for p, t in enumerate(victory):
        if t == team:
            if state.config.rules.individual:
                return theme.player[p]
            return theme.team[state.config.team_index[p]]
    return theme.text_dim


def zone_controllers(state: GameState) -> list[int]:
    """Équipe de victoire qui contrôle chaque zone (tête seule dans la zone), −1 sinon."""
    victory = state.config.victory_team_index
    out = []
    for zone in state.config.rules.zones:
        teams = {victory[p] for p in state.alive_players() if zone.contains(*state.head(p))}
        out.append(next(iter(teams)) if len(teams) == 1 else -1)
    return out


_HANDLERS = {
    "move": EffectDirector._move,
    "turn": EffectDirector._turn,
    "elimination": EffectDirector._elimination,
    "alliance": EffectDirector._alliance,
    "rupture": EffectDirector._rupture,
    "zone": EffectDirector._zone,
    "obstacles": EffectDirector._obstacles,
    "victory": EffectDirector._victory,
}


__all__ = ["EffectDirector", "LinkFlash", "Ring", "zone_color", "zone_controllers"]
