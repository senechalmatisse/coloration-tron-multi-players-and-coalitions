"""Export d'une partie en GIF ou MP4 et captures PNG (F6.2, F6.4).

``theme="neon"`` produit les images du thème néon (:mod:`tron.viz.neon`),
effets animés compris ; ``classic`` (valeur par défaut de la bibliothèque) garde
le rendu d'origine.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from tron.eval.voronoi import VoronoiHeuristic
from tron.sim.recorder import GameLog
from tron.viz.palette import Palette, make_palette
from tron.viz.render import render_frame
from tron.viz.theme import DEFAULT_QUALITY, Quality, resolve_theme


def log_frames(
    log: GameLog,
    cell: int = 12,
    every: int = 1,
    voronoi: bool = False,
    hold_end: int = 15,
    palette: Palette | None = None,
    theme: str = "classic",
    quality: str | Quality = DEFAULT_QUALITY,
    fps: float = 20.0,
) -> Iterator[np.ndarray]:
    """Images successives d'une partie journalisée (une image tous les ``every`` demi-coups).

    ``theme="neon"`` : rendu néon, effets animés au rythme de ``fps`` images/s.
    """
    if resolve_theme(theme) == "neon":
        from tron.viz.neon import neon_frames  # noqa: PLC0415 - rendu néon à la demande

        yield from neon_frames(log, cell, every, voronoi, hold_end, quality, fps)
        return
    pal = palette or make_palette(log.config.team_index)
    heur = VoronoiHeuristic(log.config.width, log.config.height, log.config.n_players)
    frame = None
    total = len(log.moves)
    for i, st in enumerate(log.states()):
        if i % every and i != total:
            continue
        owners = heur.owner_map(st) if voronoi and not st.is_terminal() else None
        frame = render_frame(st, pal, cell=cell, owners=owners)
        yield frame
    for _ in range(hold_end if frame is not None else 0):
        yield frame  # type: ignore[misc]


def save_png(frame: np.ndarray, path: str | Path) -> Path:
    import imageio.v3 as iio  # noqa: PLC0415 - dépendance chargée à la demande

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frame)
    return out


def _pad_even(frame: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = frame.shape[:2]
    ph, pw = (-h) % block, (-w) % block
    if ph or pw:
        frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return frame


def export_animation(frames: Iterable[np.ndarray], path: str | Path, fps: float = 20) -> Path:
    """Écrit les images en ``.gif`` ou ``.mp4`` selon l'extension de ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".gif":
        import imageio.v2 as iio2  # noqa: PLC0415

        with iio2.get_writer(out, mode="I", duration=1000.0 / fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]
        return out
    if suffix == ".mp4":
        return _export_mp4(frames, out, fps)
    raise ValueError(f"format d'export inconnu : {suffix} (.gif ou .mp4)")


def _export_mp4(frames: Iterable[np.ndarray], out: Path, fps: float) -> Path:
    try:
        import imageio.v2 as iio2  # noqa: PLC0415

        with iio2.get_writer(
            out, fps=fps, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for frame in frames:
                writer.append_data(_pad_even(frame))  # type: ignore[attr-defined]
        return out
    except (ImportError, ValueError, RuntimeError):
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("export MP4 impossible : installez imageio-ffmpeg ou ffmpeg")
    proc: subprocess.Popen[bytes] | None = None
    for frame in frames:
        f = _pad_even(frame)
        if proc is None:
            h, w = f.shape[:2]
            proc = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    str(out),
                ],
                stdin=subprocess.PIPE,
            )
        assert proc.stdin is not None
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    if proc is not None:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait()
    return out
