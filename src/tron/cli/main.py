"""Commande ``tron`` (F7.4) : play, replay, bench, run, analyze."""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from tron import __version__
from tron._jit import backend_name
from tron.core.config import ConfigError, GameConfig


# ------------------------------------------------------------------- play
def cmd_play(args: argparse.Namespace) -> int:
    config = GameConfig.from_yaml(args.config)
    if args.seed is not None or args.first_player is not None:
        config = config.with_seed(
            config.seed if args.seed is None else args.seed,
            args.first_player,
        )
    if not args.no_gui:
        from tron.viz.arcade_app import run_live  # noqa: PLC0415 - fenêtre à la demande

        run_live(config, cell=args.cell, speed=args.speed, voronoi=args.voronoi, paused=args.paused)
        return 0
    from tron.sim.arena import run_arena  # noqa: PLC0415

    out = args.out
    tmp = None
    if out is None and (args.gif or args.mp4):
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name)

    def report(g: int, r: object) -> None:
        from tron.sim.result import MatchResult  # noqa: PLC0415

        assert isinstance(r, MatchResult)
        think = " · ".join(
            f"{name}:{t.think_time:.2f}s" for name, t in zip(r.players, r.totals, strict=True)
        )
        print(
            f"partie {g} (graine {r.seed}, 1er joueur {r.first_player}) : victoire de "
            f"{r.winning_team} en {r.n_moves} coups · rangs {list(r.ranks)} · {think}"
        )

    try:
        results = run_arena(
            config, games=args.games, out_dir=out, logs=out is not None, on_result=report
        )
        if (args.gif or args.mp4) and out is not None:
            from tron.sim.recorder import GameLog  # noqa: PLC0415
            from tron.viz.export import export_animation, log_frames  # noqa: PLC0415

            log = GameLog.load(Path(out) / "game_0000.jsonl")
            for target in (args.gif, args.mp4):
                if target:
                    frames = log_frames(
                        log, cell=args.cell or 12, every=args.every, voronoi=args.voronoi
                    )
                    print(f"export : {export_animation(frames, target, fps=args.fps)}")
    finally:
        if tmp is not None:
            tmp.cleanup()
    if args.games > 1:
        wins: dict[str, int] = {}
        for r in results:
            wins[r.winning_team] = wins.get(r.winning_team, 0) + 1
        print("victoires par équipe :", dict(sorted(wins.items())))
    if args.out:
        print(f"résultats : {Path(args.out) / 'results.csv'}")
    return 0


# ----------------------------------------------------------------- replay
def cmd_replay(args: argparse.Namespace) -> int:
    from tron.sim.recorder import GameLog  # noqa: PLC0415

    log = GameLog.load(args.log)
    exported = False
    for target in (args.gif, args.mp4):
        if target:
            from tron.viz.export import export_animation, log_frames  # noqa: PLC0415

            frames = log_frames(log, cell=args.cell or 12, every=args.every, voronoi=args.voronoi)
            print(f"export : {export_animation(frames, target, fps=args.fps)}")
            exported = True
    if args.no_gui or exported:
        res = log.result or {}
        print(
            f"{log.config.name} : {len(log.moves)} demi-coups, "
            f"victoire de {res.get('winning_team', '?')}"
        )
        return 0
    from tron.viz.arcade_app import run_replay  # noqa: PLC0415

    run_replay(log, cell=args.cell, speed=args.speed, voronoi=args.voronoi)
    return 0


# ------------------------------------------------------------------ bench
def cmd_bench(args: argparse.Namespace) -> int:
    import random  # noqa: PLC0415

    from tron.agents.base import make_agent  # noqa: PLC0415
    from tron.core.state import GameState  # noqa: PLC0415
    from tron.eval.base import make_heuristic  # noqa: PLC0415
    from tron.eval.voronoi import VoronoiHeuristic  # noqa: PLC0415
    from tron.sim.match import ensure_warmup  # noqa: PLC0415

    ensure_warmup()
    rng = random.Random(args.seed)
    config = GameConfig.from_dict(
        {
            "grid": args.grid,
            "seed": args.seed,
            "placement": {"mode": "regular", "rotate": True},
            "heuristic": {"name": args.heuristic},
            "players": [
                {"name": f"J{i}", "agent": args.agent, "depth": args.depth}
                for i in range(args.players)
            ],
        }
    )
    print(
        f"tron {__version__} · noyaux : {backend_name()} · grille {args.grid}x{args.grid} · "
        f"{args.players} joueurs · heuristique {args.heuristic}"
    )

    def midgame(fill: float) -> GameState:
        st = GameState(config.with_seed(rng.randrange(1 << 30)))
        target = int(fill * args.grid * args.grid)
        while st.n_moves < target and not st.is_terminal():
            moves = st.legal_moves()
            if not moves:
                break
            st.apply(rng.choice(moves))
        return st

    st = midgame(args.fill)
    heur = make_heuristic(config.heuristic, args.grid, args.grid, args.players)
    # Voronoi : mesure historique sur ``counts`` (comparabilité avec les mesures de référence).
    evaluate = heur.counts if isinstance(heur, VoronoiHeuristic) else heur.shares
    evaluate(st)
    t0 = time.perf_counter()
    for _ in range(args.evals):
        evaluate(st)
    dt = time.perf_counter() - t0
    label = "Voronoi" if args.heuristic == "voronoi" else args.heuristic
    print(f"{label} : {args.evals / dt:,.0f} évaluations/s ({1e6 * dt / args.evals:.1f} µs)")

    times, nodes = [], []
    done = 0
    while done < args.decisions:
        st = midgame(rng.uniform(0.05, 0.35))
        if st.is_terminal() or len(st.legal_moves()) < 2:
            continue
        agent = make_agent(args.agent, st.current, config, depth=args.depth)
        dec = agent.choose_move(st)
        times.append(dec.stats.time)
        nodes.append(dec.stats.nodes)
        done += 1
    med = statistics.median(times)
    total_nodes = sum(nodes)
    print(
        f"{args.agent} profondeur {args.depth} : décision médiane {1000 * med:.1f} ms, "
        f"moyenne {1000 * statistics.fmean(times):.1f} ms, max {1000 * max(times):.1f} ms, "
        f"{statistics.fmean(nodes):,.0f} nœuds/décision, {total_nodes / sum(times):,.0f} nœuds/s"
    )
    return 0


# -------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    from tron.experiments.plan import Plan  # noqa: PLC0415
    from tron.experiments.runner import estimate, run_plan  # noqa: PLC0415

    plan = Plan.from_yaml(args.plan)
    conditions = plan.conditions()
    n_tasks = sum(c.games for c in conditions)
    print(f"plan {plan.name} : {len(conditions)} conditions, {n_tasks} parties")
    if args.dry_run:
        return 0
    if args.estimate:
        est = estimate(plan, samples=args.estimate, workers=args.workers)
        print(
            f"durée moyenne d'une partie : {est['mean_game_s']:.2f} s "
            f"(max {est['max_game_s']:.2f} s) · budget : {est['cpu_hours']:.1f} h CPU, "
            f"{est['wall_hours']:.1f} h avec {args.workers or 'tous les'} cœurs"
        )
        return 0
    out = args.out or Path("results") / plan.name
    summary = run_plan(
        plan,
        out,
        workers=args.workers,
        resume=not args.no_resume,
        limit=args.limit,
        logs=args.logs or None,
    )
    print(
        f"{summary.played} parties jouées ({summary.skipped} déjà présentes) en "
        f"{summary.elapsed:.1f} s → {summary.results}"
    )
    return 0


# ---------------------------------------------------------------- analyze
def cmd_analyze(args: argparse.Namespace) -> int:
    from tron.analysis.report import analyze  # noqa: PLC0415

    out = args.out or Path(args.results).parent / "analysis"
    res = analyze(args.results, out, formula=args.formula, annotate=args.annotate)
    print(f"tableau : {res.summary}")
    if res.paired:
        print(f"comparaisons appariées : {res.paired}")
    if res.regression:
        print(f"régression : {res.regression}")
    if res.cooperation:
        print(f"coopération : {res.cooperation}")
    if res.cooperation_conditions:
        print(f"coopération par condition : {res.cooperation_conditions}")
    if res.diplomacy:
        print(f"diplomatie : {res.diplomacy}")
    for path in res.figures:
        print(f"figure : {path}")
    for msg in res.messages:
        print(msg)
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tron", description="Tron multi-joueur et coalitions")
    parser.add_argument("--version", action="version", version=f"tron {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def viz_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cell", type=int, help="taille d'une case en pixels")
        p.add_argument("--speed", type=float, default=10, help="demi-coups par seconde")
        p.add_argument("--voronoi", action="store_true", help="surcouche des territoires")
        p.add_argument("--gif", type=Path, help="exporte la partie en GIF")
        p.add_argument("--mp4", type=Path, help="exporte la partie en MP4")
        p.add_argument("--fps", type=float, default=20, help="images par seconde de l'export")
        p.add_argument("--every", type=int, default=1, help="une image tous les N demi-coups")
        p.add_argument("--no-gui", action="store_true", help="sans fenêtre")

    p = sub.add_parser("play", help="joue une partie (fenêtre arcade ou sans affichage)")
    p.add_argument("config", type=Path, help="configuration YAML de la partie")
    p.add_argument("--seed", type=int)
    p.add_argument("--first-player", type=int)
    p.add_argument("--games", type=int, default=1, help="nombre de parties (sans affichage)")
    p.add_argument("--out", type=Path, help="dossier des résultats et journaux")
    p.add_argument("--paused", action="store_true", help="démarre en pause")
    viz_options(p)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("replay", help="rejoue un journal JSONL")
    p.add_argument("log", type=Path)
    viz_options(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("bench", help="mesure les performances (Voronoi, décisions)")
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--agent", default="paranoid")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--evals", type=int, default=20_000)
    p.add_argument("--decisions", type=int, default=20)
    p.add_argument("--fill", type=float, default=0.2, help="taux de remplissage pour Voronoi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--heuristic", default="voronoi", help="heuristique mesurée (voronoi, composite…)"
    )
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("run", help="exécute un plan d'expériences")
    p.add_argument("plan", type=Path)
    p.add_argument("--out", type=Path, help="dossier de sortie (défaut : results/<plan>)")
    p.add_argument("--workers", type=int, help="processus (défaut : tous les cœurs)")
    p.add_argument("--no-resume", action="store_true", help="repart de zéro")
    p.add_argument("--limit", type=int, help="nombre maximal de parties à jouer")
    p.add_argument("--logs", action="store_true", help="journaux JSONL de chaque partie")
    p.add_argument("--estimate", type=int, metavar="N", help="estime le budget sur N parties")
    p.add_argument("--dry-run", action="store_true", help="affiche la taille du plan")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="tableaux, tests, figures et régression")
    p.add_argument("results", type=Path, help="fichier results.csv")
    p.add_argument("--out", type=Path, help="dossier de sortie (défaut : <résultats>/analysis)")
    p.add_argument("--formula", help="formule de la régression logistique")
    p.add_argument("--annotate", action="store_true", help="valeurs dans les cartes de chaleur")
    p.set_defaults(func=cmd_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
