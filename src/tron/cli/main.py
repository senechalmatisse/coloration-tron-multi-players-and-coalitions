"""Commande ``tron`` (F7.4) : play, replay, compare, bench, run, analyze, train, config…"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
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
    if args.checkpoint is not None:
        return _play_to_checkpoint(config, args)
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


def _print_result(r: object) -> None:
    from tron.sim.result import MatchResult  # noqa: PLC0415

    assert isinstance(r, MatchResult)
    print(
        f"victoire de {r.winning_team} en {r.n_moves} coups ({r.n_plies} demi-coups) · "
        f"rangs {list(r.ranks)}"
    )


def _play_until(match: object, stop_after: int | None) -> None:
    from tron.sim.match import Match  # noqa: PLC0415

    assert isinstance(match, Match)
    while not match.finished and (stop_after is None or match.state.ply < stop_after):
        if match.step() is None:
            break


def _play_to_checkpoint(config: GameConfig, args: argparse.Namespace) -> int:
    """Joue une partie jusqu'au demi-coup ``--stop-after`` puis la sauvegarde (P08)."""
    from tron.sim.match import Match  # noqa: PLC0415

    if not args.no_gui or args.games != 1:
        raise ConfigError("--checkpoint exige --no-gui et une seule partie")
    match = Match(config)
    _play_until(match, args.stop_after)
    path = match.save_checkpoint(args.checkpoint)
    state = "terminée" if match.finished else f"arrêtée au demi-coup {match.state.ply}"
    print(f"partie {state} · sauvegarde : {path}")
    if match.result is not None:
        _print_result(match.result)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Reprend une partie sauvegardée, jusqu'au bout ou jusqu'à ``--stop-after``."""
    from tron.sim.match import Match, MatchObserver  # noqa: PLC0415
    from tron.sim.recorder import Recorder  # noqa: PLC0415

    observers: list[MatchObserver] = []
    if args.log is not None:
        observers.append(Recorder(args.log))
    match = Match.from_checkpoint(args.path, observers=observers)
    start = match.state.ply
    _play_until(match, args.stop_after)
    print(f"reprise au demi-coup {start} · {match.state.ply - start} demi-coups joués")
    if args.stop_after is not None and not match.finished:
        out = match.save_checkpoint(args.checkpoint or args.path)
        print(f"sauvegarde : {out}")
        return 0
    if match.result is not None:
        _print_result(match.result)
    if args.log is not None:
        print(f"journal complet : {args.log}")
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
    """Mesures historiques, suite reproductible (``--suite``), profil, campagnes (P13)."""
    import json  # noqa: PLC0415

    from tron.benchmarks import runner  # noqa: PLC0415

    if args.campaign is not None:
        rows = runner.bench_campaign(args.campaign, args.workers, args.limit)
        if args.out is not None:
            runner.write_json({"format": "tron-bench-campaign/1", "rows": rows}, args.out)
        if args.format == "json":
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            print(
                f"{'processus':>9}{'parties':>9}{'durée s':>10}{'CPU s':>9}{'parties/s':>11}"
                f"{'RSS max Mo':>12}"
            )
            for r in rows:
                print(
                    f"{r['workers']:>9}{r['games']:>9}{r['wall_s']:>10.2f}{r['cpu_s']:>9.2f}"
                    f"{r['games_per_s']:>11.2f}{r['max_rss_mb']:>12.1f}"
                )
        return 0
    if args.variants:
        from tron.benchmarks.positions import position_set  # noqa: PLC0415
        from tron.benchmarks.variants import (  # noqa: PLC0415
            VARIANTS,
            compare_variants,
            variants_text,
        )

        positions = position_set(
            args.grid or 20, args.players or 4, args.per_phase or 2, args.seed or 0
        )
        names = args.variant or list(VARIANTS)
        rows = compare_variants(
            positions, agent=args.agent or "paranoid", depth=args.depth or 6, variants=names
        )
        if args.out is not None:
            runner.write_json({"format": "tron-bench-variants/1", "rows": rows}, args.out)
        if args.format == "json":
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            print(variants_text(rows))
        exact = [r for r in rows if r["exact"]]
        n = len(positions)
        return 0 if all(r["same_ties"] == n and r["same_exact_values"] == n for r in exact) else 1
    if args.suite:
        params = runner.SuiteParams.quick_params() if args.quick else runner.SuiteParams()
        changes = {
            "grid": args.grid,
            "players": args.players,
            "per_phase": args.per_phase,
            "seed": args.seed,
            "agent": args.agent,
            "depth": args.depth,
            "simulations": args.simulations,
            "repeats": args.repeats,
        }
        params = runner.SuiteParams(
            **{**params.__dict__, **{k: v for k, v in changes.items() if v is not None}}
        )
        holder: dict[str, object] = {}

        def work() -> None:
            holder["result"] = runner.run_suite(params)

        report = None
        if args.profile:
            report = _profiled(work, args)
        else:
            work()
        result = holder["result"]
        assert isinstance(result, dict)
        if args.out is not None:
            runner.write_json(result, args.out)
        comparison = None
        if args.reference is not None:
            reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
            comparison = runner.compare(result, reference)
        if args.format == "json":
            payload: dict[str, object] = {"suite": result}
            if comparison is not None:
                payload["comparison"] = comparison
            if report is not None:
                payload["profile"] = report
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(runner.suite_text(result))
            if comparison is not None:
                print(runner.comparison_text(comparison))
        return 0
    if args.profile:
        report = _profiled(lambda: _bench_classic(args), args)
        if args.format == "json":
            print(json.dumps({"profile": report}, indent=2, ensure_ascii=False))
        return 0
    return _bench_classic(args)


def _profiled(work: Callable[[], object], args: argparse.Namespace) -> list[dict[str, object]]:
    """Exécute ``work`` sous ``cProfile`` ; texte affiché (format text) et lignes renvoyées."""
    from tron.benchmarks.runner import profile  # noqa: PLC0415

    text, rows, stats = profile(work, limit=args.top)
    if args.profile_out is not None:
        stats.dump_stats(str(args.profile_out))
    if args.format == "text":
        print(text)
    return rows


def _bench_classic(args: argparse.Namespace) -> int:
    """Mesures historiques : évaluations d'heuristique et décisions sur des positions aléatoires."""
    import random  # noqa: PLC0415

    from tron.agents.base import make_agent  # noqa: PLC0415
    from tron.core.state import GameState  # noqa: PLC0415
    from tron.eval.base import make_heuristic  # noqa: PLC0415
    from tron.eval.voronoi import VoronoiHeuristic  # noqa: PLC0415
    from tron.sim.match import ensure_warmup  # noqa: PLC0415

    ensure_warmup()
    grid = args.grid or 20
    players = args.players or 4
    agent_name = args.agent or "paranoid"
    depth = args.depth or 10
    seed = args.seed or 0
    rng = random.Random(seed)
    config = GameConfig.from_dict(
        {
            "grid": grid,
            "seed": seed,
            "placement": {"mode": "regular", "rotate": True},
            "heuristic": {"name": args.heuristic},
            "players": [
                {"name": f"J{i}", "agent": agent_name, "depth": depth} for i in range(players)
            ],
        }
    )
    print(
        f"tron {__version__} · noyaux : {backend_name()} · grille {grid}x{grid} · "
        f"{players} joueurs · heuristique {args.heuristic}"
    )

    def midgame(fill: float) -> GameState:
        st = GameState(config.with_seed(rng.randrange(1 << 30)))
        target = int(fill * grid * grid)
        while st.n_moves < target and not st.is_terminal():
            moves = st.legal_moves()
            if not moves:
                break
            st.apply(rng.choice(moves))
        return st

    st = midgame(args.fill)
    heur = make_heuristic(config.heuristic, grid, grid, players)
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
        agent = make_agent(agent_name, st.current, config, depth=depth)
        dec = agent.choose_move(st)
        times.append(dec.stats.time)
        nodes.append(dec.stats.nodes)
        done += 1
    med = statistics.median(times)
    total_nodes = sum(nodes)
    print(
        f"{agent_name} profondeur {depth} : décision médiane {1000 * med:.1f} ms, "
        f"moyenne {1000 * statistics.fmean(times):.1f} ms, max {1000 * max(times):.1f} ms, "
        f"{statistics.fmean(nodes):,.0f} nœuds/décision, {total_nodes / sum(times):,.0f} nœuds/s"
    )
    return 0


# -------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    from tron.experiments.plan import Plan  # noqa: PLC0415
    from tron.experiments.runner import estimate, run_plan  # noqa: PLC0415

    plan = Plan.from_yaml(args.plan)
    if args.pilot:
        plan = plan.pilot_plan(args.games)
    elif args.games is not None:
        plan = plan.with_games(args.games)
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


# ------------------------------------------------------------ apprentissage
def cmd_train(args: argparse.Namespace) -> int:
    from tron.learning.train import TrainingSpec, evaluation_markdown, run_training  # noqa: PLC0415

    spec = TrainingSpec.from_yaml(args.config).with_overrides(
        episodes=args.episodes,
        generations=args.generations,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    out = args.out or Path("models") / spec.name
    print(
        f"entraînement {spec.name} : {spec.total_episodes} épisodes "
        f"({spec.generations} génération(s)), apprenant {spec.player}, partie {spec.game.name}"
    )
    every = max(1, spec.total_episodes // 10)

    def progress(row: dict[str, object]) -> None:
        k = int(str(row["episode"]))
        if not args.quiet and (k + 1) % every == 0:
            print(f"  épisode {k + 1}/{spec.total_episodes} : {row['outcome']}")

    start = time.perf_counter()
    result = run_training(spec, out, on_episode=progress)
    print(
        f"modèle : {out / 'model.json'} ({time.perf_counter() - start:.1f} s, "
        f"victoires sur les derniers épisodes : {result.model.metadata['recent_win_rate']:.0%})"
    )
    if result.evaluation:
        print(
            evaluation_markdown(
                result.evaluation, len(result.train_seeds), len(result.eval_seeds)
            ).rstrip()
        )
        print(f"rapport : {out / 'evaluation.md'}")
    return 0


def cmd_evaluate_model(args: argparse.Namespace) -> int:
    from tron.learning.model import load_model  # noqa: PLC0415
    from tron.learning.train import (  # noqa: PLC0415
        REFERENCES,
        evaluate_model,
        evaluation_markdown,
        write_csv,
    )

    model = load_model(args.model)
    config = GameConfig.from_yaml(args.config)
    refs = tuple(r for r in (args.references or ",".join(REFERENCES)).split(",") if r)
    rows = evaluate_model(
        model,
        config,
        player=args.player,
        references=refs,
        games=args.games,
        depth=args.depth,
        seed=args.seed,
    )
    print(evaluation_markdown(rows).rstrip())
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(args.out / "evaluation.csv", rows)
        print(f"tableau : {args.out / 'evaluation.csv'}")
    return 0


# ---------------------------------------------------------------- compare
def cmd_compare(args: argparse.Namespace) -> int:
    import json  # noqa: PLC0415

    from tron.sim.recorder import GameLog  # noqa: PLC0415
    from tron.viz.comparison import ComparisonSession, describe  # noqa: PLC0415

    session = ComparisonSession(GameLog.load(args.log_a), GameLog.load(args.log_b))
    session.seek(session.max_ply if args.ply is None else args.ply)
    if args.png is not None or args.gif is not None:
        from tron.viz.export import export_animation, save_png  # noqa: PLC0415

        cell = args.cell or 10
        if args.png is not None:
            print(f"image : {save_png(session.frame(cell), args.png)}")
        if args.gif is not None:
            frames = session.frames(cell, every=args.every)
            print(f"animation : {export_animation(frames, args.gif, fps=args.fps)}")
    if not session.pairing.paired:
        print(f"attention : {session.pairing.warning}", file=sys.stderr)
    if args.no_gui:
        if args.format == "json":
            print(json.dumps(describe(session), indent=2, ensure_ascii=False))
        else:
            print(session.text())
        return 0
    from tron.viz.comparison_app import run_compare  # noqa: PLC0415 - fenêtre à la demande

    session.seek(0 if args.ply is None else args.ply)
    run_compare(session, cell=args.cell, speed=args.speed)
    return 0


# ---------------------------------------------------------- configuration
def cmd_config_validate(args: argparse.Namespace) -> int:
    from tron.cli.config_tools import check  # noqa: PLC0415

    failures = 0
    for path in args.files:
        try:
            checked = check(path, args.kind)
        except ConfigError as exc:
            failures += 1
            print(f"invalide : {path} : {exc}")
            continue
        print(f"valide : {path} : {checked.summary}")
    return 1 if failures else 0


def cmd_config_show(args: argparse.Namespace) -> int:
    from tron.cli.config_tools import check, dump, normalized  # noqa: PLC0415

    print(dump(normalized(check(args.file, args.kind)), args.format), end="")
    return 0


def cmd_config_template(args: argparse.Namespace) -> int:
    from tron.cli.config_tools import write_template  # noqa: PLC0415

    text = write_template(args.kind, args.out, force=args.force)
    if args.out is None or str(args.out) == "-":
        print(text, end="")
    else:
        print(f"modèle {args.kind} écrit dans {args.out}")
    return 0


def cmd_agents_list(args: argparse.Namespace) -> int:
    import json  # noqa: PLC0415

    from tron.cli.config_tools import agent_catalog, catalog_text  # noqa: PLC0415

    rows = agent_catalog()
    if args.format == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(catalog_text(rows, f"{len(rows)} agents :"))
    return 0


def cmd_heuristics_list(args: argparse.Namespace) -> int:
    import json  # noqa: PLC0415

    from tron.cli.config_tools import catalog_text, heuristic_catalog  # noqa: PLC0415

    rows = heuristic_catalog()
    if args.format == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(catalog_text(rows, f"{len(rows)} heuristiques :"))
    return 0


# ---------------------------------------------------------------- plugins
def cmd_plugins(args: argparse.Namespace) -> int:
    from tron.plugins import GROUPS, plugin_report  # noqa: PLC0415

    infos = plugin_report()
    rows = [(i.group, i.name, i.distribution, i.target, i.status, i.detail) for i in infos]
    if args.all:
        from tron.agents.base import AGENTS  # noqa: PLC0415
        from tron.eval.base import HEURISTICS  # noqa: PLC0415

        owned = {(i.group, i.name) for i in infos if i.loaded}
        registries: tuple[dict[str, type], ...] = (dict(AGENTS), dict(HEURISTICS))
        for group, registry in zip(GROUPS, registries, strict=True):
            for name, cls in sorted(registry.items()):
                if (group, name) not in owned:
                    target = f"{cls.__module__}:{cls.__qualname__}"
                    rows.append((group, name, "tron (intégré)", target, "builtin", ""))
    if not rows:
        print(f"aucune extension installée (groupes {', '.join(GROUPS)})")
        return 0
    header = ("groupe", "nom", "paquet", "classe", "état", "détail")
    widths = [max(len(str(r[k])) for r in [header, *rows]) for k in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())
    return 0 if all(i.loaded for i in infos) else 1


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
    p.add_argument("--checkpoint", type=Path, help="sauvegarde reprenable (avec --no-gui)")
    p.add_argument("--stop-after", type=int, help="demi-coup où s'arrêter pour sauvegarder")
    viz_options(p)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("resume-checkpoint", help="reprend une partie sauvegardée")
    p.add_argument("path", type=Path, help="fichier de sauvegarde (JSON)")
    p.add_argument("--log", type=Path, help="journal JSONL complet de la partie reprise")
    p.add_argument("--stop-after", type=int, help="s'arrête à ce demi-coup et sauvegarde")
    p.add_argument("--checkpoint", type=Path, help="fichier de la nouvelle sauvegarde")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("replay", help="rejoue un journal JSONL")
    p.add_argument("log", type=Path)
    viz_options(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("bench", help="mesure les performances (Voronoi, décisions, suite)")
    p.add_argument("--grid", type=int, help="taille de grille (défaut : 20)")
    p.add_argument("--players", type=int, help="nombre de joueurs (défaut : 4)")
    p.add_argument("--agent", help="agent de recherche mesuré (défaut : paranoid)")
    p.add_argument("--depth", type=int, help="profondeur (défaut : 10, suite : 3)")
    p.add_argument("--evals", type=int, default=20_000)
    p.add_argument("--decisions", type=int, default=20)
    p.add_argument("--fill", type=float, default=0.2, help="taux de remplissage pour Voronoi")
    p.add_argument("--seed", type=int, help="graine (défaut : 0)")
    p.add_argument(
        "--heuristic", default="voronoi", help="heuristique mesurée (voronoi, composite…)"
    )
    p.add_argument("--suite", action="store_true", help="suite reproductible par composant")
    p.add_argument("--quick", action="store_true", help="suite réduite (essais rapides)")
    p.add_argument("--per-phase", type=int, help="positions par phase (suite, défaut : 5)")
    p.add_argument("--simulations", type=int, help="simulations MCTS (suite, défaut : 200)")
    p.add_argument("--repeats", type=int, help="répétitions par position (suite, défaut : 200)")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--out", type=Path, help="écrit le résultat JSON dans ce fichier")
    p.add_argument("--reference", type=Path, help="compare à un résultat JSON de référence")
    p.add_argument("--profile", action="store_true", help="profil cProfile de la mesure")
    p.add_argument("--top", type=int, default=25, help="fonctions affichées par le profil")
    p.add_argument("--profile-out", type=Path, help="fichier pstats (snakeviz, pstats)")
    p.add_argument("--campaign", type=Path, help="plan : temps et mémoire selon les processus")
    p.add_argument("--variants", action="store_true", help="options de recherche comparées (P16)")
    p.add_argument(
        "--variant", action="append", help="variante à comparer (répétable, défaut : toutes)"
    )
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2], help="processus mesurés")
    p.add_argument("--limit", type=int, default=8, help="parties jouées par campagne mesurée")
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
    p.add_argument(
        "--pilot", action="store_true", help="variante pilote du plan (bloc pilot, sinon 4 parties)"
    )
    p.add_argument("--games", type=int, help="parties par condition (remplace celles du plan)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="tableaux, tests, figures et régression")
    p.add_argument("results", type=Path, help="fichier results.csv")
    p.add_argument("--out", type=Path, help="dossier de sortie (défaut : <résultats>/analysis)")
    p.add_argument("--formula", help="formule de la régression logistique")
    p.add_argument("--annotate", action="store_true", help="valeurs dans les cartes de chaleur")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("train", help="entraîne et évalue un modèle linéaire (P10)")
    p.add_argument("config", type=Path, help="configuration d'entraînement YAML")
    p.add_argument("--out", type=Path, help="dossier du modèle (défaut : models/<nom>)")
    p.add_argument("--episodes", type=int, help="épisodes par génération")
    p.add_argument("--generations", type=int, help="générations d'auto-jeu")
    p.add_argument("--eval-games", type=int, help="parties d'évaluation par référence")
    p.add_argument("--seed", type=int, help="graine de l'entraînement")
    p.add_argument("--quiet", action="store_true", help="sans suivi des épisodes")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate-model", help="compare un modèle appris aux agents de référence")
    p.add_argument("model", type=Path, help="fichier model.json")
    p.add_argument("config", type=Path, help="configuration de partie YAML")
    p.add_argument("--player", type=int, default=0, help="place jouée par le modèle")
    p.add_argument("--games", type=int, default=20, help="parties par référence")
    p.add_argument("--references", help="agents séparés par des virgules")
    p.add_argument("--depth", type=int, default=2, help="profondeur des agents de recherche")
    p.add_argument("--seed", type=int, default=0, help="graine des parties d'évaluation")
    p.add_argument("--out", type=Path, help="dossier du tableau evaluation.csv")
    p.set_defaults(func=cmd_evaluate_model)

    p = sub.add_parser("compare", help="compare deux rejeux côte à côte (P17)")
    p.add_argument("log_a", type=Path, help="premier journal JSONL")
    p.add_argument("log_b", type=Path, help="second journal JSONL")
    p.add_argument("--ply", type=int, help="demi-coup affiché (défaut : début, fin sans fenêtre)")
    p.add_argument("--cell", type=int, help="taille d'une case en pixels")
    p.add_argument("--speed", type=float, default=10, help="demi-coups par seconde")
    p.add_argument("--no-gui", action="store_true", help="panneau de comparaison en texte")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--png", type=Path, help="image côte à côte au demi-coup affiché")
    p.add_argument("--gif", type=Path, help="animation côte à côte de toute la comparaison")
    p.add_argument("--fps", type=float, default=10, help="images par seconde de l'animation")
    p.add_argument("--every", type=int, default=1, help="une image tous les N demi-coups")
    p.set_defaults(func=cmd_compare)

    from tron.cli.config_tools import KINDS, TEMPLATES  # noqa: PLC0415

    p = sub.add_parser("config", help="valide, normalise ou crée des configurations YAML")
    cfg_sub = p.add_subparsers(dest="config_command", required=True)
    q = cfg_sub.add_parser("validate", help="valide des fichiers (partie, plan, entraînement)")
    q.add_argument("files", type=Path, nargs="+")
    q.add_argument("--kind", choices=KINDS, default="auto", help="type (défaut : reconnu)")
    q.set_defaults(func=cmd_config_validate)
    q = cfg_sub.add_parser("show", help="affiche la configuration normalisée")
    q.add_argument("file", type=Path)
    q.add_argument("--kind", choices=KINDS, default="auto")
    q.add_argument("--format", choices=("yaml", "json"), default="yaml")
    q.set_defaults(func=cmd_config_show)
    q = cfg_sub.add_parser("template", help="crée un modèle de configuration commenté")
    q.add_argument("kind", choices=sorted(TEMPLATES), help="type de modèle")
    q.add_argument("--out", type=Path, help="fichier à écrire (défaut : sortie standard)")
    q.add_argument("--force", action="store_true", help="écrase un fichier existant")
    q.set_defaults(func=cmd_config_template)

    p = sub.add_parser("agents", help="agents disponibles")
    q = p.add_subparsers(dest="agents_command", required=True).add_parser(
        "list", help="agents et paramètres"
    )
    q.add_argument("--format", choices=("text", "json"), default="text")
    q.set_defaults(func=cmd_agents_list)

    p = sub.add_parser("heuristics", help="heuristiques disponibles")
    q = p.add_subparsers(dest="heuristics_command", required=True).add_parser(
        "list", help="heuristiques et paramètres"
    )
    q.add_argument("--format", choices=("text", "json"), default="text")
    q.set_defaults(func=cmd_heuristics_list)

    p = sub.add_parser("plugins", help="extensions installées (agents, heuristiques)")
    plugins = p.add_subparsers(dest="plugins_command", required=True)
    q = plugins.add_parser("list", help="nom, paquet, classe et état des extensions")
    q.add_argument("--all", action="store_true", help="inclut les éléments intégrés")
    q.set_defaults(func=cmd_plugins)
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
