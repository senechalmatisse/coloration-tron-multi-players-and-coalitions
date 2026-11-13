"""Optimisations contrôlées des moteurs de recherche (P16).

Chaque option exacte (tris, table de transposition, approfondissement itératif,
fenêtres d'aspiration) doit donner la même valeur racine, les mêmes coups ex
aequo et les mêmes valeurs racine exactes qu'une recherche **sans élagage** ;
la réduction des coups tardifs, inexacte, reste désactivée par défaut.
"""

from __future__ import annotations

import pytest

from tron.agents import make_agent
from tron.agents.search import SearchMemory, SearchOptions, SearchResult, search
from tron.agents.searchers import SearchAgent
from tron.benchmarks.positions import BenchPosition, position_set
from tron.benchmarks.variants import EXACT_VARIANTS, VARIANTS, compare_variants
from tron.cli.main import main
from tron.core import ConfigError, GameState

_POSITIONS: dict[tuple[int, int], list[BenchPosition]] = {}


def _positions(grid: int, players: int) -> list[BenchPosition]:
    key = (grid, players)
    if key not in _POSITIONS:
        _POSITIONS[key] = position_set(grid, players, per_phase=2, seed=4)
    return _POSITIONS[key]


def _run(agent: str, pos: BenchPosition, depth: int, **params: object) -> SearchResult:
    a = make_agent(agent, pos.state().current, pos.config, **params)
    assert isinstance(a, SearchAgent)
    st = pos.state()
    key, snap = st.key, st.to_snapshot()
    result = a.run_search(st, depth)
    assert st.key == key and st.to_snapshot() == snap  # état restauré
    return result


def _assert_same(result: SearchResult, ref: SearchResult) -> None:
    assert result.value == pytest.approx(ref.value, abs=1e-12)
    assert result.ties == ref.ties and result.best == ref.best
    for m in set(result.exact) & set(ref.exact):
        assert result.root_values[m] == pytest.approx(ref.root_values[m], abs=1e-12)


@pytest.mark.parametrize("variant", EXACT_VARIANTS)
@pytest.mark.parametrize(("grid", "players", "depth"), [(7, 2, 6), (8, 3, 5)])
def test_alphabeta_variants_match_full_minimax(
    variant: str, grid: int, players: int, depth: int
) -> None:
    for pos in _positions(grid, players):
        ref = _run("paranoid", pos, depth, **VARIANTS["plain"], pruning=False)
        _assert_same(_run("paranoid", pos, depth, **VARIANTS[variant]), ref)


@pytest.mark.parametrize(
    "variant", ["history", "mobility", "territory", "tt", "iterative+tt", "history+it+tt"]
)
@pytest.mark.parametrize(
    ("agent", "params"),
    [("maxn", {}), ("social", {"profile": "team", "alpha": 0.5})],
)
def test_maxn_variants_match_unpruned_maxn(
    variant: str, agent: str, params: dict[str, object]
) -> None:
    for pos in _positions(8, 3):
        ref = _run(agent, pos, 3, **params, **VARIANTS["plain"], pruning=False)
        _assert_same(_run(agent, pos, 3, **params, **VARIANTS[variant]), ref)


def test_tiny_aspiration_windows_force_researches_and_stay_exact() -> None:
    researches = 0
    for pos in _positions(8, 3):
        ref = _run("paranoid", pos, 5, **VARIANTS["plain"], pruning=False)
        result = _run("paranoid", pos, 5, **{**VARIANTS["aspiration"], "aspiration": 1e-6})
        _assert_same(result, ref)
        researches += result.researches
    assert researches > 0


class _CollidingTT(dict[int, int]):
    """Table dont toutes les clés entrent en collision (un seul emplacement)."""

    def get(self, key: int, default: int | None = None) -> int:  # type: ignore[override]
        return super().get(0, -1 if default is None else default)

    def __setitem__(self, key: int, value: int) -> None:
        super().__setitem__(0, value)


class _FixedTT(dict[int, int]):
    """Table qui propose le même coup (souvent illégal ou mauvais) pour toute clé."""

    def __init__(self, move: int) -> None:
        super().__init__()
        self.move = move

    def get(self, key: int, default: int | None = None) -> int:  # type: ignore[override]
        return self.move


@pytest.mark.parametrize(("engine", "agent"), [("alphabeta", "paranoid"), ("maxn", "maxn")])
def test_transposition_collisions_only_change_ordering(engine: str, agent: str) -> None:
    for pos in _positions(8, 3):
        ref = _run(agent, pos, 4, **VARIANTS["plain"], pruning=False)
        a = make_agent(agent, pos.state().current, pos.config, **VARIANTS["tt"])
        assert isinstance(a, SearchAgent) and a.memory.tt is not None
        for memory in (
            SearchMemory(tt=_CollidingTT(), history=None),
            *(SearchMemory(tt=_FixedTT(m), history=None) for m in range(4)),  # coups faux
        ):
            a.memory = memory
            st = pos.state()
            key = st.key
            result = a.run_search(st, 4)
            assert st.key == key
            _assert_same(result, ref)


def test_reduction_is_experimental_off_by_default_and_inactive_below_depth_three() -> None:
    assert SearchOptions().reduction == 0 and SearchOptions().aspiration is None
    for name in ("paranoid", "maxn", "social", "negamax"):
        pos = _positions(7, 2)[0]
        agent = make_agent(name, pos.state().current, pos.config)
        assert isinstance(agent, SearchAgent)
        assert agent.options.reduction == 0 and agent.options.aspiration is None
    for pos in _positions(7, 2):
        _assert_same(
            _run("paranoid", pos, 2, **VARIANTS["reduction"]),
            _run("paranoid", pos, 2, **VARIANTS["plain"]),
        )
        deep = _run("paranoid", pos, 6, **VARIANTS["reduction"])  # inexacte : coup légal seulement
        assert deep.best in pos.state().legal_moves()


def test_search_restores_the_state_on_any_error() -> None:
    pos = _positions(8, 3)[2]
    a = make_agent("paranoid", pos.state().current, pos.config, **VARIANTS["plain"])
    assert isinstance(a, SearchAgent)
    st = pos.state()
    key, snap = st.key, st.to_snapshot()
    calls = {"n": 0}
    original = a.valuation.scalar

    def failing(*args: object, **kw: object) -> float:
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("panne simulée")
        return original(*args, **kw)  # type: ignore[arg-type]

    a.valuation.scalar = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="panne"):
        a.run_search(st, 4)
    assert st.key == key and st.to_snapshot() == snap


@pytest.mark.parametrize(
    "params",
    [
        {"ordering": "random"},
        {"aspiration": 0},
        {"aspiration": True},
        {"aspiration": float("inf")},
        {"reduction": 3},
        {"reduction": True},
    ],
)
def test_option_validation(params: dict[str, object]) -> None:
    pos = _positions(7, 2)[0]
    with pytest.raises(ConfigError):
        make_agent("paranoid", 0, pos.config, **params)


def test_variant_comparison_tool_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    rows = compare_variants(_positions(7, 2), agent="paranoid", depth=5)
    assert [r["variant"] for r in rows] == list(VARIANTS)
    for r in rows:
        if r["exact"]:
            assert r["same_ties"] == r["same_exact_values"] == r["positions"]
    with pytest.raises(ValueError, match="inconnues"):
        compare_variants(_positions(7, 2), variants=("turbo",))
    code = main(
        ["bench", "--variants", "--grid", "7", "--players", "2", "--depth", "4", "--variant", "tt"]
    )
    assert code == 0 and "accord" in capsys.readouterr().out


def test_direct_search_api_accepts_new_options() -> None:
    pos = _positions(7, 2)[1]
    st: GameState = pos.state()
    a = make_agent("paranoid", st.current, pos.config)
    p = st.current
    signs = tuple(1 if q == p else -1 for q in range(st.n))
    opts = SearchOptions(ordering="mobility", iterative=True, tt=True, aspiration=0.01)
    result = search(
        "alphabeta",
        st,
        a.valuation,
        4,
        options=opts,
        coeffs=((p, 1.0),),
        signs=signs,
        camp=(p,),
    )
    plain = search(
        "alphabeta",
        st,
        a.valuation,
        4,
        options=SearchOptions(pruning=False),
        coeffs=((p, 1.0),),
        signs=signs,
        camp=(p,),
    )
    _assert_same(result, plain)
