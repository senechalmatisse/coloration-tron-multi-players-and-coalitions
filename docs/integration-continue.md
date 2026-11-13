# Intégration continue et livraison (P14)

Le flux `.github/workflows/ci.yml` s'exécute sur chaque demande de fusion et chaque poussée sur `main` :

| Tâche | Commandes |
|---|---|
| Lint, format, types | `uv sync --extra dev` · `ruff check src tests` · `ruff format --check src tests` · `mypy src` |
| Tests, noyaux Numba | `pytest` |
| Tests, Python pur | `NUMBA_DISABLE_JIT=1 pytest` |
| Couverture | `NUMBA_DISABLE_JIT=1 pytest --cov=tron.core --cov=tron.eval --cov=tron.agents --cov-fail-under=80` (rapport `coverage.xml` en artefact) |
| Charge (lundi, ou à la demande) | `pytest -m load` · `tron bench --suite --reference benchmarks/reference/suite-python-20x20-4p.json` (résultat `bench.json` en artefact) |

- **Versions** : Python vient de `.python-version` (3.14, installé par `uv python install`) ; les dépendances suivent les bornes de `pyproject.toml` (Arcade épinglé en 3.3.3, comme dans le README). Le dépôt n'a pas de fichier de verrouillage : pour des installations identiques d'une exécution à l'autre, ajouter `uv.lock` et passer `uv sync --locked`.
- **Sans écran** : aucun test n'ouvre de fenêtre (Arcade est remplacé par un bouchon, le rendu est testé sur des tableaux NumPy, `MPLBACKEND=Agg` pour les figures). La fenêtre se valide à la main : `docs/validation-gui.md`.
- **Couverture** : mesurée en Python pur, car les fonctions compilées par Numba ne s'exécutent pas dans l'interpréteur et ne seraient pas comptées. Le seuil de 80 % porte sur `core`, `eval` et `agents` ; un nouveau module de ces paquets arrive avec ses tests plutôt qu'avec un seuil abaissé.
- **Aucun seuil de temps** dans les tâches ordinaires : les régressions fonctionnelles des calculs mesurés sont détectées par les sommes de contrôle de la suite de mesures (`docs/performance.md`, § 2).

Reproduire localement les mêmes vérifications :

```bash
uv sync --extra dev
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
uv run pytest && NUMBA_DISABLE_JIT=1 uv run pytest
NUMBA_DISABLE_JIT=1 uv run pytest --cov=tron.core --cov=tron.eval --cov=tron.agents --cov-fail-under=80
uv run pytest -m load        # facultatif, plus long
```

**État de cette version.** Le flux n'a pas encore été exécuté sur GitHub depuis l'environnement de développement (sans accès réseau ni dépôt distant). Localement (Python 3.13, sans Numba ni pytest-cov) : `ruff`, `mypy` et la suite de tests passent ; la couverture de lignes de `core` + `eval` + `agents`, estimée par un traceur `sys.monitoring` équivalent, est d'environ 95 %.
