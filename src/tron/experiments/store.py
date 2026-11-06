"""Stockage incrémental des résultats en CSV, avec reprise (F5.4, F5.5)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Any


class ResultStore:
    """Fichier CSV d'une ligne par partie, écrit au fil de l'eau.

    À l'ouverture, une dernière ligne incomplète (interruption pendant
    l'écriture) est supprimée ; ``completed()`` liste les parties déjà jouées
    pour la reprise.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fields: list[str] | None = None

    def _repair(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n")
            self.path.write_bytes(data[: cut + 1] if cut >= 0 else b"")

    def header(self) -> list[str] | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open(newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), None)

    def rows(self) -> list[dict[str, str]]:
        if self.header() is None:
            return []
        with self.path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def completed(self) -> set[tuple[str, int]]:
        """Couples (condition, partie) déjà enregistrés."""
        self._repair()
        done = set()
        for row in self.rows():
            if row.get("condition") is not None and row.get("game", "").isdigit():
                done.add((row["condition"], int(row["game"])))
        return done

    def append(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._open(list(row))
        assert self._writer is not None and self._fh is not None
        if list(row) != self._fields:
            raise ValueError("colonnes incompatibles avec le fichier de résultats existant")
        self._writer.writerow(row)
        self._fh.flush()

    def _open(self, fields: list[str]) -> None:
        self._repair()
        existing = self.header()
        if existing is not None and existing != fields:
            raise ValueError(
                f"{self.path} existe avec d'autres colonnes ; choisissez un autre dossier de sortie"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        self._fields = fields
        if existing is None:
            self._writer.writeheader()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
