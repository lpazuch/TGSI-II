from __future__ import annotations

import csv
from pathlib import Path

from .utils import ensure_directory, unique_fieldnames


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    fieldnames = unique_fieldnames(rows) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)
