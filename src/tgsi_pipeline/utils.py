from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def daterange(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_month_starts(start: date, end: date) -> Iterator[date]:
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().strip().split())


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text or text in {"-", "--", "NA", "N/A", "null", "None"}:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            candidate = text.replace(".", "").replace(",", ".")
        else:
            candidate = text.replace(",", "")
    elif "," in text:
        candidate = text.replace(".", "").replace(",", ".")
    else:
        candidate = text
    try:
        return float(candidate)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def coalesce(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def parse_date_any(value: str) -> date:
    raw = value.strip()
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y%m%d",
        "%b %Y",
        "%b-%y",
        "%b-%Y",
        "%Y-%m",
        "%Y/%m",
        "%YM%m",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt).date()
            if fmt in {"%b %Y", "%b-%y", "%b-%Y", "%Y-%m", "%Y/%m", "%YM%m"}:
                return date(parsed.year, parsed.month, 1)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"Formato de data nao reconhecido: {value!r}")


def unique_fieldnames(rows: Iterable[dict[str, object]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered
