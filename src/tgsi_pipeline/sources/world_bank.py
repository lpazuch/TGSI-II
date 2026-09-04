from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import urljoin

from ..http import request_bytes, request_text
from ..models import DateRange
from ..utils import normalize_text, parse_date_any, parse_float
from ..xlsx_reader import read_first_sheet_rows


PINK_SHEET_PAGE = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "world-bank-commodities-price-data-the-pink-sheet"
)


def fetch_brent_monthly(date_range: DateRange) -> list[dict[str, Any]]:
    page = request_text(PINK_SHEET_PAGE)
    workbook_url = _discover_workbook_url(page)
    workbook = request_bytes(workbook_url)
    rows = read_first_sheet_rows(workbook, preferred_sheet_name="Monthly Prices")

    brent_column = _find_brent_column(rows)
    output: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        try:
            current_date = parse_date_any(row[0])
        except ValueError:
            continue
        amount = parse_float(row[brent_column] if brent_column < len(row) else None)
        if amount is None:
            continue
        if not (date_range.start <= current_date <= date_range.end):
            continue
        output.append(
            {
                "date": current_date.isoformat(),
                "brent_usd_bbl": amount,
                "oil_source": "WORLD_BANK_PINK_SHEET",
                "oil_frequency": "monthly",
            }
        )
    if not output:
        raise ValueError("Nao foi possivel localizar a serie de Brent no arquivo Pink Sheet.")
    return output


def _discover_workbook_url(page_html: str) -> str:
    pattern = re.compile(
        r'href=[\'"]([^\'"]*CMO-Historical-Data-Monthly\.xlsx[^\'"]*)[\'"]',
        re.IGNORECASE,
    )
    match = pattern.search(page_html)
    if not match:
        raise ValueError("Link do arquivo mensal do Pink Sheet nao encontrado.")
    return urljoin(PINK_SHEET_PAGE, unescape(match.group(1)))


def _find_brent_column(rows: list[list[str]]) -> int:
    for row in rows[:10]:
        for index, value in enumerate(row):
            probe = normalize_text(value)
            if "crude oil" in probe and "brent" in probe:
                return index
    raise ValueError("Coluna de Brent nao encontrada no Pink Sheet.")
