from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from ..html_tables import parse_html_tables
from ..http import request_text
from ..models import DateRange
from ..utils import parse_float


EIA_BRENT_DAILY_URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=d&n=PET&s=RBRTE"
EIA_BRENT_MONTHLY_URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=m&n=PET&s=RBRTE"

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def fetch_brent_series(
    date_range: DateRange,
    *,
    frequency: str,
) -> list[dict[str, Any]]:
    if frequency == "daily":
        return _fetch_daily(date_range)
    if frequency == "monthly":
        return _fetch_monthly(date_range)
    raise ValueError(f"Frequencia de Brent invalida: {frequency}")


def _fetch_daily(date_range: DateRange) -> list[dict[str, Any]]:
    html = request_text(EIA_BRENT_DAILY_URL)
    tables = parse_html_tables(html)
    table = _select_daily_table(tables)

    rows: list[dict[str, Any]] = []
    for row in table[1:]:
        padded = row + [""] * (6 - len(row))
        week_cell = padded[0].strip()
        if not week_cell:
            continue

        week_dates = _week_dates(week_cell)
        if len(week_dates) != 5:
            continue

        for current_date, value in zip(week_dates, padded[1:6]):
            amount = parse_float(value)
            if amount is None:
                continue
            if not (date_range.start <= current_date <= date_range.end):
                continue
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "brent_usd_bbl": amount,
                    "oil_source": "EIA",
                    "oil_frequency": "daily",
                }
            )
    return rows


def _fetch_monthly(date_range: DateRange) -> list[dict[str, Any]]:
    html = request_text(EIA_BRENT_MONTHLY_URL)
    tables = parse_html_tables(html)
    table = _select_monthly_table(tables)

    rows: list[dict[str, Any]] = []
    month_headers = table[0][1:13]
    for row in table[1:]:
        if not row or not row[0].strip().isdigit():
            continue
        year = int(row[0].strip())
        values = (row[1:13] + [""] * 12)[:12]
        for month_name, value in zip(month_headers, values):
            month_number = MONTHS.get(month_name.strip().lower()[:3])
            amount = parse_float(value)
            if month_number is None or amount is None:
                continue
            current_date = date(year, month_number, 1)
            if not (date_range.start <= current_date <= date_range.end):
                continue
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "brent_usd_bbl": amount,
                    "oil_source": "EIA",
                    "oil_frequency": "monthly",
                }
            )
    return rows


def _select_daily_table(tables: list[list[list[str]]]) -> list[list[str]]:
    for table in tables:
        if not table:
            continue
        header = [cell.strip().lower() for cell in table[0]]
        first = header[0] if header else ""
        if "week" in first and "mon" in header and "fri" in header:
            return table
    raise ValueError("Nao foi possivel localizar a tabela diaria do Brent na EIA.")


def _select_monthly_table(tables: list[list[list[str]]]) -> list[list[str]]:
    for table in tables:
        if not table:
            continue
        header = [cell.strip().lower() for cell in table[0]]
        if header[:4] == ["year", "jan", "feb", "mar"]:
            return table
    raise ValueError("Nao foi possivel localizar a tabela mensal do Brent na EIA.")


def _week_dates(value: str) -> list[date]:
    compact = " ".join(value.replace("\xa0", " ").split())
    pattern = re.compile(
        r"(\d{4})\s+([A-Za-z]{3})-\s*(\d{1,2})\s+to\s+([A-Za-z]{3})-\s*(\d{1,2})",
        re.IGNORECASE,
    )
    match = pattern.search(compact)
    if not match:
        return []

    year = int(match.group(1))
    start_month = MONTHS[match.group(2).lower()]
    start_day = int(match.group(3))
    end_month = MONTHS[match.group(4).lower()]
    start_year = year
    if start_month == 12 and end_month == 1:
        start_year = year - 1

    start_date = date(start_year, start_month, start_day)
    monday = start_date - timedelta(days=start_date.weekday())
    return [monday + timedelta(days=offset) for offset in range(5)]
