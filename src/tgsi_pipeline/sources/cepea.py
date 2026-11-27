from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import xlrd


def fetch_soybean_prices_from_xls(
    path: str | Path,
    *,
    price_column_brl: str = "À vista R$",
    price_column_usd: str = "À vista US$",
    series_name: str = "CEPEA/ESALQ - Paranaguá",
) -> list[dict[str, Any]]:
    workbook = xlrd.open_workbook(str(path), ignore_workbook_corruption=True)
    sheet = workbook.sheet_by_index(0)
    header_row_index = _find_header_row(sheet, ["Data", price_column_brl, price_column_usd])
    headers = [str(value).strip() for value in sheet.row_values(header_row_index)]

    try:
        date_idx = headers.index("Data")
        brl_idx = headers.index(price_column_brl)
        usd_idx = headers.index(price_column_usd)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError("Não foi possível localizar as colunas esperadas na planilha CEPEA.") from exc

    rows: list[dict[str, Any]] = []
    for row_index in range(header_row_index + 1, sheet.nrows):
        values = sheet.row_values(row_index)
        if not values or all(str(value).strip() == "" for value in values):
            continue
        date_value = str(values[date_idx]).strip()
        if not date_value:
            continue
        try:
            parsed_date = datetime.strptime(date_value, "%d/%m/%Y").date().isoformat()
        except ValueError:
            continue
        rows.append(
            {
                "date": parsed_date,
                "soy_price_brl_bag": _as_float(values[brl_idx]),
                "soy_price_usd_bag": _as_float(values[usd_idx]),
                "target_source": "CEPEA",
                "target_series_name": series_name,
                "target_frequency": "daily",
            }
        )
    return rows


def resample_monthly(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Media aritmetica mensal dos pregoes.

    Cada preco (R$ e US$) tem seu proprio contador: um pregao com valor
    ausente e' ignorado na media daquela coluna, em vez de entrar como 0
    (REV-02). Um mes sem nenhum valor valido fica com None.
    """
    fields = ("soy_price_brl_bag", "soy_price_usd_bag")
    meta: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        month = str(row["date"])[:7] + "-01"
        meta.setdefault(
            month,
            {
                "target_source": row.get("target_source"),
                "target_series_name": row.get("target_series_name"),
            },
        )
        bucket = values.setdefault(month, {field: [] for field in fields})
        for field in fields:
            value = _as_float(row.get(field))
            if value is not None:
                bucket[field].append(value)

    output: list[dict[str, Any]] = []
    for month in sorted(values):
        out: dict[str, Any] = {"date": month}
        for field in fields:
            series = values[month][field]
            out[field] = round(sum(series) / len(series), 6) if series else None
        out.update(meta[month])
        out["target_frequency"] = "monthly"
        output.append(out)
    return output


def _find_header_row(sheet: Any, expected_headers: list[str]) -> int:
    for row_index in range(min(sheet.nrows, 20)):
        headers = [str(value).strip() for value in sheet.row_values(row_index)]
        if all(header in headers for header in expected_headers):
            return row_index
    raise ValueError("Cabeçalho da planilha CEPEA não encontrado.")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
