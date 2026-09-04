from __future__ import annotations

from typing import Any

from ..http import request_json
from ..models import DateRange


BCB_PTAX_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
)


def fetch_fx_series(
    date_range: DateRange,
    *,
    currency: str = "USD",
) -> list[dict[str, Any]]:
    rows_by_date: dict[str, dict[str, Any]] = {}
    for year in date_range.years():
        start = max(date_range.start, date_range.start.replace(year=year, month=1, day=1))
        end = min(date_range.end, date_range.end.replace(year=year, month=12, day=31))
        if start > end:
            continue

        url = (
            f"{BCB_PTAX_URL}?"
            f"@moeda='{currency}'&"
            f"@dataInicial='{start.strftime('%m-%d-%Y')}'&"
            f"@dataFinalCotacao='{end.strftime('%m-%d-%Y')}'&"
            "$top=10000&$format=json&$select=cotacaoCompra,cotacaoVenda,dataHoraCotacao"
        )
        payload = request_json(url)
        for item in payload.get("value", []):
            raw_datetime = str(item.get("dataHoraCotacao", ""))
            if len(raw_datetime) < 10:
                continue
            day = raw_datetime[:10]
            rows_by_date[day] = {
                "date": day,
                "usd_brl": item.get("cotacaoVenda") or item.get("cotacaoCompra"),
                "fx_source": "BCB_PTAX",
            }

    return [rows_by_date[key] for key in sorted(rows_by_date)]
