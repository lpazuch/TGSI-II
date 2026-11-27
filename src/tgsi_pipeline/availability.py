"""Disponibilidade temporal das variaveis explicativas (REV-01).

A origem da previsao do mes t+1 e' o primeiro dia de t+1. A linha do mes t
so' pode conter informacao **publicada ate o ultimo dia de t** (regra
conservadora: nada publicado no proprio dia da origem entra).

Cada observacao bruta recebe uma data de disponibilidade::

    disponivel_em = fim_do_periodo_de_referencia + latencia

    fim_do_periodo_de_referencia = data + (period_days - 1)

e e' agregada no mes dessa data, nao no mes em que foi medida. Assim, para
fontes com atraso, a "janela de informacao" do mes t fica deslocada para tras
em ``latency_days`` dias, mas tem a mesma duracao de um mes. Isso mantem as
somas mensais (precipitacao) comparaveis entre meses.

As latencias abaixo sao limites conservadores, nao os atrasos medios. As
referencias estao em docs/methodology.md, secao 14.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .utils import month_start, parse_date_any


@dataclass(frozen=True)
class AvailabilityRule:
    period_days: int
    latency_days: int
    reference: str

    def available_on(self, observed: date) -> date:
        return observed + timedelta(days=self.period_days - 1 + self.latency_days)

    def window_start(self, month: date) -> date:
        """Primeira data de observacao cuja disponibilidade cai em ``month``."""
        return month - timedelta(days=self.period_days - 1 + self.latency_days)


RULES: dict[str, AvailabilityRule] = {
    # CEPEA e PTAX: publicacao no proprio dia do pregao.
    "fx": AvailabilityRule(1, 0, "BCB PTAX, divulgada no proprio dia util"),
    # EIA RBRTE: atualizacao semanal da tabela; uma cotacao de segunda-feira
    # pode sair so' na quarta-feira da semana seguinte (9 dias) -> 10 dias.
    "oil": AvailabilityRule(1, 10, "EIA RBRTE, atualizacao semanal"),
    # NASA POWER (GEOS-IT): "generally ready within about two days" -> 3 dias.
    "climate": AvailabilityRule(1, 3, "NASA POWER GEOS-IT, ~2 dias"),
    # SMAP SPL3SMP_E: latencia media ~1 dia 5 h (NSIDC) -> 3 dias.
    "soil_moisture": AvailabilityRule(1, 3, "NSIDC SPL3SMP_E, media ~1 dia"),
    # MOD13Q1: data do arquivo = INICIO da composicao de 16 dias. Latencia de
    # producao sem valor oficial (LP DAAC informa 2-3 dias em forum, com
    # atrasos maiores relatados) -> 16 dias apos o fim da composicao.
    "ndvi": AvailabilityRule(16, 16, "MODIS MOD13Q1.061, composicao de 16 dias"),
}


def shift_to_availability(
    rows: list[dict[str, Any]],
    rule: AvailabilityRule,
    *,
    extraction_start: date,
    require_full_window: bool,
) -> list[dict[str, Any]]:
    """Copia ``rows`` trocando ``date`` pela data de disponibilidade.

    Com ``require_full_window``, descarta as observacoes dos meses cuja janela
    de informacao comeca antes de ``extraction_start`` (o mes ficaria com
    menos dias que os demais e, numa soma, com valor artificialmente baixo).
    """
    output: list[dict[str, Any]] = []
    for row in rows:
        observed = parse_date_any(str(row["date"]))
        available = rule.available_on(observed)
        if require_full_window and rule.window_start(month_start(available)) < extraction_start:
            continue
        shifted = dict(row)
        shifted["date"] = available.isoformat()
        shifted["observed_date"] = observed.isoformat()
        output.append(shifted)
    return output


def forecast_origin(row_month: date) -> date:
    """Primeiro dia do mes seguinte ao da linha (mes-alvo)."""
    year = row_month.year + (1 if row_month.month == 12 else 0)
    month = 1 if row_month.month == 12 else row_month.month + 1
    return date(year, month, 1)
