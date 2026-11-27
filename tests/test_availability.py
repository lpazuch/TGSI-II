"""REV-01: nenhuma variavel da linha do mes t pode usar informacao que so'
fica disponivel na origem da previsao (1o dia de t+1) ou depois.

Inclui o caso que motivou a correcao: a composicao MOD13Q1 iniciada em 22/03
cobre ate 06/04 e, antes, era atribuida a marco, entrando na previsao de
abril (o proprio mes-alvo).
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path

from tgsi_pipeline.availability import RULES, forecast_origin, shift_to_availability
from tgsi_pipeline.models import DateRange, Location, ProjectConfig
from tgsi_pipeline.pipeline import _build_monthly_features
from tgsi_pipeline.sources import cepea
from tgsi_pipeline.utils import daterange, month_start

PROJECT_ROOT = Path(__file__).resolve().parent.parent
START = date(2020, 1, 1)
END = date(2020, 12, 31)


def _config() -> ProjectConfig:
    from tgsi_pipeline.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "dataset.hybrid_2006plus.json")
    config.date_range = DateRange(start=START, end=END)
    return config


def _daily(field: str, source_field: str) -> list[dict]:
    # valor = ordinal da data observada -> permite recuperar o dia observado
    return [
        {"date": d.isoformat(), "location_id": "sorriso_mt", field: float(d.toordinal()),
         source_field: "test"}
        for d in daterange(START, END)
    ]


def _ndvi_composites() -> list[dict]:
    rows = []
    for year in (2020,):
        doy = 1
        while doy <= 353:
            d = date(year, 1, 1) + timedelta(days=doy - 1)
            rows.append({"date": d.isoformat(), "location_id": "sorriso_mt",
                         "ndvi": float(d.toordinal()), "ndvi_source": "test"})
            doy += 16
    return rows


class AvailabilityRuleTests(unittest.TestCase):
    def test_ndvi_uses_end_of_composite_plus_latency(self) -> None:
        rule = RULES["ndvi"]
        # composicao 22/03/2026 a 06/04/2026 -> disponivel em 22/04/2026
        self.assertEqual(rule.available_on(date(2026, 3, 22)), date(2026, 4, 22))

    def test_every_rule_is_at_least_as_slow_as_documented_latency(self) -> None:
        self.assertGreaterEqual(RULES["climate"].latency_days, 2)
        self.assertGreaterEqual(RULES["soil_moisture"].latency_days, 2)
        self.assertGreaterEqual(RULES["oil"].latency_days, 9)
        self.assertEqual(RULES["ndvi"].period_days, 16)

    def test_shift_records_observed_date(self) -> None:
        rows = [{"date": "2020-03-30", "x": 1.0}]
        out = shift_to_availability(rows, RULES["climate"], extraction_start=START,
                                    require_full_window=True)
        self.assertEqual(out[0]["observed_date"], "2020-03-30")
        self.assertEqual(out[0]["date"], "2020-04-02")


class MonthlyFeaturesRespectOriginTests(unittest.TestCase):
    """Para cada linha do mes t, toda observacao agregada tem de estar
    disponivel antes do 1o dia de t+1. Os valores sinteticos sao o ordinal da
    data observada; a media/maximo de t revela as datas usadas."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _build_monthly_features(
            _config(),
            _daily("temperature_max_c", "climate_source"),
            [{"date": d.isoformat(), "usd_brl": float(d.toordinal()), "fx_source": "t"}
             for d in daterange(START, END)],
            [{"date": d.isoformat(), "brent_usd_bbl": float(d.toordinal()),
              "oil_source": "t", "oil_frequency": "daily"} for d in daterange(START, END)],
            _ndvi_composites(),
            _daily("soil_moisture_m3m3", "soil_moisture_source"),
        )

    def _check(self, field: str, rule_name: str, use_max: bool = False) -> None:
        rule = RULES[rule_name]
        checked = 0
        for row in self.rows:
            value = row.get(field)
            if value is None:
                continue
            origin = forecast_origin(date.fromisoformat(row["date"]))
            # para o maximo, o proprio valor e' a ultima data usada; para a
            # media de datas consecutivas, a ultima data e' >= media
            last_observed = date.fromordinal(int(round(value)))
            self.assertLess(rule.available_on(last_observed), origin,
                            f"{field} de {row['date']} usa dado disponivel em/apos {origin}")
            checked += 1
        self.assertGreater(checked, 6)

    def test_climate_max_uses_only_data_available_before_origin(self) -> None:
        self._check("temperature_max_c", "climate", use_max=True)

    def test_climate_window_is_shifted_by_latency(self) -> None:
        june = next(r for r in self.rows if r["date"] == "2020-06-01")
        # ultimo dia observado na janela de junho: 30/06 - 3 dias = 27/06
        self.assertEqual(date.fromordinal(int(june["temperature_max_c"])), date(2020, 6, 27))

    def test_ndvi_never_includes_target_month(self) -> None:
        for row in self.rows:
            if row.get("ndvi") is None:
                continue
            origin = forecast_origin(date.fromisoformat(row["date"]))
            composite_starts = [
                date.fromordinal(int(c["ndvi"])) for c in _ndvi_composites()
                if month_start(RULES["ndvi"].available_on(date.fromisoformat(c["date"])))
                == date.fromisoformat(row["date"])
            ]
            self.assertTrue(composite_starts)
            for start in composite_starts:
                composite_end = start + timedelta(days=15)
                self.assertLess(composite_end, origin)
                self.assertLess(RULES["ndvi"].available_on(start), origin)

    def test_march_composite_spilling_into_april_is_not_in_march(self) -> None:
        # composicao 2020-03-21 (DOY 81) cobre ate 05/04 -> nao pode estar em marco
        march = next(r for r in self.rows if r["date"] == "2020-03-01")
        if march.get("ndvi") is not None:
            self.assertNotAlmostEqual(march["ndvi"], float(date(2020, 3, 21).toordinal()))
            self.assertLess(march["ndvi"], float(date(2020, 3, 21).toordinal()))

    def test_brent_and_soil_moisture_respect_latency(self) -> None:
        for field, rule in (("brent_usd_bbl", "oil"), ("soil_moisture_m3m3", "soil_moisture")):
            for row in self.rows:
                value = row.get(field)
                if value is None:
                    continue
                origin = forecast_origin(date.fromisoformat(row["date"]))
                # media de datas consecutivas: ultima data = 2*media - primeira;
                # basta conferir a janela pela regra
                window_end = origin - timedelta(days=1 + RULES[rule].latency_days)
                window_start = RULES[rule].window_start(date.fromisoformat(row["date"]))
                expected = (window_start.toordinal() + window_end.toordinal()) / 2
                self.assertAlmostEqual(value, expected, places=4, msg=f"{field} {row['date']}")

    def test_first_month_with_incomplete_window_is_empty(self) -> None:
        jan = next(r for r in self.rows if r["date"] == "2020-01-01")
        self.assertIsNone(jan.get("temperature_max_c"))
        self.assertIsNone(jan.get("brent_usd_bbl"))
        # cambio nao tem latencia: janeiro completo
        self.assertIsNotNone(jan.get("usd_brl"))


class CepeaMonthlyMeanTests(unittest.TestCase):
    def test_missing_price_is_not_counted_as_zero(self) -> None:
        rows = [
            {"date": "2021-05-03", "soy_price_brl_bag": 100.0, "soy_price_usd_bag": 20.0},
            {"date": "2021-05-04", "soy_price_brl_bag": None, "soy_price_usd_bag": 22.0},
            {"date": "2021-05-05", "soy_price_brl_bag": 110.0, "soy_price_usd_bag": ""},
        ]
        out = cepea.resample_monthly(rows)
        self.assertEqual(out[0]["soy_price_brl_bag"], 105.0)
        self.assertEqual(out[0]["soy_price_usd_bag"], 21.0)

    def test_month_without_valid_price_is_none(self) -> None:
        out = cepea.resample_monthly([{"date": "2021-06-01", "soy_price_brl_bag": None,
                                       "soy_price_usd_bag": None}])
        self.assertIsNone(out[0]["soy_price_brl_bag"])
        self.assertIsNone(out[0]["soy_price_usd_bag"])


if __name__ == "__main__":
    unittest.main()
