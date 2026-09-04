"""Garante que as anomalias mensais nao usam informacao futura.

Correcao do checkpoint: a versao do TGSI I calculava a "normal" de cada
mes-calendario sobre a serie inteira (inclusive o futuro), o que vaza
informacao numa divisao temporal treino/teste. A versao corrigida usa
apenas o passado (janela expansiva).
"""
from __future__ import annotations

import unittest

from tgsi_pipeline.pipeline import _build_monthly_ml_features

ANOMALY_COLUMNS = [
    "precipitation_mm_month_anomaly",
    "temperature_mean_c_month_anomaly",
    "ndvi_month_anomaly",
    "soil_moisture_m3m3_month_anomaly",
    "usd_brl_month_anomaly",
    "brent_usd_bbl_month_anomaly",
]


def _series(n_years: int) -> list[dict]:
    rows: list[dict] = []
    for year in range(2010, 2010 + n_years):
        for month in range(1, 13):
            k = (year - 2010) * 12 + month
            rows.append(
                {
                    "date": f"{year}-{month:02d}-01",
                    "location_id": "loc1",
                    "usd_brl": 1.0 + 0.01 * k,
                    "brent_usd_bbl": 50.0 + k,
                    "precipitation_mm": 100.0 + (k % 7) * 10,
                    "temperature_mean_c": 20.0 + (k % 5),
                    "ndvi": 0.4 + 0.001 * k,
                    "soil_moisture_m3m3": 0.25 + 0.0005 * k,
                }
            )
    return rows


class CausalAnomalyTests(unittest.TestCase):
    def test_first_occurrence_of_each_month_has_no_anomaly(self) -> None:
        out = _build_monthly_ml_features(_series(3))
        first_12 = out[:12]
        for row in first_12:
            for col in ANOMALY_COLUMNS:
                self.assertIsNone(row[col], f"{row['date']} {col} deveria ser None (sem historico do mes)")

    def test_anomaly_equals_mean_of_strictly_past_same_month(self) -> None:
        rows = _series(4)
        out = _build_monthly_ml_features(rows)
        # 4o janeiro da serie (indice 36): normal = media dos 3 janeiros anteriores
        jan_year4 = out[36]
        self.assertEqual(jan_year4["date"], "2013-01-01")
        past_jan_usd = [r["usd_brl"] for r in rows[:36] if r["date"].endswith("-01-01")]
        self.assertEqual(len(past_jan_usd), 3)
        expected = round(jan_year4["usd_brl"] - sum(past_jan_usd) / 3, 6)
        self.assertAlmostEqual(jan_year4["usd_brl_month_anomaly"], expected, places=6)

    def test_future_rows_do_not_change_past_anomalies(self) -> None:
        """A propriedade anti-vazamento: ver mais dados no futuro nao pode
        alterar a anomalia ja calculada para um mes do passado."""
        short = _build_monthly_ml_features(_series(3))
        long = _build_monthly_ml_features(_series(6))
        for i in range(len(short)):
            for col in ANOMALY_COLUMNS:
                self.assertEqual(
                    short[i][col],
                    long[i][col],
                    f"linha {short[i]['date']} {col} mudou ao acrescentar anos futuros",
                )


if __name__ == "__main__":
    unittest.main()
