"""Garante que o alvo t -> t+1 e' pareado por aritmetica de data, nao pela
"proxima linha disponivel" da serie de alvo.

Correcao da auditoria de 13/09/2026, item 16: se a serie CEPEA tiver uma
lacuna (um mes sem cotacao), pareamento por posicao associaria um mes ao
alvo de um mes mais distante no futuro, em vez de ficar em branco.
"""
from __future__ import annotations

import unittest

from tgsi_pipeline.pipeline import _attach_monthly_target


def _feature_row(date: str) -> dict:
    return {"date": date, "location_id": "loc1"}


def _target_row(date: str, brl: float) -> dict:
    return {
        "date": date,
        "soy_price_brl_bag": brl,
        "soy_price_usd_bag": brl / 5.0,
        "target_source": "CEPEA",
        "target_series_name": "CEPEA/ESALQ - Paranagua",
    }


class TargetPairingTests(unittest.TestCase):
    def test_next_month_is_paired_by_calendar_arithmetic(self) -> None:
        feature_rows = [_feature_row("2019-12-01"), _feature_row("2020-01-01")]
        target_rows = [
            _target_row("2019-12-01", 100.0),
            _target_row("2020-01-01", 110.0),
            _target_row("2020-02-01", 120.0),
        ]
        out = _attach_monthly_target(feature_rows, target_rows)
        dec = next(r for r in out if r["date"] == "2019-12-01")
        self.assertEqual(dec["soy_price_brl_bag_next_month"], 110.0)

    def test_gap_in_target_series_leaves_next_month_as_none(self) -> None:
        # Janeiro = 100, Fevereiro AUSENTE, Marco = 120.
        feature_rows = [_feature_row("2021-01-01")]
        target_rows = [
            _target_row("2021-01-01", 100.0),
            # 2021-02-01 propositalmente ausente
            _target_row("2021-03-01", 120.0),
        ]
        out = _attach_monthly_target(feature_rows, target_rows)
        jan = out[0]
        # NAO pode "pular" para o valor de marco so' porque e' a proxima
        # linha disponivel na serie de alvo.
        self.assertIsNone(jan["soy_price_brl_bag_next_month"])
        self.assertNotEqual(jan.get("soy_price_brl_bag_next_month"), 120.0)

    def test_last_month_of_series_has_no_next_month(self) -> None:
        feature_rows = [_feature_row("2026-04-01")]
        target_rows = [_target_row("2026-04-01", 128.0)]
        out = _attach_monthly_target(feature_rows, target_rows)
        self.assertIsNone(out[0]["soy_price_brl_bag_next_month"])


if __name__ == "__main__":
    unittest.main()
