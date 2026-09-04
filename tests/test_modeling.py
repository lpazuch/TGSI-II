from __future__ import annotations

import unittest

from tgsi_pipeline.modeling import evaluate_walk_forward


class ModelingEvaluationTest(unittest.TestCase):
    def test_walk_forward_returns_metrics_and_predictions(self) -> None:
        rows = [
            {
                "date": f"2024-{month:02d}-01",
                "target_month": f"2024-{month + 1:02d}-01" if month < 12 else "2025-01-01",
                "target_variable": "soy_price_brl_bag_next_month",
                "target_value": 100.0 + month,
                "target_source": "CEPEA",
                "target_series_name": "CEPEA/ESALQ - Paranaguá",
                "soy_price_brl_bag": 99.0 + month,
                "usd_brl": 5.0 + month * 0.01,
                "brent_usd_bbl": 80.0 + month,
                "month": float(month),
                "quarter": float(((month - 1) // 3) + 1),
                "soy_planting_window": 1.0 if month in {9, 10, 11, 12} else 0.0,
                "soy_harvest_window": 1.0 if month in {1, 2, 3, 4} else 0.0,
                "precipitation_mm": 100.0 + month * 2,
                "temperature_mean_c": 26.0 + month * 0.1,
                "relative_humidity_pct": 65.0 + month * 0.2,
                "ndvi": 0.55 + month * 0.01,
                "soil_moisture_m3m3": 0.22 + month * 0.005,
                "usd_brl_lag_1": 4.9 + month * 0.01 if month > 1 else None,
                "brent_usd_bbl_lag_1": 79.0 + month if month > 1 else None,
                "precipitation_mm_lag_1": 98.0 + month * 2 if month > 1 else None,
                "temperature_mean_c_lag_1": 25.9 + month * 0.1 if month > 1 else None,
                "relative_humidity_pct_lag_1": 64.8 + month * 0.2 if month > 1 else None,
                "ndvi_lag_1": 0.54 + month * 0.01 if month > 1 else None,
                "soil_moisture_m3m3_lag_1": 0.215 + month * 0.005 if month > 1 else None,
                "usd_brl_rolling_mean_3": 4.95 + month * 0.01,
                "brent_usd_bbl_rolling_mean_3": 79.5 + month,
                "precipitation_mm_rolling_mean_3": 99.0 + month * 2,
                "temperature_mean_c_rolling_mean_3": 25.95 + month * 0.1,
                "relative_humidity_pct_rolling_mean_3": 64.9 + month * 0.2,
                "ndvi_rolling_mean_3": 0.545 + month * 0.01,
                "soil_moisture_m3m3_rolling_mean_3": 0.217 + month * 0.005,
            }
            for month in range(1, 9)
        ]

        evaluation = evaluate_walk_forward(rows, min_train_size=4)

        self.assertEqual(evaluation["min_train_size"], 4)
        self.assertEqual(len(evaluation["predictions"]), 4)
        self.assertIn("naive_last_value", evaluation["metrics"])
        self.assertIn("rolling_mean_3", evaluation["metrics"])
        self.assertIn("linear_regression", evaluation["metrics"])
        self.assertIn("arima_1_1_0", evaluation["metrics"])
        self.assertIn("sarimax_exog", evaluation["metrics"])
        self.assertIn("random_forest", evaluation["metrics"])
        self.assertIn("ensemble_arima_rf", evaluation["metrics"])
        self.assertIn("lstm_window_12", evaluation["metrics"])
        self.assertEqual(evaluation["metrics"]["naive_last_value"]["count"], 4)


if __name__ == "__main__":
    unittest.main()
