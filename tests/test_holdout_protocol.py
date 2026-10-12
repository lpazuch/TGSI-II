"""Garante que ARIMA, Random Forest e XGBoost percorrem EXATAMENTE as mesmas
12 origens de previsao no holdout, sob o mesmo protocolo de walk-forward
expansivo de 1 passo (treina com [0, i), preve i, reajusta).

Correcao da auditoria (11/10/2026, item 18): o ARIMA era reajustado mes a
mes, mas RF/XGBoost eram treinados uma unica vez antes dos 12 meses de teste.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "holdout_12m", PROJECT_ROOT / "experiments" / "holdout_12m.py"
)
holdout_12m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout_12m)

N_ROWS = 48
HOLDOUT = holdout_12m.HOLDOUT_MONTHS
FEATURES = ["soy_price_brl_bag", "usd_brl", "usd_brl_lag_1"]


def _synthetic_target_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=N_ROWS, freq="MS")
    price = 100 + np.cumsum(rng.normal(0, 1, N_ROWS + 1))
    usd = 5 + np.cumsum(rng.normal(0, 0.05, N_ROWS))
    return pd.DataFrame(
        {
            "date": dates,
            "target_month": dates + pd.offsets.MonthBegin(1),
            "target_variable": holdout_12m.TARGET_VARIABLE,
            "target_value": price[1:],
            "soy_price_brl_bag": price[:-1],
            "usd_brl": usd,
            "usd_brl_lag_1": np.r_[np.nan, usd[:-1]],
        }
    )


def _recording_forecaster(calls: list):
    def forecaster(train, test_row, features):
        calls.append((train.copy(), test_row.copy()))
        return float(train["target_value"].iloc[-1])

    return forecaster


class HoldoutProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _synthetic_target_df()

    def test_origins_are_the_last_holdout_rows(self) -> None:
        origins = holdout_12m.walk_forward_origins(N_ROWS, HOLDOUT)
        self.assertEqual(origins, list(range(N_ROWS - HOLDOUT, N_ROWS)))
        self.assertEqual(len(origins), 12)

    def test_every_model_sees_same_origins_and_expanding_training_set(self) -> None:
        calls = {"arima_1_1_0": [], "random_forest": [], "xgboost": []}
        forecasters = {name: _recording_forecaster(c) for name, c in calls.items()}
        _, _, origins_df = holdout_12m.run_models(self.df, FEATURES, HOLDOUT, forecasters)

        expected_origins = list(range(N_ROWS - HOLDOUT, N_ROWS))
        for name, model_calls in calls.items():
            # um reajuste por origem, nao um unico treino antes do holdout
            self.assertEqual(len(model_calls), HOLDOUT, name)
            for i, (train, test_row) in zip(expected_origins, model_calls):
                # treino = exatamente as linhas [0, i); teste = linha i
                pd.testing.assert_frame_equal(train, self.df.iloc[:i])
                pd.testing.assert_frame_equal(test_row, self.df.iloc[i : i + 1])
                # sem vazamento: todo alvo do treino ja e' conhecido em date[i]
                self.assertLessEqual(train["target_month"].max(), test_row["date"].iloc[0])

        per_model = {
            name: g.drop(columns="model").reset_index(drop=True)
            for name, g in origins_df.groupby("model")
        }
        self.assertEqual(set(per_model), set(calls))
        reference = per_model["arima_1_1_0"]
        self.assertEqual(reference["origin_index"].tolist(), expected_origins)
        for name, frame in per_model.items():
            pd.testing.assert_frame_equal(frame, reference, obj=name)

    def test_real_forecasters_share_origins_and_refit_each_month(self) -> None:
        factory_calls = {"random_forest": 0, "xgboost": 0}
        original_rf = holdout_12m.random_forest_factory
        original_xgb = holdout_12m.xgboost_factory

        def counting(name, factory):
            def wrapped():
                factory_calls[name] += 1
                return factory()

            return wrapped

        forecasters = {
            "arima_1_1_0": holdout_12m.arima_forecaster,
            "random_forest": holdout_12m.make_tabular_forecaster(
                counting("random_forest", original_rf)
            ),
        }
        if holdout_12m.HAS_XGBOOST:
            forecasters["xgboost"] = holdout_12m.make_tabular_forecaster(
                counting("xgboost", original_xgb)
            )

        predictions, metrics_df, origins_df = holdout_12m.run_models(
            self.df, FEATURES, HOLDOUT, forecasters
        )

        self.assertEqual(len(predictions), HOLDOUT)
        self.assertEqual(set(metrics_df["model"]), set(forecasters))
        for name in forecasters:
            self.assertFalse(predictions[name].isna().any(), name)
        self.assertEqual(factory_calls["random_forest"], HOLDOUT)
        if holdout_12m.HAS_XGBOOST:
            self.assertEqual(factory_calls["xgboost"], HOLDOUT)

        origin_sets = {
            name: tuple(g["forecast_target_month"]) for name, g in origins_df.groupby("model")
        }
        self.assertEqual(len(set(origin_sets.values())), 1)
        self.assertEqual(
            next(iter(origin_sets.values())), tuple(predictions["target_month"])
        )

    def test_default_forecasters_cover_the_three_models(self) -> None:
        names = set(holdout_12m.build_forecasters())
        expected = {"arima_1_1_0", "random_forest"}
        if holdout_12m.HAS_XGBOOST:
            expected.add("xgboost")
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
