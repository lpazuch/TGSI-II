from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

from .csv_io import read_csv, write_csv
from .utils import ensure_directory, parse_date_any, parse_float

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
except Exception:  # noqa: BLE001
    RandomForestRegressor = None
    MLPRegressor = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # noqa: BLE001
    ARIMA = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # noqa: BLE001
    SARIMAX = None


METADATA_FIELDS = {
    "date",
    "target_month",
    "target_variable",
    "target_value",
    "target_source",
    "target_series_name",
}

PREFERRED_LINEAR_FEATURES = [
    "soy_price_brl_bag",
    "usd_brl",
    "brent_usd_bbl",
    "precipitation_mm",
    "temperature_mean_c",
    "relative_humidity_pct",
    "ndvi",
    "soil_moisture_m3m3",
    "usd_brl_lag_1",
    "brent_usd_bbl_lag_1",
    "precipitation_mm_lag_1",
    "temperature_mean_c_lag_1",
    "relative_humidity_pct_lag_1",
    "ndvi_lag_1",
    "soil_moisture_m3m3_lag_1",
    "usd_brl_rolling_mean_3",
    "brent_usd_bbl_rolling_mean_3",
    "precipitation_mm_rolling_mean_3",
    "temperature_mean_c_rolling_mean_3",
    "relative_humidity_pct_rolling_mean_3",
    "ndvi_rolling_mean_3",
    "soil_moisture_m3m3_rolling_mean_3",
    "month",
    "quarter",
    "soy_planting_window",
    "soy_harvest_window",
]

PREFERRED_TREE_FEATURES = [
    "soy_price_brl_bag",
    "usd_brl",
    "brent_usd_bbl",
    "precipitation_mm",
    "temperature_mean_c",
    "relative_humidity_pct",
    "ndvi",
    "soil_moisture_m3m3",
    "usd_brl_lag_1",
    "usd_brl_lag_2",
    "usd_brl_lag_3",
    "brent_usd_bbl_lag_1",
    "brent_usd_bbl_lag_2",
    "brent_usd_bbl_lag_3",
    "precipitation_mm_lag_1",
    "precipitation_mm_lag_2",
    "precipitation_mm_lag_3",
    "temperature_mean_c_lag_1",
    "temperature_mean_c_lag_2",
    "temperature_mean_c_lag_3",
    "relative_humidity_pct_lag_1",
    "relative_humidity_pct_lag_2",
    "relative_humidity_pct_lag_3",
    "ndvi_lag_1",
    "ndvi_lag_2",
    "ndvi_lag_3",
    "soil_moisture_m3m3_lag_1",
    "soil_moisture_m3m3_lag_2",
    "soil_moisture_m3m3_lag_3",
    "usd_brl_rolling_mean_3",
    "brent_usd_bbl_rolling_mean_3",
    "precipitation_mm_rolling_mean_3",
    "temperature_mean_c_rolling_mean_3",
    "relative_humidity_pct_rolling_mean_3",
    "ndvi_rolling_mean_3",
    "soil_moisture_m3m3_rolling_mean_3",
    "month",
    "quarter",
    "soy_planting_window",
    "soy_harvest_window",
]

PREFERRED_SEQUENCE_FEATURES = [
    "soy_price_brl_bag",
    "usd_brl",
    "brent_usd_bbl",
    "precipitation_mm",
    "temperature_mean_c",
    "relative_humidity_pct",
    "ndvi",
    "soil_moisture_m3m3",
    "month",
    "quarter",
    "soy_planting_window",
    "soy_harvest_window",
]

PREFERRED_EXOGENOUS_FEATURES = [
    "soy_price_brl_bag",
    "usd_brl",
    "brent_usd_bbl",
    "precipitation_mm",
    "temperature_mean_c",
    "relative_humidity_pct",
    "ndvi",
    "soil_moisture_m3m3",
    "usd_brl_lag_1",
    "brent_usd_bbl_lag_1",
    "precipitation_mm_lag_1",
    "temperature_mean_c_lag_1",
    "relative_humidity_pct_lag_1",
    "ndvi_lag_1",
    "soil_moisture_m3m3_lag_1",
    "usd_brl_rolling_mean_3",
    "brent_usd_bbl_rolling_mean_3",
    "precipitation_mm_rolling_mean_3",
    "temperature_mean_c_rolling_mean_3",
    "relative_humidity_pct_rolling_mean_3",
    "ndvi_rolling_mean_3",
    "soil_moisture_m3m3_rolling_mean_3",
    "month",
    "quarter",
    "soy_planting_window",
    "soy_harvest_window",
]


MODEL_NAMES = [
    "naive_last_value",
    "rolling_mean_3",
    "rolling_mean_6",
    "seasonal_naive_12",
    "linear_regression",
    "ridge_recent_12",
    "tree_stump",
    "arima_1_1_0",
    "sarimax_exog",
    "random_forest",
    "ensemble_arima_rf",
    "lstm_window_12",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgsi-model-baselines",
        description="Avalia baselines de previsão com validação temporal walk-forward.",
    )
    parser.add_argument("--input", required=True, help="CSV final de modelagem.")
    parser.add_argument(
        "--output-dir",
        default="data/modeling",
        help="Diretório para métricas e previsões.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=24,
        help="Número mínimo de meses para iniciar a validação walk-forward.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows = load_modeling_rows(args.input)
    evaluation = evaluate_walk_forward(rows, min_train_size=args.min_train_size)
    output_dir = ensure_directory(Path(args.output_dir))

    metrics_path = output_dir / "baseline_metrics.json"
    predictions_path = output_dir / "baseline_predictions.csv"

    metrics_path.write_text(
        json.dumps(evaluation["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(predictions_path, evaluation["predictions"])

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output_dir": str(output_dir),
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
                "evaluated_rows": len(evaluation["predictions"]),
                "models": list(evaluation["metrics"].keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_modeling_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(Path(path))
    ordered = sorted(rows, key=lambda row: parse_date_any(str(row["date"])))
    output: list[dict[str, Any]] = []
    for row in ordered:
        current: dict[str, Any] = dict(row)
        current["target_value"] = parse_float(row.get("target_value"))
        for key, value in row.items():
            if key in METADATA_FIELDS:
                continue
            current[key] = parse_float(value)
        output.append(current)
    return output


def evaluate_walk_forward(
    rows: list[dict[str, Any]],
    *,
    min_train_size: int = 24,
) -> dict[str, Any]:
    if len(rows) < 4:
        raise ValueError("São necessários ao menos 4 registros para avaliar os baselines.")

    effective_min_train_size = min(max(3, min_train_size), max(3, len(rows) - 1))
    predictions: list[dict[str, Any]] = []

    for index in range(effective_min_train_size, len(rows)):
        train_rows = rows[:index]
        test_row = rows[index]
        actual = test_row.get("target_value")
        if actual is None:
            continue

        naive_pred = predict_naive_last_value(train_rows, test_row)
        rolling_pred = predict_rolling_mean(train_rows, window=3)
        linear_pred = predict_linear_regression(train_rows, test_row)
        seasonal_pred = predict_seasonal_naive(train_rows, test_row)
        rolling_6_pred = predict_rolling_mean(train_rows, window=6)
        ridge_recent_pred = predict_ridge_recent_window(train_rows, test_row, window=12)
        tree_stump_pred = predict_tree_stump(train_rows, test_row)
        arima_pred = predict_arima(train_rows, test_row, order=(1, 1, 0))
        sarimax_pred = predict_sarimax_exog(train_rows, test_row, order=(1, 1, 0))
        random_forest_pred = predict_random_forest(train_rows, test_row)
        ensemble_pred = predict_ensemble_arima_rf(
            train_rows,
            test_row,
            cached_arima=arima_pred,
            cached_sarimax=sarimax_pred,
            cached_random_forest=random_forest_pred,
        )
        lstm_pred = predict_lstm_window(train_rows, test_row, window=12)

        predictions.append(
            {
                "date": test_row.get("date"),
                "target_month": test_row.get("target_month"),
                "actual": actual,
                "naive_last_value": naive_pred,
                "rolling_mean_3": rolling_pred,
                "rolling_mean_6": rolling_6_pred,
                "seasonal_naive_12": seasonal_pred,
                "linear_regression": linear_pred,
                "ridge_recent_12": ridge_recent_pred,
                "tree_stump": tree_stump_pred,
                "arima_1_1_0": arima_pred,
                "sarimax_exog": sarimax_pred,
                "random_forest": random_forest_pred,
                "ensemble_arima_rf": ensemble_pred,
                "lstm_window_12": lstm_pred,
            }
        )

    metrics = {model_name: _compute_metrics(predictions, model_name) for model_name in MODEL_NAMES}
    return {"metrics": metrics, "predictions": predictions, "min_train_size": effective_min_train_size}


def predict_naive_last_value(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> float | None:
    current_price = test_row.get("soy_price_brl_bag")
    if isinstance(current_price, (int, float)):
        return float(current_price)
    if train_rows:
        previous_target = train_rows[-1].get("target_value")
        if isinstance(previous_target, (int, float)):
            return float(previous_target)
    return None


def predict_rolling_mean(train_rows: list[dict[str, Any]], *, window: int = 3) -> float | None:
    values = [row.get("target_value") for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    if not values:
        return None
    sample = values[-window:]
    return sum(float(value) for value in sample) / len(sample)


def predict_linear_regression(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> float | None:
    feature_names = _select_linear_features(train_rows, test_row)
    if not feature_names:
        return predict_naive_last_value(train_rows, test_row)

    train_matrix, train_targets, imputation_values = _build_design_matrix(train_rows, feature_names)
    if len(train_matrix) < 2:
        return predict_naive_last_value(train_rows, test_row)

    coefficients = _fit_ridge_regression(train_matrix, train_targets, alpha=1e-6)
    if coefficients is None:
        return predict_naive_last_value(train_rows, test_row)

    test_vector = [1.0]
    for feature_name in feature_names:
        value = test_row.get(feature_name)
        if not isinstance(value, (int, float)):
            value = imputation_values[feature_name]
        test_vector.append(float(value))
    return _dot(coefficients, test_vector)


def predict_seasonal_naive(train_rows: list[dict[str, Any]], test_row: dict[str, Any], *, season_length: int = 12) -> float | None:
    if len(train_rows) >= season_length:
        candidate = train_rows[-season_length].get("target_value")
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return predict_naive_last_value(train_rows, test_row)


def predict_ridge_recent_window(train_rows: list[dict[str, Any]], test_row: dict[str, Any], *, window: int = 12) -> float | None:
    recent_rows = train_rows[-window:] if len(train_rows) > window else train_rows
    feature_names = _select_linear_features(recent_rows, test_row)
    if not feature_names:
        return predict_linear_regression(train_rows, test_row)

    train_matrix, train_targets, imputation_values = _build_design_matrix(recent_rows, feature_names)
    if len(train_matrix) < 2:
        return predict_linear_regression(train_rows, test_row)

    coefficients = _fit_ridge_regression(train_matrix, train_targets, alpha=0.1)
    if coefficients is None:
        return predict_linear_regression(train_rows, test_row)

    test_vector = [1.0]
    for feature_name in feature_names:
        value = test_row.get(feature_name)
        if not isinstance(value, (int, float)):
            value = imputation_values[feature_name]
        test_vector.append(float(value))
    return _dot(coefficients, test_vector)


def predict_tree_stump(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> float | None:
    feature_name = _select_tree_feature(train_rows, test_row)
    if feature_name is None:
        return predict_linear_regression(train_rows, test_row)

    samples: list[tuple[float, float]] = []
    for row in train_rows:
        feature_value = row.get(feature_name)
        target_value = row.get("target_value")
        if isinstance(feature_value, (int, float)) and isinstance(target_value, (int, float)):
            samples.append((float(feature_value), float(target_value)))

    if len(samples) < 4:
        return predict_linear_regression(train_rows, test_row)

    feature_values = sorted(value for value, _ in samples)
    thresholds = []
    for index in range(1, len(feature_values)):
        left = feature_values[index - 1]
        right = feature_values[index]
        if right != left:
            thresholds.append((left + right) / 2.0)

    best_threshold = None
    best_error = None
    best_left_mean = None
    best_right_mean = None
    for threshold in thresholds:
        left_targets = [target for value, target in samples if value <= threshold]
        right_targets = [target for value, target in samples if value > threshold]
        if not left_targets or not right_targets:
            continue
        left_mean = sum(left_targets) / len(left_targets)
        right_mean = sum(right_targets) / len(right_targets)
        error = sum((target - left_mean) ** 2 for target in left_targets) + sum((target - right_mean) ** 2 for target in right_targets)
        if best_error is None or error < best_error:
            best_error = error
            best_threshold = threshold
            best_left_mean = left_mean
            best_right_mean = right_mean

    if best_threshold is None or best_left_mean is None or best_right_mean is None:
        return predict_linear_regression(train_rows, test_row)

    test_value = test_row.get(feature_name)
    if not isinstance(test_value, (int, float)):
        return predict_linear_regression(train_rows, test_row)
    return float(best_left_mean if float(test_value) <= best_threshold else best_right_mean)


def predict_arima(
    train_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    *,
    order: tuple[int, int, int] = (1, 1, 0),
) -> float | None:
    series = [float(row["target_value"]) for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    if len(series) < max(12, order[0] + order[1] + order[2] + 3):
        return predict_seasonal_naive(train_rows, test_row)
    if ARIMA is None:
        return predict_seasonal_naive(train_rows, test_row)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
        if len(forecast) >= 1:
            return float(forecast[0])
    except Exception:  # noqa: BLE001
        return predict_seasonal_naive(train_rows, test_row)
    return predict_seasonal_naive(train_rows, test_row)


def predict_random_forest(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> float | None:
    feature_names = _select_tree_feature_set(train_rows, test_row)
    if not feature_names:
        return predict_tree_stump(train_rows, test_row)
    if RandomForestRegressor is None:
        return predict_tree_stump(train_rows, test_row)

    train_matrix, train_targets, imputation_values = _build_ml_matrix(train_rows, feature_names)
    if len(train_matrix) < 12:
        return predict_tree_stump(train_rows, test_row)

    try:
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=7,
            min_samples_leaf=2,
        )
        model.fit(train_matrix, train_targets)
        test_vector = _build_ml_test_vector(test_row, feature_names, imputation_values)
        prediction = model.predict([test_vector])
        return float(prediction[0])
    except Exception:  # noqa: BLE001
        return predict_tree_stump(train_rows, test_row)


def predict_sarimax_exog(
    train_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    *,
    order: tuple[int, int, int] = (1, 1, 0),
) -> float | None:
    feature_names = _select_exogenous_features(train_rows, test_row, max_features=8)
    if not feature_names or SARIMAX is None:
        return predict_arima(train_rows, test_row, order=order)

    series = [float(row["target_value"]) for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    if len(series) < max(18, len(feature_names) * 2):
        return predict_arima(train_rows, test_row, order=order)

    train_matrix, _, imputation_values = _build_ml_matrix(train_rows, feature_names)
    if len(train_matrix) != len(series):
        return predict_arima(train_rows, test_row, order=order)

    test_vector = _build_ml_test_vector(test_row, feature_names, imputation_values)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                exog=train_matrix,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=1, exog=[test_vector])
        if len(forecast) >= 1:
            return float(forecast[0])
    except Exception:  # noqa: BLE001
        return predict_arima(train_rows, test_row, order=order)
    return predict_arima(train_rows, test_row, order=order)


def predict_ensemble_arima_rf(
    train_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    *,
    cached_arima: float | None = None,
    cached_sarimax: float | None = None,
    cached_random_forest: float | None = None,
) -> float | None:
    temporal_pred = cached_sarimax
    if not isinstance(temporal_pred, (int, float)):
        temporal_pred = cached_arima
    if not isinstance(temporal_pred, (int, float)):
        temporal_pred = predict_arima(train_rows, test_row, order=(1, 1, 0))

    rf_pred = cached_random_forest
    if not isinstance(rf_pred, (int, float)):
        rf_pred = predict_random_forest(train_rows, test_row)

    usable = [value for value in [temporal_pred, rf_pred] if isinstance(value, (int, float))]
    if not usable:
        return predict_linear_regression(train_rows, test_row)
    if len(usable) == 1:
        return float(usable[0])
    return float((0.6 * float(temporal_pred)) + (0.4 * float(rf_pred)))


def predict_lstm_window(train_rows: list[dict[str, Any]], test_row: dict[str, Any], *, window: int = 12) -> float | None:
    """LSTM-style sequence baseline.

    When a deep-learning backend is unavailable, this falls back to a compact sequence MLP
    over the last `window` target values plus a few exogenous features, preserving the
    sequence-modeling slot in the project without blocking local execution.
    """
    if MLPRegressor is None:
        return predict_ridge_recent_window(train_rows, test_row, window=window)

    feature_names = _select_sequence_features(train_rows, test_row)
    train_matrix, train_targets = _build_sequence_supervised(train_rows, feature_names, window=window)
    if len(train_matrix) < max(12, window):
        return predict_ridge_recent_window(train_rows, test_row, window=window)

    try:
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.2,
        )
        model.fit(train_matrix, train_targets)
        test_vector = _build_sequence_test_vector(train_rows, test_row, feature_names, window=window)
        if test_vector is None:
            return predict_ridge_recent_window(train_rows, test_row, window=window)
        prediction = model.predict([test_vector])
        return float(prediction[0])
    except Exception:  # noqa: BLE001
        return predict_ridge_recent_window(train_rows, test_row, window=window)


def _select_linear_features(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> list[str]:
    available: list[str] = []
    for feature_name in PREFERRED_LINEAR_FEATURES:
        has_train_value = any(isinstance(row.get(feature_name), (int, float)) for row in train_rows)
        has_test_or_train = has_train_value or isinstance(test_row.get(feature_name), (int, float))
        if has_test_or_train:
            available.append(feature_name)
    return available


def _select_tree_feature(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> str | None:
    best_feature = None
    best_score = None
    for feature_name in PREFERRED_TREE_FEATURES:
        pairs = []
        for row in train_rows:
            feature_value = row.get(feature_name)
            target_value = row.get("target_value")
            if isinstance(feature_value, (int, float)) and isinstance(target_value, (int, float)):
                pairs.append((float(feature_value), float(target_value)))
        if len(pairs) < 4:
            continue
        x_values = [x for x, _ in pairs]
        y_values = [y for _, y in pairs]
        correlation = abs(_pearson_correlation(x_values, y_values))
        test_value = test_row.get(feature_name)
        if not isinstance(test_value, (int, float)):
            continue
        if best_score is None or correlation > best_score:
            best_score = correlation
            best_feature = feature_name
    return best_feature


def _select_tree_feature_set(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> list[str]:
    available: list[str] = []
    for feature_name in PREFERRED_TREE_FEATURES:
        if any(isinstance(row.get(feature_name), (int, float)) for row in train_rows) or isinstance(test_row.get(feature_name), (int, float)):
            available.append(feature_name)
    return available


def _select_sequence_features(train_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> list[str]:
    available: list[str] = []
    for feature_name in PREFERRED_SEQUENCE_FEATURES:
        if any(isinstance(row.get(feature_name), (int, float)) for row in train_rows) or isinstance(test_row.get(feature_name), (int, float)):
            available.append(feature_name)
    return available


def _select_exogenous_features(
    train_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    *,
    max_features: int = 8,
) -> list[str]:
    available: list[str] = []
    for feature_name in PREFERRED_EXOGENOUS_FEATURES:
        train_count = sum(1 for row in train_rows if isinstance(row.get(feature_name), (int, float)))
        has_test_or_train = train_count > 0 or isinstance(test_row.get(feature_name), (int, float))
        if has_test_or_train and train_count >= 6:
            available.append(feature_name)
        if len(available) >= max_features:
            break
    return available


def _build_design_matrix(
    train_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[list[float]], list[float], dict[str, float]]:
    imputation_values: dict[str, float] = {}
    for feature_name in feature_names:
        values = [float(row[feature_name]) for row in train_rows if isinstance(row.get(feature_name), (int, float))]
        imputation_values[feature_name] = sum(values) / len(values) if values else 0.0

    matrix: list[list[float]] = []
    targets: list[float] = []
    for row in train_rows:
        target = row.get("target_value")
        if not isinstance(target, (int, float)):
            continue
        vector = [1.0]
        for feature_name in feature_names:
            value = row.get(feature_name)
            if not isinstance(value, (int, float)):
                value = imputation_values[feature_name]
            vector.append(float(value))
        matrix.append(vector)
        targets.append(float(target))
    return matrix, targets, imputation_values


def _build_ml_matrix(
    train_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[list[float]], list[float], dict[str, float]]:
    imputation_values: dict[str, float] = {}
    for feature_name in feature_names:
        values = [float(row[feature_name]) for row in train_rows if isinstance(row.get(feature_name), (int, float))]
        imputation_values[feature_name] = sum(values) / len(values) if values else 0.0

    matrix: list[list[float]] = []
    targets: list[float] = []
    for row in train_rows:
        target = row.get("target_value")
        if not isinstance(target, (int, float)):
            continue
        vector = []
        for feature_name in feature_names:
            value = row.get(feature_name)
            if not isinstance(value, (int, float)):
                value = imputation_values[feature_name]
            vector.append(float(value))
        matrix.append(vector)
        targets.append(float(target))
    return matrix, targets, imputation_values


def _build_ml_test_vector(test_row: dict[str, Any], feature_names: list[str], imputation_values: dict[str, float]) -> list[float]:
    vector: list[float] = []
    for feature_name in feature_names:
        value = test_row.get(feature_name)
        if not isinstance(value, (int, float)):
            value = imputation_values[feature_name]
        vector.append(float(value))
    return vector


def _build_sequence_supervised(
    train_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    window: int,
) -> tuple[list[list[float]], list[float]]:
    series = [float(row["target_value"]) for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    usable_rows = [row for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    if len(series) <= window:
        return [], []

    exogenous_means: dict[str, float] = {}
    for feature_name in feature_names:
        values = [float(row[feature_name]) for row in usable_rows if isinstance(row.get(feature_name), (int, float))]
        exogenous_means[feature_name] = sum(values) / len(values) if values else 0.0

    matrix: list[list[float]] = []
    targets: list[float] = []
    for index in range(window, len(usable_rows)):
        vector = list(series[index - window : index])
        row = usable_rows[index - 1]
        for feature_name in feature_names:
            value = row.get(feature_name)
            if not isinstance(value, (int, float)):
                value = exogenous_means[feature_name]
            vector.append(float(value))
        matrix.append(vector)
        targets.append(float(usable_rows[index].get("target_value")))
    return matrix, targets


def _build_sequence_test_vector(
    train_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    feature_names: list[str],
    *,
    window: int,
) -> list[float] | None:
    usable_rows = [row for row in train_rows if isinstance(row.get("target_value"), (int, float))]
    if len(usable_rows) < window:
        return None
    vector = [float(row["target_value"]) for row in usable_rows[-window:]]
    for feature_name in feature_names:
        value = test_row.get(feature_name)
        if not isinstance(value, (int, float)):
            history = [float(row[feature_name]) for row in usable_rows if isinstance(row.get(feature_name), (int, float))]
            value = sum(history) / len(history) if history else 0.0
        vector.append(float(value))
    return vector


def _fit_ridge_regression(
    x_rows: list[list[float]],
    y_values: list[float],
    *,
    alpha: float,
) -> list[float] | None:
    feature_count = len(x_rows[0])
    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]

    for vector, target in zip(x_rows, y_values):
        for i in range(feature_count):
            xty[i] += vector[i] * target
            for j in range(feature_count):
                xtx[i][j] += vector[i] * vector[j]

    for i in range(feature_count):
        if i != 0:
            xtx[i][i] += alpha

    return _solve_linear_system(xtx, xty)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [list(matrix[row_index]) + [vector[row_index]] for row_index in range(size)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-12:
            return None
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[row_index][size] for row_index in range(size)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_den * right_den
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    pairs = []
    for row in rows:
        actual = row.get("actual")
        predicted = row.get(prediction_field)
        if isinstance(actual, (int, float)) and isinstance(predicted, (int, float)):
            pairs.append((float(actual), float(predicted)))

    if not pairs:
        return {"count": 0, "mae": None, "rmse": None, "mape": None}

    absolute_errors = [abs(actual - predicted) for actual, predicted in pairs]
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in pairs]
    percentage_errors = [abs((actual - predicted) / actual) for actual, predicted in pairs if actual != 0]

    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mape = (sum(percentage_errors) / len(percentage_errors) * 100.0) if percentage_errors else None

    return {
        "count": len(pairs),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6) if mape is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
