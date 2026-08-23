#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # noqa: BLE001
    tk = None
    filedialog = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False


HOLDOUT_MONTHS = 12
BRL_TARGET = "soy_price_brl_bag_next_month"
USD_TARGET = "soy_price_usd_bag_next_month"
META_COLUMNS = {
    "date",
    "target_month",
    "target_variable",
    "target_value",
    "target_source",
    "target_series_name",
}


def select_csv_file() -> Path:
    if filedialog is not None and tk is not None:
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo features_monthly_modeling.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if file_path:
            return Path(file_path)

    raw = input("Caminho do CSV modelado: ").strip()
    if not raw:
        raise SystemExit("Nenhum arquivo foi selecionado.")
    return Path(raw)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, scope: str) -> dict:
    return {
        "scope": scope,
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def build_tabular_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def prepare_target_frame(df: pd.DataFrame, target_variable: str) -> tuple[pd.DataFrame, list[str]]:
    target_df = (
        df.loc[df["target_variable"] == target_variable]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    if len(target_df) <= HOLDOUT_MONTHS:
        raise ValueError(
            f"Target {target_variable} não possui observações suficientes para holdout de {HOLDOUT_MONTHS} meses."
        )

    feature_candidates = [
        col
        for col in target_df.columns
        if col not in META_COLUMNS and col != "target_value"
    ]
    numeric_features = [
        col for col in feature_candidates if pd.api.types.is_numeric_dtype(target_df[col])
    ]
    numeric_features = [col for col in numeric_features if not target_df[col].isna().all()]

    if not numeric_features:
        raise ValueError(f"Nenhuma feature numérica utilizável para {target_variable}.")

    return target_df, numeric_features


def train_test_split_time(target_df: pd.DataFrame, features: list[str]):
    train_df = target_df.iloc[:-HOLDOUT_MONTHS].copy()
    test_df = target_df.iloc[-HOLDOUT_MONTHS:].copy()

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df["target_value"].astype(float).to_numpy()
    y_test = test_df["target_value"].astype(float).to_numpy()

    return train_df, test_df, X_train, X_test, y_train, y_test


def run_direct_models(
    target_df: pd.DataFrame,
    features: list[str],
    scope_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df, X_train, X_test, y_train, y_test = train_test_split_time(target_df, features)

    predictions = pd.DataFrame(
        {
            "date": test_df["date"].to_numpy(),
            "target_month": test_df["target_month"].to_numpy(),
            "actual": y_test,
        }
    )
    metrics: list[dict] = []

    arima_fit = ARIMA(y_train, order=(1, 1, 0)).fit()
    arima_pred = np.asarray(arima_fit.forecast(steps=HOLDOUT_MONTHS), dtype=float)
    predictions["arima_1_1_0"] = arima_pred
    metrics.append(evaluate_model("arima_1_1_0", y_test, arima_pred, scope_name))

    rf_model = build_tabular_pipeline(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )
    rf_model.fit(X_train, y_train)
    rf_pred = np.asarray(rf_model.predict(X_test), dtype=float)
    predictions["random_forest"] = rf_pred
    metrics.append(evaluate_model("random_forest", y_test, rf_pred, scope_name))

    if HAS_XGBOOST:
        xgb_model = build_tabular_pipeline(
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_weight=1,
                objective="reg:squarederror",
                random_state=42,
            )
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = np.asarray(xgb_model.predict(X_test), dtype=float)
        predictions["xgboost"] = xgb_pred
        metrics.append(evaluate_model("xgboost", y_test, xgb_pred, scope_name))
    else:
        print("[aviso] xgboost não está disponível; seguindo sem esse modelo.")

    metrics_df = pd.DataFrame(metrics).sort_values(["scope", "mae"]).reset_index(drop=True)
    return predictions, metrics_df


def build_usd_to_brl_view(
    brl_df: pd.DataFrame,
    usd_df: pd.DataFrame,
    usd_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    brl_train, brl_test, _, _, _, y_brl_test = train_test_split_time(
        brl_df,
        [col for col in brl_df.columns if col not in META_COLUMNS and col != "target_value" and pd.api.types.is_numeric_dtype(brl_df[col])],
    )
    usd_train, usd_test, _, _, _, _ = train_test_split_time(
        usd_df,
        [col for col in usd_df.columns if col not in META_COLUMNS and col != "target_value" and pd.api.types.is_numeric_dtype(usd_df[col])],
    )

    merge_cols = ["date", "target_month"]
    fx_view = (
        brl_test[["date", "target_month", "target_value", "usd_brl"]]
        .rename(columns={"target_value": "actual_brl", "usd_brl": "actual_fx"})
        .merge(
            usd_test[["date", "target_month", "target_value"]].rename(columns={"target_value": "actual_usd"}),
            on=merge_cols,
            how="inner",
        )
        .merge(
            usd_predictions.rename(columns={"actual": "actual_usd_from_pred_table"}),
            on=merge_cols,
            how="inner",
        )
    )

    fx_train_series = brl_train["usd_brl"].astype(float).to_numpy()
    fx_arima = ARIMA(fx_train_series, order=(1, 1, 0)).fit().forecast(steps=HOLDOUT_MONTHS)
    fx_view["predicted_fx_arima"] = np.asarray(fx_arima, dtype=float)
    fx_view["predicted_fx_flat"] = float(brl_train["usd_brl"].dropna().iloc[-1])

    usd_model_columns = [c for c in ["arima_1_1_0", "random_forest", "xgboost"] if c in fx_view.columns]
    output = fx_view[["date", "target_month", "actual_brl", "actual_fx", "actual_usd"]].copy()

    metrics: list[dict] = []
    for model_col in usd_model_columns:
        actual_fx_name = f"{model_col}_usd_to_brl_actual_fx"
        flat_fx_name = f"{model_col}_usd_to_brl_flat_fx"
        arima_fx_name = f"{model_col}_usd_to_brl_arima_fx"

        output[actual_fx_name] = fx_view[model_col] * fx_view["actual_fx"]
        output[flat_fx_name] = fx_view[model_col] * fx_view["predicted_fx_flat"]
        output[arima_fx_name] = fx_view[model_col] * fx_view["predicted_fx_arima"]

        metrics.append(evaluate_model(actual_fx_name, y_brl_test, output[actual_fx_name].to_numpy(), "usd_to_brl"))
        metrics.append(evaluate_model(flat_fx_name, y_brl_test, output[flat_fx_name].to_numpy(), "usd_to_brl"))
        metrics.append(evaluate_model(arima_fx_name, y_brl_test, output[arima_fx_name].to_numpy(), "usd_to_brl"))

    metrics_df = pd.DataFrame(metrics).sort_values(["scope", "mae"]).reset_index(drop=True)
    return output, metrics_df


def plot_direct_results(predictions: pd.DataFrame, metrics_df: pd.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    plot_df = predictions.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    axes[0].plot(plot_df["date"], plot_df["actual"], marker="o", linewidth=3, color="black", label="actual")

    for col in [c for c in plot_df.columns if c not in {"date", "target_month", "actual"}]:
        axes[0].plot(plot_df["date"], plot_df[col], marker="o", linewidth=2, label=col)

    axes[0].set_title(title)
    axes[0].set_xlabel("Data")
    axes[0].set_ylabel("Preço")
    axes[0].legend()

    metrics_plot = metrics_df.set_index("model")[["mae", "rmse", "mape"]]
    metrics_plot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Métricas por modelo")
    axes[1].set_xlabel("Modelo")
    axes[1].set_ylabel("Valor")
    axes[1].tick_params(axis="x", rotation=0)

    plt.show()


def plot_usd_to_brl_results(predictions: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)

    plot_df = predictions.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    axes[0].plot(plot_df["date"], plot_df["actual_brl"], marker="o", linewidth=3, color="black", label="actual_brl")
    for col in [c for c in plot_df.columns if c not in {"date", "target_month", "actual_brl", "actual_fx", "actual_usd"}]:
        axes[0].plot(plot_df["date"], plot_df[col], marker="o", linewidth=1.8, label=col)
    axes[0].set_title("Visão USD → BRL no holdout de 12 meses")
    axes[0].set_xlabel("Data")
    axes[0].set_ylabel("Preço em R$/saca")
    axes[0].legend(ncol=2, fontsize=9)

    metrics_plot = metrics_df.set_index("model")[["mae", "rmse", "mape"]]
    metrics_plot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Métricas da conversão USD → BRL")
    axes[1].set_xlabel("Modelo")
    axes[1].set_ylabel("Valor")
    axes[1].tick_params(axis="x", rotation=35)

    plt.show()


SCENARIO_LABELS = {
    "P10": "cenário pessimista",
    "P50": "mediana",
    "P90": "cenário otimista",
}


def build_scenario_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    meta_cols = {"date", "target_month", "actual", "actual_brl", "actual_fx", "actual_usd"}
    actual_cols = [col for col in ["actual", "actual_brl", "actual_usd", "actual_fx"] if col in predictions.columns]
    rows: list[dict[str, object]] = []

    for _, row in predictions.iterrows():
        base_payload = {
            "date": row.get("date"),
            "target_month": row.get("target_month"),
            **{col: row.get(col) for col in actual_cols},
        }
        for model_col in [col for col in predictions.columns if col not in meta_cols]:
            rows.append(
                {
                    **base_payload,
                    "model": model_col,
                    "P10": np.nan,
                    "P50": row.get(model_col),
                    "P90": np.nan,
                    "P10_label": SCENARIO_LABELS["P10"],
                    "P50_label": SCENARIO_LABELS["P50"],
                    "P90_label": SCENARIO_LABELS["P90"],
                    "scenario_support": "point_only",
                    "notes": "Pipeline atual gera previsão pontual; P50 preenchido e P10/P90 reservados para intervalos/cenários futuros.",
                }
            )

    return pd.DataFrame(rows)


def build_scenario_manifest(outputs: dict[str, pd.DataFrame]) -> dict[str, object]:
    manifest_outputs: dict[str, object] = {}
    for name, df in outputs.items():
        if name.endswith("metrics"):
            continue
        model_cols = [
            col
            for col in df.columns
            if col not in {"date", "target_month", "actual", "actual_brl", "actual_fx", "actual_usd"}
        ]
        manifest_outputs[name] = {
            "models": model_cols,
            "scenario_columns": ["P10", "P50", "P90"],
            "current_behavior": "P50 preenchido; P10/P90 reservados para quando o pipeline gerar intervalos ou quantis.",
        }

    return {
        "scenario_labels": SCENARIO_LABELS,
        "outputs": manifest_outputs,
    }


def save_outputs(base_csv: Path, outputs: dict[str, pd.DataFrame]) -> None:
    out_dir = base_csv.parent / f"{base_csv.stem}_holdout_12m_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_outputs: dict[str, pd.DataFrame] = {}

    for name, df in outputs.items():
        if name.endswith("metrics"):
            path = out_dir / f"{name}.json"
            path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            scenario_outputs[f"{name}_scenarios"] = build_scenario_frame(df)

    for name, df in scenario_outputs.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)

    manifest_path = out_dir / "scenario_convention.json"
    manifest_path.write_text(
        json.dumps(build_scenario_manifest(outputs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nArquivos salvos em: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout 12 meses — ARIMA, RF, XGBoost")
    parser.add_argument("--input", type=Path, default=None, help="Caminho para features_monthly_modeling.csv")
    args = parser.parse_args()

    csv_path = args.input if args.input is not None else select_csv_file()
    if not csv_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {csv_path}")

    print(f"\nArquivo selecionado: {csv_path}")
    df = pd.read_csv(csv_path)

    required_columns = {"date", "target_month", "target_variable", "target_value", "usd_brl"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise SystemExit(f"O CSV não possui as colunas obrigatórias: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["target_month"] = pd.to_datetime(df["target_month"])

    brl_df, brl_features = prepare_target_frame(df, BRL_TARGET)
    usd_df, usd_features = prepare_target_frame(df, USD_TARGET)

    print(f"\nHoldout configurado: {HOLDOUT_MONTHS} meses")
    print(f"Observações BRL: {len(brl_df)} | features: {len(brl_features)}")
    print(f"Observações USD: {len(usd_df)} | features: {len(usd_features)}")

    brl_predictions, brl_metrics = run_direct_models(brl_df, brl_features, "brl_direct")
    usd_predictions, usd_metrics = run_direct_models(usd_df, usd_features, "usd_direct")
    usd_brl_predictions, usd_brl_metrics = build_usd_to_brl_view(brl_df, usd_df, usd_predictions)

    print("\n=== Métricas BRL direto ===")
    print(brl_metrics.to_string(index=False))

    print("\n=== Métricas USD direto ===")
    print(usd_metrics.to_string(index=False))

    print("\n=== Métricas USD -> BRL ===")
    print(usd_brl_metrics.to_string(index=False))

    plot_direct_results(brl_predictions, brl_metrics, "Holdout 12m — modelos diretos em BRL")
    plot_direct_results(usd_predictions, usd_metrics, "Holdout 12m — modelos diretos em USD")
    plot_usd_to_brl_results(usd_brl_predictions, usd_brl_metrics)

    save_outputs(
        csv_path,
        {
            "brl_direct_predictions": brl_predictions,
            "brl_direct_metrics": brl_metrics,
            "usd_direct_predictions": usd_predictions,
            "usd_direct_metrics": usd_metrics,
            "usd_to_brl_predictions": usd_brl_predictions,
            "usd_to_brl_metrics": usd_brl_metrics,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nExecução interrompida pelo usuário.")
