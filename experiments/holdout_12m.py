#!/usr/bin/env python3
"""
Holdout final de 12 meses — ARIMA(1,1,0), Random Forest, XGBoost.

Alvo unico do TGSI II: ``soy_price_brl_bag_next_month`` (preco da soja
CEPEA/ESALQ - Paranagua, R$/saca, um mes a frente).

  dados do mes t  ->  preco da soja em BRL no mes t+1

Os ultimos 12 meses da serie sao reservados como teste final; o restante
e usado para treino. Nenhuma observacao futura entra no treino.

Historico: a versao do TGSI I tambem rodava uma linha de modelagem em USD
(``soy_price_usd_bag_next_month``) e uma conversao USD->BRL usando o cambio
realizado do holdout. Isso foi REMOVIDO — ver docs/methodology.md secao
"Alvo em BRL e a questao do USD". A coluna ``soy_price_usd_bag`` continua no
dataset como variavel explicativa / rastreabilidade, mas nao ha mais um
experimento oficial em USD.

Saidas em results/holdout_12m/:
  - holdout_12m_predictions.csv   : realizado vs. previsto por modelo
  - holdout_12m_metrics.json      : MAE / RMSE / MAPE / R2 por modelo
  - holdout_12m_forecast.png      : grafico realizado vs. previsto
  - holdout_12m_metrics.png       : barras de metricas por modelo

Uso:
  PYTHONPATH=src python experiments/holdout_12m.py
  PYTHONPATH=src python experiments/holdout_12m.py --input caminho/para/features_monthly_modeling.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # sem display; salva PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
OUT_DIR = PROJECT_ROOT / "results" / "holdout_12m"

HOLDOUT_MONTHS = 12
TARGET_VARIABLE = "soy_price_brl_bag_next_month"
META_COLUMNS = {
    "date",
    "target_month",
    "target_variable",
    "target_value",
    "target_source",
    "target_series_name",
}


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_tabular_pipeline(model) -> Pipeline:
    # imputer fica DENTRO do pipeline: as estatisticas de imputacao sao
    # aprendidas so no fit (treino), nunca no teste.
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def prepare_target_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_df = (
        df.loc[df["target_variable"] == TARGET_VARIABLE]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    if len(target_df) <= HOLDOUT_MONTHS:
        raise ValueError(
            f"Alvo {TARGET_VARIABLE} sem observacoes suficientes para holdout de {HOLDOUT_MONTHS} meses."
        )

    feature_candidates = [
        col for col in target_df.columns if col not in META_COLUMNS and col != "target_value"
    ]
    numeric_features = [
        col for col in feature_candidates if pd.api.types.is_numeric_dtype(target_df[col])
    ]
    numeric_features = [col for col in numeric_features if not target_df[col].isna().all()]
    if not numeric_features:
        raise ValueError("Nenhuma feature numerica utilizavel.")
    return target_df, numeric_features


def train_test_split_time(target_df: pd.DataFrame, features: list[str]):
    train_df = target_df.iloc[:-HOLDOUT_MONTHS].copy()
    test_df = target_df.iloc[-HOLDOUT_MONTHS:].copy()
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df["target_value"].astype(float).to_numpy()
    y_test = test_df["target_value"].astype(float).to_numpy()
    return train_df, test_df, X_train, X_test, y_train, y_test


def run_models(target_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    metrics.append(evaluate_model("arima_1_1_0", y_test, arima_pred))

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
    metrics.append(evaluate_model("random_forest", y_test, rf_pred))

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
        metrics.append(evaluate_model("xgboost", y_test, xgb_pred))
    else:
        print("[aviso] xgboost nao disponivel; seguindo sem esse modelo.")

    metrics_df = pd.DataFrame(metrics).sort_values("mae").reset_index(drop=True)
    return predictions, metrics_df


def plot_forecast(predictions: pd.DataFrame, out_dir: Path) -> None:
    plot_df = predictions.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(plot_df["date"], plot_df["actual"], marker="o", linewidth=3, color="black", label="Realizado")
    for col in [c for c in plot_df.columns if c not in {"date", "target_month", "actual"}]:
        ax.plot(plot_df["date"], plot_df[col], marker="o", linewidth=2, label=col)
    ax.set_title("Holdout final de 12 meses — realizado vs. previsto (alvo BRL, t+1)")
    ax.set_xlabel("Mes previsto")
    ax.set_ylabel("Preco da soja (R$/saca)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_dir / "holdout_12m_forecast.png", dpi=150)
    plt.close(fig)


def plot_metrics(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df.set_index("model")[["mae", "rmse", "mape"]].plot(kind="bar", ax=ax)
    ax.set_title("Metricas por modelo — holdout de 12 meses")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "holdout_12m_metrics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout final de 12 meses — ARIMA, RF, XGBoost (alvo BRL)")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV de modelagem (padrao: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Arquivo nao encontrado: {args.input}\n"
            "Gere a base primeiro:  PYTHONPATH=src python scripts/build_dataset.py"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Arquivo de entrada: {args.input}")
    df = pd.read_csv(args.input)

    required = {"date", "target_month", "target_variable", "target_value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"O CSV nao possui as colunas obrigatorias: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["target_month"] = pd.to_datetime(df["target_month"])

    target_df, features = prepare_target_frame(df)
    print(f"Holdout: {HOLDOUT_MONTHS} meses | observacoes: {len(target_df)} | features: {len(features)}")
    print(f"Treino : {target_df['date'].iloc[0].date()} -> {target_df['date'].iloc[-HOLDOUT_MONTHS - 1].date()}")
    print(f"Teste  : {target_df['date'].iloc[-HOLDOUT_MONTHS].date()} -> {target_df['date'].iloc[-1].date()}")

    predictions, metrics_df = run_models(target_df, features)

    print("\n=== Metricas (holdout 12m, alvo BRL) ===")
    print(metrics_df.to_string(index=False))

    predictions.to_csv(OUT_DIR / "holdout_12m_predictions.csv", index=False)
    (OUT_DIR / "holdout_12m_metrics.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_forecast(predictions, OUT_DIR)
    plot_metrics(metrics_df, OUT_DIR)
    print(f"\nArquivos salvos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nExecucao interrompida pelo usuario.")
