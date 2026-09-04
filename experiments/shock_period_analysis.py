#!/usr/bin/env python3
"""
Análise de robustez em períodos de choque — todos os modelos.

Executa walk-forward 1-passo para naive_last_value, ARIMA(1,1,0),
Random Forest e XGBoost, comparando as métricas por período:
  - pré-choque : antes de 2020-01
  - choque     : 2020-01 a 2022-12 (COVID + crise de oferta + guerra na Ucrânia)
  - pós-choque : 2023-01 em diante

Base de dados: run_cepea_hybrid_2015plus_manual_remote_merge (2015-04 → 2026-03).
Todos os modelos são avaliados nos mesmos períodos para comparação justa.

Saídas em results/shock_period/:
  - shock_period_walkforward.csv   : previsões step-by-step de todos os modelos
  - shock_period_metrics.json/.csv : métricas por período × modelo
  - shock_period_errors.png        : erro absoluto ao longo do tempo por modelo
  - shock_period_forecast.png      : real vs. previsto por modelo

Uso:
  PYTHONPATH=src .venv/bin/python experiments/shock_period_analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless: salva PNGs sem display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
OUT_DIR = PROJECT_ROOT / "results" / "shock_period"

TARGET_VARIABLE = "soy_price_brl_bag_next_month"
MIN_TRAIN = 24

SHOCK_START = pd.Timestamp("2020-01-01")
SHOCK_END = pd.Timestamp("2022-12-01")

META_COLS = {
    "date", "target_month", "target_variable", "target_value",
    "target_source", "target_series_name",
    "soy_price_brl_bag", "soy_price_usd_bag",
}

PERIOD_COLORS = {
    "pre_shock":  "#16a34a",
    "shock":      "#dc2626",
    "pos_shock":  "#2563eb",
}
PERIOD_LABELS = {
    "pre_shock":  "Pré-choque (< 2020)",
    "shock":      "Choque (2020–2022)",
    "pos_shock":  "Pós-choque (≥ 2023)",
}

MODEL_STYLES = {
    "naive_last_value": dict(color="#6b7280", linestyle=":",  linewidth=1.4),
    "arima_1_1_0":      dict(color="#f59e0b", linestyle="--", linewidth=1.4),
    "random_forest":    dict(color="#2563eb", linestyle="-.", linewidth=1.4),
    "xgboost":          dict(color="#7c3aed", linestyle="--", linewidth=1.4),
}
MODEL_LABELS = {
    "naive_last_value": "Persistência",
    "arima_1_1_0":      "ARIMA(1,1,0)",
    "random_forest":    "Random Forest",
    "xgboost":          "XGBoost",
}


def classify_period(date: pd.Timestamp) -> str:
    if date < SHOCK_START:
        return "pre_shock"
    if date <= SHOCK_END:
        return "shock"
    return "pos_shock"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in META_COLS and c != "target_value"]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all()]


def build_ml_pipeline(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def walkforward_naive(series: pd.Series, min_train: int) -> dict[pd.Timestamp, float]:
    preds = {}
    for i in range(min_train, len(series)):
        preds[series.index[i]] = float(series.iloc[i - 1])
    return preds


def walkforward_arima(series: pd.Series, min_train: int) -> dict[pd.Timestamp, float]:
    preds = {}
    for i in range(min_train, len(series)):
        train = series.iloc[:i].values
        try:
            pred = float(ARIMA(train, order=(1, 1, 0)).fit().forecast(steps=1)[0])
        except Exception:
            pred = float("nan")
        preds[series.index[i]] = pred
    return preds


def walkforward_ml(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train: int,
    model_factory,
) -> dict[pd.Timestamp, float]:
    preds = {}
    y = df["target_value"].values.astype(float)
    X = df[feature_cols].values

    for i in range(min_train, len(df)):
        X_train, y_train = X[:i], y[:i]
        X_test = X[i : i + 1]
        try:
            pipeline = build_ml_pipeline(model_factory())
            pipeline.fit(X_train, y_train)
            pred = float(pipeline.predict(X_test)[0])
        except Exception:
            pred = float("nan")
        preds[df["date"].iloc[i]] = pred

    return preds


def compute_metrics(
    actuals: pd.Series,
    preds: dict[str, dict[pd.Timestamp, float]],
) -> list[dict]:
    rows = []
    for model_name, pred_dict in preds.items():
        common_dates = sorted(set(actuals.index) & set(pred_dict.keys()))
        if not common_dates:
            continue
        y_true = np.array([actuals[d] for d in common_dates])
        y_pred = np.array([pred_dict[d] for d in common_dates])
        periods = [classify_period(d) for d in common_dates]

        for period in ["pre_shock", "shock", "pos_shock", "all"]:
            mask = np.array([True] * len(periods)) if period == "all" else np.array([p == period for p in periods])
            n = mask.sum()
            if n == 0:
                continue
            yt, yp = y_true[mask], y_pred[mask]
            valid = ~np.isnan(yp)
            if valid.sum() == 0:
                continue
            yt, yp = yt[valid], yp[valid]
            rows.append({
                "model": model_name,
                "period": period,
                "label_period": PERIOD_LABELS.get(period, "Total"),
                "label_model": MODEL_LABELS.get(model_name, model_name),
                "n_obs": int(valid.sum()),
                "date_start": str(min(d for d, p in zip(common_dates, periods) if period == "all" or p == period)),
                "date_end": str(max(d for d, p in zip(common_dates, periods) if period == "all" or p == period)),
                "mae": round(float(mean_absolute_error(yt, yp)), 6),
                "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 6),
                "mape": round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 6),
            })
    return rows


def build_walkforward_df(
    dates: list[pd.Timestamp],
    actuals: dict[pd.Timestamp, float],
    preds: dict[str, dict[pd.Timestamp, float]],
) -> pd.DataFrame:
    rows = []
    for d in sorted(dates):
        row = {"date": d, "actual": actuals.get(d, float("nan")), "period": classify_period(d)}
        for model_name, pred_dict in preds.items():
            row[model_name] = pred_dict.get(d, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_errors(wf: pd.DataFrame, model_names: list[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    shock_mask = wf["period"] == "shock"
    if shock_mask.any():
        ax.axvspan(
            wf.loc[shock_mask, "date"].min(),
            wf.loc[shock_mask, "date"].max(),
            alpha=0.08, color="#dc2626",
        )

    for model in model_names:
        if model not in wf.columns:
            continue
        abs_err = (wf["actual"] - wf[model]).abs()
        style = MODEL_STYLES.get(model, {})
        ax.plot(wf["date"], abs_err, label=MODEL_LABELS.get(model, model),
                alpha=0.85, **style)

    ax.axvline(SHOCK_START, color="#dc2626", linestyle="--", linewidth=0.8)
    ax.axvline(SHOCK_END + pd.offsets.MonthEnd(1), color="#dc2626", linestyle="--", linewidth=0.8)
    ax.set_title("Erro absoluto por mês — Walk-forward 1-passo — todos os modelos")
    ax.set_xlabel("Data")
    ax.set_ylabel("Erro absoluto (R$/saca)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_dir / "shock_period_errors.png", dpi=150)
    plt.close(fig)
    print("  Salvo: shock_period_errors.png")


def plot_forecast(wf: pd.DataFrame, model_names: list[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    shock_mask = wf["period"] == "shock"
    if shock_mask.any():
        ax.axvspan(
            wf.loc[shock_mask, "date"].min(),
            wf.loc[shock_mask, "date"].max(),
            alpha=0.08, color="#dc2626", label="Período de choque",
        )

    ax.plot(wf["date"], wf["actual"], color="black", linewidth=2, label="Realizado", zorder=5)
    for model in model_names:
        if model not in wf.columns:
            continue
        style = MODEL_STYLES.get(model, {})
        ax.plot(wf["date"], wf[model], label=MODEL_LABELS.get(model, model),
                alpha=0.8, **style)

    ax.set_title("Real vs. Previsto — Walk-forward 1-passo — Soja CEPEA/ESALQ")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (R$/saca)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_dir / "shock_period_forecast.png", dpi=150)
    plt.close(fig)
    print("  Salvo: shock_period_forecast.png")


def plot_mae_bars(metrics: list[dict], out_dir: Path) -> None:
    df = pd.DataFrame(metrics)
    df = df[df["period"] != "all"]
    period_order = ["pre_shock", "shock", "pos_shock"]
    model_order = ["naive_last_value", "arima_1_1_0", "random_forest", "xgboost"]
    model_order = [m for m in model_order if m in df["model"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(model_order)
    n_periods = len(period_order)
    width = 0.7 / n_models
    x = np.arange(n_periods)

    for i, model in enumerate(model_order):
        maes = []
        for period in period_order:
            row = df[(df["model"] == model) & (df["period"] == period)]
            maes.append(float(row["mae"].iloc[0]) if len(row) else float("nan"))
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width=width * 0.9,
                      label=MODEL_LABELS.get(model, model),
                      color=MODEL_STYLES.get(model, {}).get("color", "#6b7280"),
                      alpha=0.85)
        for bar, val in zip(bars, maes):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in period_order])
    ax.set_ylabel("MAE (R$/saca)")
    ax.set_title("MAE por período e modelo — Walk-forward 1-passo")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_dir / "shock_period_mae_bars.png", dpi=150)
    plt.close(fig)
    print("  Salvo: shock_period_mae_bars.png")


def print_table(metrics: list[dict], model_names: list[str]) -> None:
    period_order = ["pre_shock", "shock", "pos_shock", "all"]
    df = pd.DataFrame(metrics)

    print(f"\n{'Modelo':<20} {'Período':<25} {'N':>5} {'MAE':>8} {'RMSE':>8} {'MAPE%':>7}")
    print("-" * 78)
    for model in model_names:
        mdf = df[df["model"] == model]
        for period in period_order:
            row = mdf[mdf["period"] == period]
            if row.empty:
                continue
            r = row.iloc[0]
            label = PERIOD_LABELS.get(period, "Total")
            m_label = MODEL_LABELS.get(model, model)
            print(f"{m_label:<20} {label:<25} {r['n_obs']:>5} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['mape']:>7.2f}")
        print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando: {DATA_FILE.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = (
        df[df["target_variable"] == TARGET_VARIABLE]
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"Observações: {len(df)} | {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"Período de choque: {SHOCK_START.date()} → {SHOCK_END.date()}")
    print(f"Treino mínimo: {MIN_TRAIN} meses → previsões a partir de {df['date'].iloc[MIN_TRAIN].date()}")

    feature_cols = get_feature_cols(df)
    print(f"Features para ML: {len(feature_cols)} colunas")

    target_series = pd.Series(
        df["target_value"].values.astype(float),
        index=df["date"],
    )
    actuals = dict(zip(df["date"], df["target_value"].astype(float)))

    all_preds: dict[str, dict[pd.Timestamp, float]] = {}

    print("\nWalk-forward — naive_last_value...")
    all_preds["naive_last_value"] = walkforward_naive(target_series, MIN_TRAIN)

    print("Walk-forward — ARIMA(1,1,0)... (pode levar alguns instantes)")
    all_preds["arima_1_1_0"] = walkforward_arima(target_series, MIN_TRAIN)

    print("Walk-forward — Random Forest...")
    all_preds["random_forest"] = walkforward_ml(
        df, feature_cols, MIN_TRAIN,
        lambda: RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
    )

    if HAS_XGBOOST:
        print("Walk-forward — XGBoost...")
        all_preds["xgboost"] = walkforward_ml(
            df, feature_cols, MIN_TRAIN,
            lambda: XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42,
            ),
        )
    else:
        print("[aviso] xgboost não disponível, pulando.")

    model_names = list(all_preds.keys())
    all_dates = sorted(set().union(*[set(p.keys()) for p in all_preds.values()]))

    wf = build_walkforward_df(all_dates, actuals, all_preds)
    out_csv = OUT_DIR / "shock_period_walkforward.csv"
    wf.to_csv(out_csv, index=False)
    print(f"\n  Salvo: shock_period_walkforward.csv ({len(wf)} linhas)")

    metrics = compute_metrics(target_series, all_preds)
    print_table(metrics, model_names)

    out_json = OUT_DIR / "shock_period_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Salvo: shock_period_metrics.json")

    metrics_df = pd.DataFrame(metrics)
    out_mcsv = OUT_DIR / "shock_period_metrics.csv"
    metrics_df.to_csv(out_mcsv, index=False)
    print(f"  Salvo: shock_period_metrics.csv")

    print("\nGerando plots...")
    plot_errors(wf, model_names, OUT_DIR)
    plot_forecast(wf, model_names, OUT_DIR)
    plot_mae_bars(metrics, OUT_DIR)

    # Destaque comparativo
    df_m = pd.DataFrame(metrics)
    for model in model_names:
        pre = df_m[(df_m["model"] == model) & (df_m["period"] == "pre_shock")]
        shock = df_m[(df_m["model"] == model) & (df_m["period"] == "shock")]
        if not pre.empty and not shock.empty:
            ratio = shock.iloc[0]["mae"] / pre.iloc[0]["mae"]
            label = MODEL_LABELS.get(model, model)
            print(f"  {label}: MAE choque é {ratio:.2f}× o MAE pré-choque")

    print(f"\nAnálise completa. Arquivos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido pelo usuário.")
