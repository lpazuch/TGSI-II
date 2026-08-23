#!/usr/bin/env python3
"""
Análise de sensibilidade: MAE do ARIMA(1,1,0) por janela de treino.

Testa diferentes janelas (36 a 216 meses) com holdout fixo de 24 meses.
Mostra como a estabilidade do modelo varia com o tamanho da base histórica.

Saídas em docs/academic_outputs/sensitivity_analysis/:
  - sensitivity_window_results.csv / .json
  - sensitivity_window_mae.png

Uso:
  PYTHONPATH=src python scripts/sensitivity_analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "run_cepea_hybrid_2006plus_long_remote"
    / "raw"
    / "target_soy_cepea_monthly.csv"
)
OUT_DIR = PROJECT_ROOT / "docs" / "academic_outputs" / "sensitivity_analysis"

HOLDOUT_MONTHS = 24
ARIMA_ORDER = (1, 1, 0)
WINDOW_SIZES = [36, 48, 60, 72, 84, 96, 120, 144, 180, 216]


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["soy_price_brl_bag"]).reset_index(drop=True)
    return pd.Series(df["soy_price_brl_bag"].values, index=df["date"], name="soy_price_brl_bag")


def evaluate_window(series: pd.Series, window: int, holdout: int) -> dict | None:
    test = series.iloc[-holdout:]
    train_pool = series.iloc[:-holdout]

    if len(train_pool) < window:
        return None

    train = train_pool.iloc[-window:]
    try:
        model = ARIMA(train.values, order=ARIMA_ORDER).fit()
        preds = np.asarray(model.forecast(steps=holdout), dtype=float)
    except Exception:
        return None

    y_true = test.values.astype(float)
    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mape = float(np.mean(np.abs((y_true - preds) / y_true)) * 100)

    return {
        "window_months": window,
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6),
    }


def plot_results(results: list[dict], out_dir: Path) -> None:
    windows = [r["window_months"] for r in results]
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(windows, maes, marker="o", linewidth=2, color="#2563eb", label="MAE")
    axes[0].set_xlabel("Janela de treino (meses)")
    axes[0].set_ylabel("MAE (R$/saca)")
    axes[0].set_title("Sensibilidade ao tamanho da janela de treino — ARIMA(1,1,0)")
    axes[0].set_xticks(windows)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    best = min(results, key=lambda r: r["mae"])
    axes[0].annotate(
        f"Mín: {best['window_months']}m\nMAE={best['mae']:.3f}",
        xy=(best["window_months"], best["mae"]),
        xytext=(best["window_months"] + 8, best["mae"] + 0.5),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    axes[1].plot(windows, rmses, marker="s", linewidth=2, color="#dc2626", label="RMSE")
    axes[1].set_xlabel("Janela de treino (meses)")
    axes[1].set_ylabel("RMSE (R$/saca)")
    axes[1].set_title("RMSE por janela de treino — ARIMA(1,1,0)")
    axes[1].set_xticks(windows)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.savefig(out_dir / "sensitivity_window_mae.png", dpi=150)
    plt.close(fig)
    print("  Salvo: sensitivity_window_mae.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando: {DATA_FILE.relative_to(PROJECT_ROOT)}")
    series = load_series(DATA_FILE)
    print(f"Série: {len(series)} obs | {series.index[0].date()} → {series.index[-1].date()}")
    print(
        f"Holdout fixo: {HOLDOUT_MONTHS} meses "
        f"({series.index[-HOLDOUT_MONTHS].date()} → {series.index[-1].date()})"
    )

    results = []
    print(f"\nTestando {len(WINDOW_SIZES)} janelas de treino com ARIMA{ARIMA_ORDER}:")
    for w in WINDOW_SIZES:
        r = evaluate_window(series, w, HOLDOUT_MONTHS)
        if r is None:
            print(f"  Janela {w:>4}m: dados insuficientes, ignorada.")
            continue
        results.append(r)
        print(
            f"  Janela {w:>4}m | treino {r['train_start']} → {r['train_end']}"
            f" | MAE {r['mae']:.4f} | RMSE {r['rmse']:.4f} | MAPE {r['mape']:.2f}%"
        )

    if not results:
        sys.exit("Nenhuma janela válida encontrada.")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "sensitivity_window_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Salvo: sensitivity_window_results.csv")

    out_json = OUT_DIR / "sensitivity_window_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Salvo: sensitivity_window_results.json")

    best = min(results, key=lambda r: r["mae"])
    worst = max(results, key=lambda r: r["mae"])
    print(f"\n=== Resumo ===")
    print(f"  Menor MAE: janela {best['window_months']}m → {best['mae']:.4f}")
    print(f"  Maior MAE: janela {worst['window_months']}m → {worst['mae']:.4f}")
    print(f"  Variação: {worst['mae'] - best['mae']:.4f} R$/saca")
    print(f"\n{df[['window_months', 'mae', 'rmse', 'mape']].to_string(index=False)}")

    print("\nGerando plot...")
    plot_results(results, OUT_DIR)

    print(f"\nAnálise completa. Arquivos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido pelo usuário.")
