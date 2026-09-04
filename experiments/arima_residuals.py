#!/usr/bin/env python3
"""
Análise de resíduos do modelo ARIMA(1,1,0) — soja CEPEA/ESALQ Paranaguá.

Gera:
  - residuals_time.png   : resíduos ao longo do tempo
  - residuals_acf.png    : ACF dos resíduos
  - residuals_pacf.png   : PACF dos resíduos
  - residuals_hist.png   : histograma dos resíduos
  - residual_analysis.json : estatísticas + teste Ljung-Box

Uso:
  PYTHONPATH=src python experiments/arima_residuals.py
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "target_soy_cepea_monthly.csv"
OUT_DIR = PROJECT_ROOT / "results" / "arima_residuals"

HOLDOUT_MONTHS = 24
ARIMA_ORDER = (1, 1, 0)
LJUNG_BOX_LAGS = [6, 12, 18, 24]


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["soy_price_brl_bag"]).reset_index(drop=True)
    return pd.Series(df["soy_price_brl_bag"].values, index=df["date"], name="soy_price_brl_bag")


def run_ljung_box(residuals: np.ndarray, lags: list[int]) -> list[dict]:
    lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
    results = []
    for lag in lags:
        row = lb.loc[lag]
        results.append(
            {
                "lag": int(lag),
                "lb_stat": round(float(row["lb_stat"]), 6),
                "lb_pvalue": round(float(row["lb_pvalue"]), 6),
                "white_noise": bool(float(row["lb_pvalue"]) > 0.05),
            }
        )
    return results


def plot_residuals(residuals: np.ndarray, dates: pd.DatetimeIndex, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, residuals, color="#2563eb", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Resíduos do ARIMA(1,1,0) — série temporal")
    ax.set_xlabel("Data")
    ax.set_ylabel("Resíduo (R$/saca)")
    plt.tight_layout()
    fig.savefig(out_dir / "residuals_time.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: residuals_time.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=24, ax=ax, title="ACF dos Resíduos — ARIMA(1,1,0)")
    plt.tight_layout()
    fig.savefig(out_dir / "residuals_acf.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: residuals_acf.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(residuals, lags=24, ax=ax, title="PACF dos Resíduos — ARIMA(1,1,0)", method="ywm")
    plt.tight_layout()
    fig.savefig(out_dir / "residuals_pacf.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: residuals_pacf.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=25, color="#2563eb", edgecolor="white", alpha=0.85)
    ax.set_title("Histograma dos Resíduos — ARIMA(1,1,0)")
    ax.set_xlabel("Resíduo (R$/saca)")
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    fig.savefig(out_dir / "residuals_hist.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: residuals_hist.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando: {DATA_FILE.relative_to(PROJECT_ROOT)}")
    series = load_series(DATA_FILE)
    print(f"Série: {len(series)} obs | {series.index[0].date()} → {series.index[-1].date()}")

    train = series.iloc[:-HOLDOUT_MONTHS]
    holdout_start = series.index[-HOLDOUT_MONTHS].date()
    holdout_end = series.index[-1].date()
    print(f"Treino: {len(train)} obs | Holdout reservado: {holdout_start} → {holdout_end}")

    print(f"\nAjustando ARIMA{ARIMA_ORDER} nos dados de treino...")
    model = ARIMA(train.values, order=ARIMA_ORDER).fit()

    residuals = np.asarray(model.resid, dtype=float)
    residual_dates = train.index[len(train) - len(residuals):]

    print(f"\nEstatísticas dos resíduos ({len(residuals)} obs):")
    print(f"  Média     : {residuals.mean():.4f}")
    print(f"  Desvio pad: {residuals.std():.4f}")
    print(f"  Min / Max : {residuals.min():.4f} / {residuals.max():.4f}")
    print(f"  AIC treino: {model.aic:.4f}")
    print(f"  BIC treino: {model.bic:.4f}")

    lb_results = run_ljung_box(residuals, LJUNG_BOX_LAGS)

    print("\n=== Teste de Ljung-Box (H0: resíduos são ruído branco) ===")
    print(f"{'Lag':>5}  {'Stat':>10}  {'p-valor':>10}  {'Ruído branco?':>14}")
    print("-" * 48)
    for r in lb_results:
        verdict = "SIM (p>0.05)" if r["white_noise"] else "NÃO (p≤0.05)"
        print(f"{r['lag']:>5}  {r['lb_stat']:>10.4f}  {r['lb_pvalue']:>10.4f}  {verdict:>14}")

    summary = {
        "arima_order": list(ARIMA_ORDER),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "train_obs": int(len(train)),
        "holdout_months": HOLDOUT_MONTHS,
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "residuals_stats": {
            "n": len(residuals),
            "mean": round(float(residuals.mean()), 6),
            "std": round(float(residuals.std()), 6),
            "min": round(float(residuals.min()), 6),
            "max": round(float(residuals.max()), 6),
        },
        "ljung_box": lb_results,
    }

    out_json = OUT_DIR / "residual_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Salvo: residual_analysis.json")

    print("\nGerando plots...")
    plot_residuals(residuals, residual_dates, OUT_DIR)

    print(f"\nAnálise completa. Arquivos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido pelo usuário.")
