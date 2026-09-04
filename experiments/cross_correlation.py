#!/usr/bin/env python3
"""
Análise de correlação cruzada (CCF) entre features e preço da soja.

Justifica as defasagens escolhidas para cada variável exógena, documentando
a cadeia causal: vegetação → produtividade → oferta → mercado → preço.

Saídas em results/cross_correlation/:
  - ccf_cambio_e_petroleo.png
  - ccf_clima.png
  - ccf_sensoriamento_remoto.png
  - ccf_summary.csv
  - ccf_results.json

Uso:
  PYTHONPATH=src python experiments/cross_correlation.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
OUT_DIR = PROJECT_ROOT / "results" / "cross_correlation"

TARGET_VARIABLE = "soy_price_brl_bag_next_month"
MAX_LAG = 12

FEATURE_GROUPS = {
    "Câmbio e Petróleo": ["usd_brl", "brent_usd_bbl"],
    "Clima": ["precipitation_mm", "temperature_mean_c", "relative_humidity_pct", "wind_speed_ms"],
    "Sensoriamento Remoto": ["ndvi", "soil_moisture_m3m3"],
}

LABELS = {
    "usd_brl": "Câmbio USD/BRL",
    "brent_usd_bbl": "Petróleo Brent (USD/bbl)",
    "precipitation_mm": "Precipitação (mm)",
    "temperature_mean_c": "Temperatura média (°C)",
    "relative_humidity_pct": "Umidade relativa (%)",
    "wind_speed_ms": "Velocidade do vento (m/s)",
    "ndvi": "NDVI",
    "soil_moisture_m3m3": "Umidade do solo (m³/m³)",
}


def cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """CCF de x (feature em t) com y (target em t+lag), lags 0..max_lag."""
    x_std = (x - x.mean()) / (x.std() or 1.0)
    y_std = (y - y.mean()) / (y.std() or 1.0)
    lags = np.arange(0, max_lag + 1)
    cors = np.empty(len(lags))
    for i, lag in enumerate(lags):
        if lag == 0:
            cors[i] = float(np.corrcoef(x_std, y_std)[0, 1])
        else:
            cors[i] = float(np.corrcoef(x_std[:-lag], y_std[lag:])[0, 1])
    return lags, cors


def plot_ccf_group(
    group_name: str, features: list[str], df: pd.DataFrame, out_dir: Path
) -> dict:
    available = [f for f in features if f in df.columns and df[f].notna().sum() >= 10]
    if not available:
        return {}

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.8 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    conf = 1.96 / np.sqrt(len(df))
    results: dict = {}

    for ax, feat in zip(axes, available):
        valid = df[["target_value", feat]].dropna()
        if len(valid) < 12:
            ax.set_visible(False)
            continue

        lags, cors = cross_corr(valid[feat].values, valid["target_value"].values, MAX_LAG)
        colors = ["#2563eb" if abs(c) > conf else "#93c5fd" for c in cors]

        ax.bar(lags, cors, color=colors, width=0.55, edgecolor="white", linewidth=0.3)
        ax.axhline(conf, color="#dc2626", linestyle="--", linewidth=0.9, label=f"±{conf:.3f} (IC 95%)")
        ax.axhline(-conf, color="#dc2626", linestyle="--", linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"CCF: {LABELS.get(feat, feat)}  →  Preço da soja t+1")
        ax.set_xlabel("Lag (meses)")
        ax.set_ylabel("Correlação")
        ax.set_xticks(lags)
        ax.legend(fontsize=8, loc="upper right")

        best_idx = int(np.argmax(np.abs(cors)))
        best_lag = int(lags[best_idx])
        best_corr = float(cors[best_idx])

        results[feat] = {
            "label": LABELS.get(feat, feat),
            "n_obs": int(len(valid)),
            "lags": lags.tolist(),
            "correlations": [round(float(c), 6) for c in cors],
            "best_lag": best_lag,
            "best_corr": round(best_corr, 6),
            "confidence_band_95": round(float(conf), 6),
        }

    slug = (
        group_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("ç", "c")
        .replace("â", "a")
        .replace("ã", "a")
        .replace("ê", "e")
        .replace("ó", "o")
        .replace("ô", "o")
    )
    fig.suptitle(f"Correlação Cruzada — {group_name}", fontsize=13, fontweight="bold")
    out_path = out_dir / f"ccf_{slug}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando: {DATA_FILE.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = (
        df[df["target_variable"] == TARGET_VARIABLE]
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"Observações ({TARGET_VARIABLE}): {len(df)}")
    print(f"Período: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    all_results: dict = {}
    for group, features in FEATURE_GROUPS.items():
        print(f"\nGrupo: {group}")
        group_results = plot_ccf_group(group, features, df, OUT_DIR)
        all_results.update(group_results)

    if not all_results:
        sys.exit("Nenhuma feature disponível para análise.")

    rows = [
        {
            "feature": feat,
            "label": res["label"],
            "n_obs": res["n_obs"],
            "best_lag": res["best_lag"],
            "best_corr": res["best_corr"],
            "conf_95": res["confidence_band_95"],
            "significant": abs(res["best_corr"]) > res["confidence_band_95"],
        }
        for feat, res in all_results.items()
    ]
    summary_df = pd.DataFrame(rows).sort_values("best_corr", key=abs, ascending=False)

    print("\n=== Resumo: melhor lag por variável ===")
    print(f"{'Feature':<25} {'Lag*':>6} {'Corr*':>8} {'Signif?':>9}")
    print("-" * 52)
    for _, row in summary_df.iterrows():
        sig = "SIM" if row["significant"] else "NÃO"
        print(f"{row['feature']:<25} {row['best_lag']:>6} {row['best_corr']:>8.4f} {sig:>9}")

    out_csv = OUT_DIR / "ccf_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  Salvo: ccf_summary.csv")

    out_json = OUT_DIR / "ccf_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Salvo: ccf_results.json")

    print(f"\nAnálise completa. Arquivos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido pelo usuário.")
