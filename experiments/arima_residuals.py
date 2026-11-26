#!/usr/bin/env python3
"""
Diagnostico do ARIMA(1,1,0) e comparacao de especificacoes — soja CEPEA/ESALQ.

Tudo aqui usa SOMENTE dados anteriores ao holdout oficial (meses-alvo
mai/2025 -> abr/2026, ver holdout_12m.py). Revisao de 16/11/2026: a versao
anterior treinava ate abr/2024 e media o MAE em mai/2024 -> abr/2026, janela
que continha os 12 meses do holdout.

Janelas (serie mensal a partir de abr/2006; mar/2006 e' mes parcial e fica
fora, como na base de modelagem):
  - treino     : abr/2006 -> abr/2023 (205 obs) — ajuste, AIC e residuos
  - validacao  : mai/2023 -> abr/2025 (24 meses) — MAE de 1 passo,
                 walk-forward expansivo (reajuste a cada mes)
  - holdout    : mai/2025 -> abr/2026 — NAO e' usado neste script

Gera em results/arima_residuals/:
  - residuals_time.png / residuals_acf.png / residuals_pacf.png / residuals_hist.png
  - residual_analysis.json : estatisticas, Ljung-Box e comparacao de especificacoes
  - spec_comparison.csv    : AIC (treino) e MAE (validacao) por especificacao

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

# legibilidade na impressao (figuras com ~15 cm no artigo)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "target_soy_cepea_monthly.csv"
OUT_DIR = PROJECT_ROOT / "results" / "arima_residuals"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tgsi_pipeline.config import load_config  # noqa: E402
from tgsi_pipeline.pipeline import complete_month_bounds  # noqa: E402

# Apenas meses inteiramente cobertos pela extracao (mar/2006 e abr/2026 sao
# parciais e ficam fora, como na base de modelagem).
_DATE_RANGE = load_config(PROJECT_ROOT / "configs" / "dataset.hybrid_2006plus.json").date_range
SERIES_START, SERIES_END = complete_month_bounds(_DATE_RANGE.start, _DATE_RANGE.end)
HOLDOUT_MONTHS = 12  # holdout oficial: os 12 ultimos meses-alvo da serie
VALIDATION_MONTHS = 24
ARIMA_ORDER = (1, 1, 0)
CANDIDATE_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (2, 1, 1)]
LJUNG_BOX_LAGS = [6, 12, 18, 24]


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["soy_price_brl_bag"]).reset_index(drop=True)
    df = df[(df["date"] >= pd.Timestamp(SERIES_START)) & (df["date"] <= pd.Timestamp(SERIES_END))]
    return pd.Series(df["soy_price_brl_bag"].values, index=df["date"], name="soy_price_brl_bag")


def split_windows(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Treino e validacao, ambos estritamente anteriores ao holdout oficial."""
    pre = series.iloc[:-HOLDOUT_MONTHS]
    if len(pre) <= VALIDATION_MONTHS:
        raise ValueError("Serie curta demais antes do holdout.")
    return pre.iloc[:-VALIDATION_MONTHS], pre.iloc[-VALIDATION_MONTHS:]


def one_step_mae(train: pd.Series, valid: pd.Series, order: tuple) -> float:
    """MAE de 1 passo em walk-forward expansivo, reajustando a cada mes."""
    history = list(train.values)
    errors = []
    for y in valid.values:
        fc = ARIMA(np.asarray(history), order=order).fit().forecast(1)[0]
        errors.append(abs(y - fc))
        history.append(y)
    return float(np.mean(errors))


def compare_specs(train: pd.Series, valid: pd.Series) -> pd.DataFrame:
    rows = []
    for order in CANDIDATE_ORDERS:
        aic = ARIMA(train.values, order=order).fit().aic
        rows.append({"order": f"ARIMA{order}", "aic_train": round(float(aic), 2),
                     "mae_valid": round(one_step_mae(train, valid, order), 3)})
    persist = np.concatenate([[train.values[-1]], valid.values[:-1]])
    rows.append({"order": "persistencia", "aic_train": None,
                 "mae_valid": round(float(np.mean(np.abs(valid.values - persist))), 3)})
    return pd.DataFrame(rows)


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
    ax.set_xlabel("Defasagem (meses)")
    ax.set_ylabel("Autocorrelação")
    plt.tight_layout()
    fig.savefig(out_dir / "residuals_acf.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: residuals_acf.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(residuals, lags=24, ax=ax, title="PACF dos Resíduos — ARIMA(1,1,0)", method="ywm")
    ax.set_xlabel("Defasagem (meses)")
    ax.set_ylabel("Autocorrelação parcial")
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

    train, valid = split_windows(series)
    print(f"Treino   : {len(train)} obs | {train.index[0].date()} → {train.index[-1].date()}")
    print(f"Validação: {len(valid)} obs | {valid.index[0].date()} → {valid.index[-1].date()}")
    print(f"Holdout ({series.index[-HOLDOUT_MONTHS].date()} em diante) não é usado.")

    print("\nComparando especificações (AIC no treino, MAE 1 passo na validação)...")
    specs = compare_specs(train, valid)
    print(specs.to_string(index=False))
    specs.to_csv(OUT_DIR / "spec_comparison.csv", index=False)

    print(f"\nAjustando ARIMA{ARIMA_ORDER} no treino...")
    model = ARIMA(train.values, order=ARIMA_ORDER).fit()

    # o primeiro residuo de um modelo com d = 1 e' o proprio nivel da serie
    # (sem valor anterior para diferenciar); fica fora do diagnostico
    d = ARIMA_ORDER[1]
    residuals = np.asarray(model.resid, dtype=float)[d:]
    residual_dates = train.index[d:]

    print(f"\nEstatísticas dos resíduos ({len(residuals)} obs):")
    print(f"  Média     : {residuals.mean():.4f}")
    print(f"  Desvio pad: {residuals.std():.4f}")
    print(f"  Min / Max : {residuals.min():.4f} / {residuals.max():.4f}")
    print(f"  AIC treino: {model.aic:.4f}")

    lb_results = run_ljung_box(residuals, LJUNG_BOX_LAGS)
    print("\n=== Teste de Ljung-Box (H0: resíduos são ruído branco) ===")
    for r in lb_results:
        verdict = "SIM (p>0.05)" if r["white_noise"] else "NÃO (p≤0.05)"
        print(f"{r['lag']:>5}  {r['lb_stat']:>10.4f}  {r['lb_pvalue']:>10.4f}  {verdict:>14}")

    summary = {
        "arima_order": list(ARIMA_ORDER),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "train": {"start": str(train.index[0].date()), "end": str(train.index[-1].date()),
                  "obs": int(len(train))},
        "validation": {"start": str(valid.index[0].date()), "end": str(valid.index[-1].date()),
                       "obs": int(len(valid))},
        "first_holdout_target": str(series.index[-HOLDOUT_MONTHS].date()),
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
        "spec_comparison": specs.where(specs.notna(), None).to_dict(orient="records"),
    }
    out_json = OUT_DIR / "residual_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  Salvo: residual_analysis.json, spec_comparison.csv")

    print("\nGerando plots...")
    plot_residuals(residuals, residual_dates, OUT_DIR)
    print(f"\nAnálise completa. Arquivos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrompido pelo usuário.")
