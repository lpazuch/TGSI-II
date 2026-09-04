#!/usr/bin/env python3
"""
run_experiments.py — roda todos os experimentos do TGSI II em sequencia.

Pre-requisito: a base ja construida em
    data/processed/features_monthly_modeling.csv
(gere com:  PYTHONPATH=src python scripts/build_dataset.py)

Executa, nesta ordem:
  1. Holdout final de 12 meses  — ARIMA(1,1,0), Random Forest, XGBoost (alvo BRL)
  2. Diagnostico de residuos do ARIMA — Ljung-Box, ACF/PACF
  3. Sensibilidade do ARIMA por janela de treino
  4. Correlacao cruzada features x alvo (justificativa das defasagens)
  5. Robustez no periodo de choque 2020-2022 — walk-forward, todos os modelos

Saidas em results/<experimento>/.

Uso:
    PYTHONPATH=src python scripts/run_experiments.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"

EXPERIMENTS = [
    ("1/5  Holdout final de 12 meses — ARIMA, Random Forest, XGBoost", "holdout_12m.py"),
    ("2/5  Diagnostico de residuos do ARIMA (Ljung-Box, ACF/PACF)", "arima_residuals.py"),
    ("3/5  Sensibilidade do ARIMA por janela de treino", "arima_sensitivity.py"),
    ("4/5  Correlacao cruzada — features x alvo", "cross_correlation.py"),
    ("5/5  Periodo de choque 2020-2022 — walk-forward, todos os modelos", "shock_period_analysis.py"),
]


def main() -> None:
    print("=" * 64)
    print("  TGSI II — UFSM | Previsao do preco mensal da soja (CEPEA/ESALQ)")
    print("  Alvo: soy_price_brl_bag_next_month  (R$/saca, t -> t+1)")
    print("=" * 64)

    if not DATASET.exists():
        sys.exit(
            f"[ERRO] Base nao encontrada:\n  {DATASET}\n"
            "Gere primeiro:  PYTHONPATH=src python scripts/build_dataset.py"
        )

    env_hint = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    for title, script in EXPERIMENTS:
        bar = "-" * 62
        print(f"\n{bar}\n  {title}\n{bar}")
        cmd = [sys.executable, str(EXPERIMENTS_DIR / script)]
        print(f"$ PYTHONPATH=src {' '.join(cmd)}")
        result = subprocess.run(cmd, env={**_os_environ(), **env_hint})
        if result.returncode != 0:
            sys.exit(f"\n[ERRO] {script} falhou com codigo {result.returncode}.")

    print("\n" + "=" * 64)
    print("  Concluido. Resultados em:")
    print("=" * 64)
    for label, slug in [
        ("Holdout 12m", "holdout_12m"),
        ("Residuos ARIMA", "arima_residuals"),
        ("Sensibilidade ARIMA", "arima_sensitivity"),
        ("Correlacao cruzada", "cross_correlation"),
        ("Periodo de choque", "shock_period"),
    ]:
        path = RESULTS_DIR / slug
        mark = "OK " if path.exists() else " ? "
        print(f"  [{mark}] {label:22s} results/{slug}/")
    print()


def _os_environ() -> dict:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    main()
