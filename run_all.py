#!/usr/bin/env python3
"""
run_all.py — reprodução completa da modelagem e análises do TCC.

Pré-requisito: os dados já estão em
  data/run_cepea_hybrid_2006plus_long_remote/

Dependências:
  pip install -r requirements.txt

Executa em sequência:
  1. Holdout 12 meses — ARIMA, RF, XGBoost
  2. Análise de resíduos ARIMA (Ljung-Box, ACF/PACF)
  3. Análise de sensibilidade por janela de treino
  4. Correlação cruzada features × alvo
  5. Período de choque 2020–2022 — todos os modelos

Uso:
    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "data" / "run_cepea_hybrid_2006plus_long_remote"
FEATURES_CSV = RUN_DIR / "processed" / "features_monthly_modeling.csv"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SCRIPTS = [
    ("1/5  Holdout 12 meses — ARIMA, Random Forest, XGBoost",
     "holdout_12m.py", ["--input", str(FEATURES_CSV)]),
    ("2/5  Análise de resíduos ARIMA (Ljung-Box, ACF/PACF)",
     "analyze_residuals.py", []),
    ("3/5  Sensibilidade — MAE por janela de treino",
     "sensitivity_analysis.py", []),
    ("4/5  Correlação cruzada — features × alvo",
     "cross_correlation_analysis.py", []),
    ("5/5  Período de choque 2020–2022 — todos os modelos",
     "shock_period_analysis.py", []),
]


def main() -> None:
    print("=" * 62)
    print("  TCC TGSI I — UFSM | Previsão de Preços da Soja (CEPEA)")
    print("  Dataset: run_cepea_hybrid_2006plus_long_remote (20 anos)")
    print("=" * 62)

    if not FEATURES_CSV.exists():
        sys.exit(
            f"[ERRO] Arquivo não encontrado:\n  {FEATURES_CSV}\n"
            "Certifique-se de que os dados estão em data/run_cepea_hybrid_2006plus_long_remote/"
        )

    for title, script, extra_args in SCRIPTS:
        bar = "─" * 60
        print(f"\n{bar}\n  {title}\n{bar}")
        cmd = [sys.executable, str(SCRIPTS_DIR / script), *extra_args]
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"\n[ERRO] {script} falhou com código {result.returncode}.")

    print("\n" + "=" * 62)
    print("  Concluído. Outputs salvos em:")
    print("=" * 62)
    outputs = [
        ("Holdout 12m",        FEATURES_CSV.parent / "features_monthly_modeling_holdout_12m_outputs"),
        ("Resíduos",           PROJECT_ROOT / "docs/academic_outputs/residual_analysis"),
        ("Sensibilidade",      PROJECT_ROOT / "docs/academic_outputs/sensitivity_analysis"),
        ("Cross-correlação",   PROJECT_ROOT / "docs/academic_outputs/cross_correlation"),
        ("Choque 2020-2022",   PROJECT_ROOT / "docs/academic_outputs/shock_period"),
    ]
    for label, path in outputs:
        mark = "✓" if path.exists() else "?"
        print(f"  {mark}  {label:20s}  {path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
