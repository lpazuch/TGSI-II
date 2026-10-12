#!/usr/bin/env python3
"""
Holdout final de 12 meses — ARIMA(1,1,0), Random Forest, XGBoost.

Alvo unico do TGSI II: ``soy_price_brl_bag_next_month`` (preco da soja
CEPEA/ESALQ - Paranagua, R$/saca, um mes a frente).

  dados do mes t  ->  preco da soja em BRL no mes t+1

Os ultimos 12 meses da serie sao reservados como teste final; o restante
e usado para treino. Nenhuma observacao futura entra no treino.

Protocolo UNICO para os tres modelos — walk-forward expansivo de 1 passo
(auditoria de 13/09/2026, item 12; uniformizado em 11/10/2026, item 18):

  para cada uma das 12 origens i (as 12 ultimas linhas da base):
      treina com as linhas [0, i)        (tudo o que se sabe em date[i])
      preve a linha i                    (preco de target_month[i] = t+1)
      incorpora a linha i e reajusta do zero para a origem seguinte

ARIMA(1,1,0), Random Forest e XGBoost passam pela MESMA funcao
``run_walk_forward`` com a MESMA lista de origens (``walk_forward_origins``);
cada modelo so' difere no ``forecaster`` (ARIMA: serie univariada do alvo;
RF/XGB: Pipeline imputer+modelo sobre as features). Hiperparametros fixos,
os mesmos em todas as origens. Antes desta uniformizacao, o ARIMA ja era
reajustado mes a mes, mas RF/XGB eram treinados uma unica vez antes do
holdout — ver docs/methodology.md secao 9.

Historico: a versao do TGSI I tambem rodava uma linha de modelagem em USD
(``soy_price_usd_bag_next_month``) e uma conversao USD->BRL usando o cambio
realizado do holdout. Isso foi REMOVIDO — ver docs/methodology.md secao
"Alvo em BRL e a questao do USD". A coluna ``soy_price_usd_bag`` continua no
dataset como variavel explicativa / rastreabilidade, mas nao ha mais um
experimento oficial em USD.

Saidas em results/holdout_12m/:
  - holdout_12m_predictions.csv   : realizado vs. previsto por modelo
  - holdout_12m_origins.csv       : as 12 origens de cada modelo (treino/alvo)
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
from typing import Callable

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

# Fonte unica do conjunto de colunas de metadado (nao-feature), compartilhada
# com shock_period_analysis.py e com tgsi_pipeline.modeling. Corrigido na
# auditoria de 13/09/2026, item 15: os dois experimentos usavam conjuntos de
# features diferentes (shock_period excluia soy_price_brl_bag/usd_bag; este
# script nao excluia) — agora ambos importam a mesma constante.
from tgsi_pipeline.modeling import METADATA_FIELDS as META_COLUMNS

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
OUT_DIR = PROJECT_ROOT / "results" / "holdout_12m"

HOLDOUT_MONTHS = 12
TARGET_VARIABLE = "soy_price_brl_bag_next_month"


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


def walk_forward_origins(n_obs: int, holdout_months: int) -> list[int]:
    """Origens de previsao do holdout: indices (posicionais) das linhas de
    teste. Para cada origem ``i``, o treino e' ``target_df.iloc[:i]`` e a
    previsao e' feita para a linha ``i`` (features de ``date[i]`` -> preco de
    ``target_month[i]``). Lista UNICA, compartilhada pelos tres modelos."""
    if n_obs <= holdout_months:
        raise ValueError("Observacoes insuficientes para o holdout.")
    return list(range(n_obs - holdout_months, n_obs))


def run_walk_forward(
    model_name: str,
    target_df: pd.DataFrame,
    features: list[str],
    origins: list[int],
    forecaster: Callable[[pd.DataFrame, pd.DataFrame, list[str]], float],
) -> tuple[np.ndarray, list[dict]]:
    """Walk-forward expansivo de 1 passo, identico para todos os modelos:

        treina com as linhas [0, i)  ->  preve a linha i  ->  i += 1

    A linha ``i-1`` (ultima do treino) tem alvo = preco de ``date[i]``, que ja
    e' conhecido na origem ``date[i]``; portanto o treino usa exatamente a
    informacao disponivel no momento da previsao. O modelo e' reajustado do
    zero em cada origem (``forecaster`` recebe so' o recorte de treino).

    Retorna as previsoes e um registro por origem (para auditoria/teste).
    """
    predictions: list[float] = []
    log: list[dict] = []
    for i in origins:
        train = target_df.iloc[:i]
        test_row = target_df.iloc[i : i + 1]
        predictions.append(float(forecaster(train, test_row, features)))
        log.append(
            {
                "model": model_name,
                "origin_index": i,
                "n_train": len(train),
                "train_first_date": train["date"].iloc[0],
                "train_last_date": train["date"].iloc[-1],
                "train_last_target_month": train["target_month"].iloc[-1],
                "forecast_date": test_row["date"].iloc[0],
                "forecast_target_month": test_row["target_month"].iloc[0],
            }
        )
    return np.asarray(predictions, dtype=float), log


def arima_forecaster(train: pd.DataFrame, test_row: pd.DataFrame, features: list[str]) -> float:
    """ARIMA(1,1,0) univariado sobre a serie de alvo do treino; preve 1 passo."""
    series = train["target_value"].astype(float).to_numpy()
    try:
        return float(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1)[0])
    except Exception:  # noqa: BLE001
        return float(series[-1]) if len(series) else float("nan")


def make_tabular_forecaster(model_factory: Callable[[], object]):
    """Forecaster para RF/XGBoost: novo Pipeline (imputer + modelo) ajustado
    so' no treino da origem, prevendo a partir das features da linha de teste."""

    def forecaster(train: pd.DataFrame, test_row: pd.DataFrame, features: list[str]) -> float:
        pipeline = build_tabular_pipeline(model_factory())
        pipeline.fit(train[features], train["target_value"].astype(float).to_numpy())
        return float(pipeline.predict(test_row[features])[0])

    return forecaster


def random_forest_factory() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def xgboost_factory():
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        objective="reg:squarederror",
        random_state=42,
    )


def build_forecasters() -> dict[str, Callable]:
    forecasters: dict[str, Callable] = {
        "arima_1_1_0": arima_forecaster,
        "random_forest": make_tabular_forecaster(random_forest_factory),
    }
    if HAS_XGBOOST:
        forecasters["xgboost"] = make_tabular_forecaster(xgboost_factory)
    else:
        print("[aviso] xgboost nao disponivel; seguindo sem esse modelo.")
    return forecasters


def run_models(
    target_df: pd.DataFrame,
    features: list[str],
    holdout_months: int = HOLDOUT_MONTHS,
    forecasters: dict[str, Callable] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = walk_forward_origins(len(target_df), holdout_months)
    test_df = target_df.iloc[origins]
    y_test = test_df["target_value"].astype(float).to_numpy()

    predictions = pd.DataFrame(
        {
            "date": test_df["date"].to_numpy(),
            "target_month": test_df["target_month"].to_numpy(),
            "actual": y_test,
        }
    )
    metrics: list[dict] = []
    origin_logs: list[dict] = []

    for name, forecaster in (forecasters or build_forecasters()).items():
        pred, log = run_walk_forward(name, target_df, features, origins, forecaster)
        predictions[name] = pred
        metrics.append(evaluate_model(name, y_test, pred))
        origin_logs.extend(log)

    metrics_df = pd.DataFrame(metrics).sort_values("mae").reset_index(drop=True)
    return predictions, metrics_df, pd.DataFrame(origin_logs)


def plot_forecast(predictions: pd.DataFrame, out_dir: Path) -> None:
    plot_df = predictions.copy()
    # Eixo X = target_month (o mes AO QUAL o valor previsto pertence), nao
    # date (o mes de entrada/features). Corrigido na auditoria de 13/09/2026,
    # item 14 — usar `date` deslocava a previsao de t+1 para a posicao de t.
    plot_df["target_month"] = pd.to_datetime(plot_df["target_month"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(plot_df["target_month"], plot_df["actual"], marker="o", linewidth=3, color="black", label="Realizado")
    for col in [c for c in plot_df.columns if c not in {"date", "target_month", "actual"}]:
        ax.plot(plot_df["target_month"], plot_df[col], marker="o", linewidth=2, label=col)
    ax.set_title("Holdout 12 meses — walk-forward expansivo 1 passo (alvo BRL, t+1)")
    ax.set_xlabel("Mes previsto (target_month)")
    ax.set_ylabel("Preco da soja (R$/saca)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_dir / "holdout_12m_forecast.png", dpi=150)
    plt.close(fig)


def plot_metrics(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df.set_index("model")[["mae", "rmse", "mape"]].plot(kind="bar", ax=ax)
    ax.set_title("Metricas por modelo — holdout 12m, walk-forward expansivo 1 passo")
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
    print("Protocolo: walk-forward expansivo de 1 passo, reajuste mensal (ARIMA, RF e XGBoost)")
    print(f"Treino inicial: {target_df['date'].iloc[0].date()} -> {target_df['date'].iloc[-HOLDOUT_MONTHS - 1].date()}")
    print(f"Origens (date): {target_df['date'].iloc[-HOLDOUT_MONTHS].date()} -> {target_df['date'].iloc[-1].date()}")

    predictions, metrics_df, origins_df = run_models(target_df, features)

    print("\n=== Metricas (holdout 12m, alvo BRL) ===")
    print(metrics_df.to_string(index=False))

    predictions.to_csv(OUT_DIR / "holdout_12m_predictions.csv", index=False)
    origins_df.to_csv(OUT_DIR / "holdout_12m_origins.csv", index=False)
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
