#!/usr/bin/env python3
"""
Teste de Diebold-Mariano (DM) para comparar a acuracia preditiva dos modelos.

H0: os dois modelos tem a mesma acuracia esperada, E[d_t] = 0, onde
    d_t = L(e_A,t) - L(e_B,t)       (L = perda; e = realizado - previsto)
Estatistica (Diebold e Mariano, 1995), horizonte h = 1:
    DM = media(d) / sqrt( V / n ),   V = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k
Correcao para amostras pequenas (Harvey, Leybourne e Newbold, 1997 — HLN):
    DM* = DM * sqrt( (n + 1 - 2h + h(h-1)/n) / n ),   p-valor pela t de Student
    com n - 1 graus de liberdade (bilateral).
Como verificacao de robustez, tambem e' calculada a variancia de longo prazo
de Newey-West (nucleo de Bartlett, defasagem floor(4 (n/100)^(2/9))), que
acomoda eventual autocorrelacao do diferencial de perdas.

Sinal: DM* < 0 => o modelo A tem perda MENOR (e' mais preciso) que o B.

Amostras (nenhuma previsao nova e' gerada — usa as saidas dos experimentos):
  holdout_12m      results/holdout_12m/holdout_12m_predictions.csv (12 previsoes;
                   persistencia = preco observado em `date`, da base de modelagem)
  walkforward_216  results/shock_period/shock_period_walkforward.csv
                   (216 previsoes, alvos mai/2008 -> abr/2026), e os recortes
                   pre-choque / choque / pos-choque desse mesmo arquivo

Perdas: absoluta (|e|, coerente com o MAE — principal) e quadratica (e^2).

Saidas em results/diebold_mariano/:
  - dm_results.csv / .json

Pre-requisito: rodar antes holdout_12m.py e shock_period_analysis.py.
Uso:
  PYTHONPATH=src python experiments/diebold_mariano.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
DATASET = PROJECT_ROOT / "data" / "processed" / "features_monthly_modeling.csv"
OUT_DIR = RESULTS / "diebold_mariano"

MODELS = ["arima_1_1_0", "persistencia", "random_forest", "xgboost"]
LABELS = {"arima_1_1_0": "ARIMA(1,1,0)", "persistencia": "Persistência",
          "random_forest": "Random Forest", "xgboost": "XGBoost"}
LOSSES = {"absoluta": np.abs, "quadratica": np.square}


def _autocov(d: np.ndarray, k: int) -> float:
    n = len(d)
    dc = d - d.mean()
    return float(np.dot(dc[k:], dc[: n - k]) / n)


def newey_west_lag(n: int) -> int:
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int = 1, loss=np.abs,
                    hac_lag: int | None = None) -> dict:
    """DM com correcao HLN. ``hac_lag`` != None usa variancia de Newey-West
    (Bartlett) com essa defasagem, em vez da variancia padrao do DM."""
    e_a = np.asarray(e_a, dtype=float)
    e_b = np.asarray(e_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("Series de erro com tamanhos diferentes.")
    d = loss(e_a) - loss(e_b)
    n = len(d)
    if n < 3:
        raise ValueError("Amostra pequena demais para o teste DM.")
    if hac_lag is None:
        var = _autocov(d, 0) + 2 * sum(_autocov(d, k) for k in range(1, h))
    else:
        var = _autocov(d, 0) + 2 * sum((1 - k / (hac_lag + 1)) * _autocov(d, k)
                                       for k in range(1, hac_lag + 1))
    if var <= 0:
        return {"n": n, "mean_d": float(d.mean()), "dm": float("nan"),
                "dm_hln": float("nan"), "p_value": float("nan")}
    dm = float(d.mean() / math.sqrt(var / n))
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = float(2 * stats.t.sf(abs(hln), df=n - 1))
    return {"n": n, "mean_d": float(d.mean()), "dm": dm, "dm_hln": float(hln), "p_value": p}


def load_holdout() -> pd.DataFrame:
    pred = pd.read_csv(RESULTS / "holdout_12m" / "holdout_12m_predictions.csv",
                       parse_dates=["date", "target_month"])
    base = pd.read_csv(DATASET, parse_dates=["date"])[["date", "soy_price_brl_bag"]]
    pred = pred.merge(base, on="date", how="left")
    if pred["soy_price_brl_bag"].isna().any():
        raise SystemExit("Persistencia: preco observado ausente para alguma origem do holdout.")
    return pred.rename(columns={"soy_price_brl_bag": "persistencia"})


def load_walkforward() -> pd.DataFrame:
    wf = pd.read_csv(RESULTS / "shock_period" / "shock_period_walkforward.csv",
                     parse_dates=["target_month"])
    return wf.rename(columns={"naive_last_value": "persistencia"}).dropna(subset=MODELS)


def run_sample(name: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    y = df["actual"].to_numpy(dtype=float)
    errors = {m: y - df[m].to_numpy(dtype=float) for m in MODELS}
    lag = newey_west_lag(len(df))
    for a, b in itertools.combinations(MODELS, 2):
        for loss_name, loss in LOSSES.items():
            r = diebold_mariano(errors[a], errors[b], h=1, loss=loss)
            r_nw = diebold_mariano(errors[a], errors[b], h=1, loss=loss, hac_lag=lag)
            rows.append({
                "sample": name, "model_a": a, "model_b": b, "loss": loss_name,
                "n": r["n"], "loss_a": float(loss(errors[a]).mean()),
                "loss_b": float(loss(errors[b]).mean()), "mean_d": r["mean_d"],
                "dm_hln": r["dm_hln"], "p_value": r["p_value"],
                "nw_lag": lag, "dm_hln_nw": r_nw["dm_hln"], "p_value_nw": r_nw["p_value"],
            })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = {"holdout_12m": load_holdout()}
    wf = load_walkforward()
    samples["walkforward_216"] = wf
    for period in ("pre_shock", "shock", "pos_shock"):
        samples[f"walkforward_{period}"] = wf[wf["period"] == period]

    rows = []
    for name, df in samples.items():
        rows.extend(run_sample(name, df))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "dm_results.csv", index=False)
    (OUT_DIR / "dm_results.json").write_text(
        json.dumps(res.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    show = res[res["loss"] == "absoluta"].copy()
    show["par"] = show["model_a"].map(LABELS) + " vs " + show["model_b"].map(LABELS)
    pd.set_option("display.width", 200)
    print("Perda absoluta (DM com correcao HLN; DM* < 0 => primeiro modelo mais preciso)")
    print(show[["sample", "par", "n", "loss_a", "loss_b", "dm_hln", "p_value", "dm_hln_nw",
                "p_value_nw"]].round(4).to_string(index=False))
    sq = res[res["loss"] == "quadratica"].copy()
    sq["par"] = sq["model_a"].map(LABELS) + " vs " + sq["model_b"].map(LABELS)
    print("\nPerda quadratica")
    print(sq[["sample", "par", "n", "dm_hln", "p_value"]].round(4).to_string(index=False))
    print(f"\nArquivos salvos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        sys.exit(f"[ERRO] {exc}\nRode antes holdout_12m.py e shock_period_analysis.py.")
