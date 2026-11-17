#!/usr/bin/env python3
"""
Ablacao sobre as variaveis efetivamente relevantes — Random Forest e XGBoost.

Duas etapas, sem usar nenhum dado dos periodos de avaliacao:

1) SELECAO (importancia por permutacao, por variavel de origem)
   As 54 features sao agrupadas em 13 variaveis de origem (``feature_family``):
   cada variavel exogena leva junto seus lags, media movel e anomalia (ex.:
   "cambio" = usd_brl + usd_brl_lag_1..3 + usd_brl_rolling_mean_3 +
   usd_brl_month_anomaly). Permutar a familia inteira, com a MESMA permutacao
   de linhas para todas as colunas dela, evita subestimar a importancia de
   variaveis cujos lags sao muito correlacionados entre si.

   Para cada modelo:
     - ajusta o Pipeline (imputer + modelo, mesmos hiperparametros do holdout)
       nas linhas [0, n - 60 - 24)       -> date abr/2006 .. mar/2019
     - mede, nas 24 linhas seguintes     -> date abr/2019 .. mar/2021
       (alvos mai/2019 .. abr/2021), o aumento do MAE ao permutar cada familia
       (N_REPEATS permutacoes, semente fixa).
   Essas 24 linhas sao ANTERIORES a todas as origens avaliadas depois (as 60
   da janela estendida e, portanto, as 12 do holdout) — a selecao nao ve o
   periodo de teste.

   Criterio (definido a priori): familia relevante se
       media(ΔMAE) - 2 * desvio(ΔMAE) > 0.

2) ABLACAO (leave-one-variable-out sobre as relevantes)
   Configuracoes, por modelo:
     todas_54        todas as features (o modelo do holdout oficial)
     relevantes      so' as familias selecionadas para aquele modelo
     sem_<familia>   relevantes menos uma familia (uma configuracao por familia)
   Protocolo identico ao ``holdout_12m.py`` (walk-forward expansivo de 1 passo,
   reajuste em cada origem), em duas janelas:
     holdout_12m    12 origens (alvos mai/2025 -> abr/2026)
     estendida_60m  60 origens (alvos mai/2021 -> abr/2026)
   Referencias nas mesmas janelas: ARIMA(1,1,0) e persistencia ingenua.

Saidas em results/ablation/:
  - importance.csv               : ΔMAE medio/desvio por familia e modelo; selecionada?
  - ablation_metrics.csv / .json : MAE/RMSE/MAPE/R2 por modelo, configuracao e janela
  - importance_permutation.png   : importancia por permutacao (etapa 1)
  - ablation_delta_mae.png       : variacao do MAE na ablacao (etapa 2)

Uso:
  PYTHONPATH=src python experiments/ablation.py
  PYTHONPATH=src python experiments/ablation.py --replot   # so' refaz os graficos
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from holdout_12m import (  # mesmo protocolo e mesmos modelos do holdout oficial
    DEFAULT_INPUT,
    HAS_XGBOOST,
    arima_forecaster,
    build_tabular_pipeline,
    evaluate_model,
    make_tabular_forecaster,
    prepare_target_frame,
    random_forest_factory,
    run_walk_forward,
    walk_forward_origins,
    xgboost_factory,
)

OUT_DIR = PROJECT_ROOT / "results" / "ablation"
WINDOWS = {"holdout_12m": 12, "estendida_60m": 60}
SELECTION_VAL_MONTHS = 24
N_REPEATS = 30
SEED = 42

GROUP_LABELS = {
    "preco": "Preço corrente",
    "cambio": "Câmbio",
    "brent": "Brent",
    "clima": "Clima",
    "sensoriamento": "Sensoriamento remoto",
    "calendario": "Calendário",
}

FAMILY_LABELS = {
    "preco_brl": "Preço corrente (R$)",
    "preco_usd": "Preço corrente (US$)",
    "cambio": "Câmbio",
    "brent": "Brent",
    "precipitacao": "Precipitação",
    "temperatura_media": "Temperatura média",
    "temperatura_max": "Temperatura máxima",
    "temperatura_min": "Temperatura mínima",
    "umidade_relativa": "Umidade relativa",
    "vento": "Vento",
    "ndvi": "NDVI",
    "umidade_solo": "Umidade do solo",
    "calendario": "Calendário",
}

MODEL_TITLES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def feature_group(name: str) -> str:
    """Grupo tematico (preco, cambio, brent, clima, sensoriamento, calendario)."""
    if name.startswith("soy_price_"):
        return "preco"
    if name.startswith("usd_brl"):
        return "cambio"
    if name.startswith("brent_usd_bbl"):
        return "brent"
    if name.startswith(("ndvi", "soil_moisture")):
        return "sensoriamento"
    if name in {"month", "quarter", "soy_planting_window", "soy_harvest_window"}:
        return "calendario"
    if name.startswith(("precipitation", "temperature", "relative_humidity", "wind_speed")):
        return "clima"
    raise ValueError(f"Feature sem grupo definido: {name}")


def feature_family(name: str) -> str:
    """Variavel de origem: a variavel bruta + seus lags, media movel e anomalia."""
    prefixes = [
        ("soy_price_brl_bag", "preco_brl"),
        ("soy_price_usd_bag", "preco_usd"),
        ("usd_brl", "cambio"),
        ("brent_usd_bbl", "brent"),
        ("precipitation_mm", "precipitacao"),
        ("temperature_mean_c", "temperatura_media"),
        ("temperature_max_c", "temperatura_max"),
        ("temperature_min_c", "temperatura_min"),
        ("relative_humidity_pct", "umidade_relativa"),
        ("wind_speed_ms", "vento"),
        ("ndvi", "ndvi"),
        ("soil_moisture_m3m3", "umidade_solo"),
    ]
    for prefix, fam in prefixes:
        if name.startswith(prefix):
            return fam
    if name in {"month", "quarter", "soy_planting_window", "soy_harvest_window"}:
        return "calendario"
    raise ValueError(f"Feature sem variavel de origem definida: {name}")


def families_of(features: list[str]) -> dict[str, list[str]]:
    fams: dict[str, list[str]] = {}
    for f in features:
        fams.setdefault(feature_family(f), []).append(f)
    return fams


def select_by_rule(importance: pd.DataFrame) -> pd.Series:
    """Criterio a priori: media(ΔMAE) - 2*desvio(ΔMAE) > 0."""
    return (importance["delta_mae_mean"] - 2 * importance["delta_mae_std"]) > 0


def permutation_importance_by_family(model_factory, train: pd.DataFrame, val: pd.DataFrame,
                                     features: list[str], n_repeats: int = N_REPEATS,
                                     seed: int = SEED) -> pd.DataFrame:
    pipe = build_tabular_pipeline(model_factory())
    pipe.fit(train[features], train["target_value"].astype(float).to_numpy())
    y = val["target_value"].astype(float).to_numpy()
    base_mae = float(np.mean(np.abs(y - pipe.predict(val[features]))))
    rng = np.random.default_rng(seed)
    rows = []
    for fam, cols in families_of(features).items():
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(val))
            X = val[features].copy()
            X[cols] = val[cols].to_numpy()[perm]  # mesma permutacao p/ toda a familia
            deltas.append(float(np.mean(np.abs(y - pipe.predict(X)))) - base_mae)
        rows.append({"family": fam, "n_features": len(cols), "base_mae": base_mae,
                     "delta_mae_mean": float(np.mean(deltas)),
                     "delta_mae_std": float(np.std(deltas, ddof=1))})
    return pd.DataFrame(rows)


def build_configs(features: list[str], relevant_families: list[str]) -> dict[str, list[str]]:
    fams = families_of(features)
    relevant = [f for f in features if feature_family(f) in relevant_families]
    configs = {"todas_54": list(features), "relevantes": relevant}
    for fam in relevant_families:
        configs[f"sem_{fam}"] = [f for f in relevant if f not in set(fams[fam])]
    return configs


def naive_predictions(target_df: pd.DataFrame, origins: list[int]) -> np.ndarray:
    """Persistencia: previsao de target_month[i] = preco observado em date[i]."""
    return target_df.iloc[origins]["soy_price_brl_bag"].astype(float).to_numpy()


# ----------------------------------------------------------------------- graficos
def plot_importance(imp: pd.DataFrame, out_path: Path) -> None:
    models = [m for m in MODEL_TITLES if m in set(imp["model"])]
    order = list(FAMILY_LABELS)
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(order))
    for ax, model in zip(axes, models):
        sub = imp[imp["model"] == model].set_index("family").reindex(order)
        colors = ["#2563eb" if s else "#9ca3af" for s in sub["selected"]]
        ax.barh(y, sub["delta_mae_mean"], xerr=2 * sub["delta_mae_std"], color=colors,
                error_kw=dict(ecolor="#374151", capsize=2, linewidth=0.8))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([FAMILY_LABELS[f] for f in order])
        ax.invert_yaxis()
        ax.set_title(MODEL_TITLES[model])
        ax.set_xlabel("Aumento do MAE ao permutar a variável (R$/saca)")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2563eb", label="Relevante (média − 2·dp > 0)"),
                        Patch(color="#9ca3af", label="Não relevante")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Importância por permutação — validação abr/2019 a mar/2021 "
                 "(barras de erro: ±2 desvios-padrão)", fontsize=10.5)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


X_CLIP = 60.0  # % — barras alem disso sao cortadas e anotadas


def plot_delta(metrics: pd.DataFrame, out_path: Path) -> None:
    models = [m for m in MODEL_TITLES if m in set(metrics["model"])]
    win_style = {"holdout_12m": ("#2563eb", "Holdout de 12 meses"),
                 "estendida_60m": ("#f59e0b", "Janela estendida de 60 meses")}
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.0))
    axes = np.atleast_1d(axes)
    h = 0.38
    for ax, model in zip(axes, models):
        sub_m = metrics[metrics["model"] == model]
        order = ["todas_54"] + [c for c in sub_m["config"].unique() if c.startswith("sem_")]
        labels = ["todas as 54 variáveis"] + [
            f"sem {FAMILY_LABELS[c[4:]].lower().replace('(r$)', '(R$)').replace('(us$)', '(US$)')}"
            .replace("sem ndvi", "sem NDVI")
            for c in order[1:]]
        y = np.arange(len(order))
        for k, (win, (color, wlabel)) in enumerate(win_style.items()):
            sub = sub_m[sub_m["window"] == win].set_index("config")
            delta = np.array([sub.loc[c, "delta_mae_pct"] for c in order], dtype=float)
            ypos = y + (k - 0.5) * h
            ax.barh(ypos, np.clip(delta, -X_CLIP, X_CLIP), height=h, color=color, label=wlabel)
            for yy, d in zip(ypos, delta):
                if abs(d) > X_CLIP:
                    ax.text(np.sign(d) * X_CLIP * 0.97, yy, f"{d:+.0f}% ", va="center",
                            ha="right" if d > 0 else "left", fontsize=8, color="white",
                            fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-X_CLIP / 2, X_CLIP)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(MODEL_TITLES[model])
        ax.set_xlabel("Variação do MAE vs. modelo só com as relevantes (%)")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Ablação sobre as variáveis relevantes — walk-forward expansivo de 1 passo "
                 "(valores > 0: remover piora o modelo)", fontsize=10.5)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def replot_from_csv() -> None:
    imp = pd.read_csv(OUT_DIR / "importance.csv")
    plot_importance(imp, OUT_DIR / "importance_permutation.png")
    metrics = pd.read_csv(OUT_DIR / "ablation_metrics.csv")
    plot_delta(metrics[metrics["config"] != "referencia"], OUT_DIR / "ablation_delta_mae.png")


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DEFAULT_INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df["target_month"] = pd.to_datetime(df["target_month"])
    target_df, features = prepare_target_frame(df)
    n = len(target_df)

    tab_models = {"random_forest": random_forest_factory}
    if HAS_XGBOOST:
        tab_models["xgboost"] = xgboost_factory

    # ---------------- etapa 1: selecao (dados anteriores a todas as origens avaliadas)
    first_eval = n - max(WINDOWS.values())
    val_start = first_eval - SELECTION_VAL_MONTHS
    train_sel = target_df.iloc[:val_start]
    val_sel = target_df.iloc[val_start:first_eval]
    print(f"Selecao: treino date {train_sel['date'].iloc[0].date()} -> {train_sel['date'].iloc[-1].date()} "
          f"({len(train_sel)}) | validacao date {val_sel['date'].iloc[0].date()} -> "
          f"{val_sel['date'].iloc[-1].date()} ({len(val_sel)})")
    imps = []
    for model, factory in tab_models.items():
        imp = permutation_importance_by_family(factory, train_sel, val_sel, features)
        imp.insert(0, "model", model)
        imp["selected"] = select_by_rule(imp)
        imps.append(imp)
        print(f"\n  {model}: MAE base na validacao = {imp['base_mae'].iloc[0]:.3f}")
        print(imp.sort_values("delta_mae_mean", ascending=False)[
            ["family", "n_features", "delta_mae_mean", "delta_mae_std", "selected"]
        ].round(4).to_string(index=False))
    importance = pd.concat(imps, ignore_index=True)
    importance.to_csv(OUT_DIR / "importance.csv", index=False)
    relevant = {m: importance[(importance["model"] == m) & importance["selected"]]
                .sort_values("delta_mae_mean", ascending=False)["family"].tolist()
                for m in tab_models}

    # ---------------- etapa 2: ablacao sobre as relevantes
    rows: list[dict] = []
    for win, w in WINDOWS.items():
        origins = walk_forward_origins(n, w)
        test = target_df.iloc[origins]
        y = test["target_value"].astype(float).to_numpy()
        print(f"\n=== Janela {win}: {w} origens | alvo {test['target_month'].iloc[0].date()} -> "
              f"{test['target_month'].iloc[-1].date()} ===")
        refs = {"persistencia": naive_predictions(target_df, origins)}
        refs["arima_1_1_0"], _ = run_walk_forward("arima_1_1_0", target_df, features, origins, arima_forecaster)
        for model, pred in refs.items():
            rows.append({"window": win, "model": model, "config": "referencia", "n_features": 0,
                         **{k: v for k, v in evaluate_model(model, y, pred).items() if k != "model"}})
        for model, factory in tab_models.items():
            forecaster = make_tabular_forecaster(factory)
            for config, cols in build_configs(features, relevant[model]).items():
                pred, _ = run_walk_forward(model, target_df, cols, origins, forecaster)
                m = evaluate_model(model, y, pred)
                rows.append({"window": win, "model": model, "config": config, "n_features": len(cols),
                             **{k: v for k, v in m.items() if k != "model"}})
                print(f"  {model:<14} {config:<24} ({len(cols):>2}) MAE {m['mae']:.4f}")

    metrics = pd.DataFrame(rows)
    base = (metrics[metrics["config"] == "relevantes"]
            .set_index(["window", "model"])["mae"].rename("mae_relevantes"))
    metrics = metrics.join(base, on=["window", "model"])
    metrics["delta_mae"] = metrics["mae"] - metrics["mae_relevantes"]
    metrics["delta_mae_pct"] = 100 * metrics["delta_mae"] / metrics["mae_relevantes"]
    metrics.to_csv(OUT_DIR / "ablation_metrics.csv", index=False)
    (OUT_DIR / "ablation_metrics.json").write_text(
        json.dumps({"relevant_families": relevant,
                    "selection": {"train_dates": [str(train_sel["date"].iloc[0].date()),
                                                  str(train_sel["date"].iloc[-1].date())],
                                  "validation_dates": [str(val_sel["date"].iloc[0].date()),
                                                       str(val_sel["date"].iloc[-1].date())],
                                  "n_repeats": N_REPEATS, "seed": SEED,
                                  "rule": "mean(delta_mae) - 2*std(delta_mae) > 0"},
                    "metrics": metrics.to_dict(orient="records")},
                   indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    replot_from_csv()
    print("\nFamilias relevantes:", relevant)
    show = metrics[["window", "model", "config", "n_features", "mae", "delta_mae_pct"]].copy()
    print("\n" + show.round(3).to_string(index=False))
    print(f"\nArquivos salvos em: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_from_csv()
    else:
        main()
