# TGSI II — Previsão do preço mensal da soja (CEPEA/ESALQ)

Trabalho de Graduação em Sistemas de Informação II — UFSM.

---

## Objetivo

Prever o **preço mensal da soja** no indicador **CEPEA/ESALQ – Paranaguá**,
**um mês à frente**:

```
features do mês t   →   preço da soja (R$/saca) no mês t+1
```

Alvo: `soy_price_brl_bag_next_month` (R$ por saca de 60 kg). Horizonte t+1.
Unidade analítica: mês. Escopo geográfico: indicador de praça de referência.

## Dados

| Grupo | Fonte | Variáveis |
|---|---|---|
| Alvo | CEPEA/ESALQ – Paranaguá (planilha `.xls`) | `soy_price_brl_bag`, `soy_price_usd_bag` |
| Câmbio | Banco Central do Brasil — PTAX | `usd_brl` |
| Petróleo | EIA (Brent RBRTE) · fallback World Bank Pink Sheet | `brent_usd_bbl` |
| Clima | INMET (estação A904) · fallback NASA POWER | precipitação, temperatura, umidade, vento |
| NDVI | NASA MODIS `MOD13Q1.061` via AppEEARS | `ndvi` |
| Umidade do solo | NASA SMAP `SPL3SMP_E.006` via AppEEARS (desde 2015) | `soil_moisture_m3m3` |

Ponto geográfico: Sorriso–MT (−12,5425 / −55,7211).
Janela: **abr/2006 → abr/2026**; base final com **240 observações mensais**.
Detalhes de origem, frequência, transformação e unidade de cada variável:
**[`docs/data_sources.md`](docs/data_sources.md)**.

> **Dados não são versionados** (ver `.gitignore`). Os arquivos de entrada
> (`data/raw/*.csv` e `data/raw/CEPEA_*.xls`) são recebidos junto com o código;
> a base processada é reconstruída a partir deles.

## Pipeline — como a base é construída

```
data/raw/  (CSVs de entrada + CEPEA_*.xls)
    │  leitura + coerção de tipos
    ▼
integração das fontes  →  agregação mensal          (tgsi_pipeline._build_monthly_features)
    ▼
features derivadas: month/quarter, janelas de safra,
lags 1–3, média móvel 3, anomalias mensais CAUSAIS   (tgsi_pipeline._build_monthly_ml_features)
    ▼
anexo do alvo  (t → t+1)                             (tgsi_pipeline._attach_monthly_target)
    ▼
seleção de colunas de modelagem                      (tgsi_pipeline._build_monthly_modeling_dataset)
    ▼
data/processed/features_monthly_modeling.csv
```

Comando:

```bash
PYTHONPATH=src python scripts/build_dataset.py            # offline: só data/raw/ → data/processed/
PYTHONPATH=src python scripts/build_dataset.py --online   # re-baixa das fontes (rede + credenciais Earthdata)
```

O modo offline reproduz o dataset do TGSI I com **0 diferenças** fora das
colunas de anomalia (que foram corrigidas — ver abaixo).

## Variáveis

- **Alvo:** `soy_price_brl_bag_next_month` (R$/saca, t+1).
- **Exógenas (nível):** `usd_brl`, `brent_usd_bbl`, clima, `ndvi`,
  `soil_moisture_m3m3`.
- **Derivadas:** `<var>_lag_1/2/3`, `<var>_rolling_mean_3` (só passado);
  `<var>_month_anomaly` (nível − normal histórica **do mesmo mês, só com
  passado** — causal, sem look-ahead); `month`, `quarter`,
  `soy_planting_window`, `soy_harvest_window`.
- `soy_price_usd_bag` permanece como variável explicativa / rastreabilidade —
  **não** é um alvo (ver `docs/methodology.md`).

## Validação

- **Holdout final:** últimos **12 meses** como teste; o restante é treino.
- **Walk-forward 1-passo:** treino mínimo 24 meses; a cada mês treina com o
  passado e prevê o mês seguinte (usado na análise do período de choque
  2020–2022).
- **Sensibilidade:** holdout fixo de 24 meses, variando a janela de treino.

Regra: **passado → treino, futuro → teste**. Sem k-fold aleatório.
Transformações que aprendem parâmetros (imputação, anomalias) respeitam a
ordem temporal. Detalhes: **[`docs/methodology.md`](docs/methodology.md)**.

## Modelos

| Modelo | Situação |
|---|---|
| ARIMA(1,1,0) | ativo |
| Random Forest | ativo |
| XGBoost | ativo |
| LSTM | **slot reservado** — hoje um MLP sequencial (só scikit-learn); troca por LSTM real (Keras/PyTorch) é passo posterior |

Baseline de persistência (`naive_last_value`) como piso de comparação.
Nesta fase: sem novos algoritmos, sem tuning agressivo, sem novas features.

## Execução

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # pacote tgsi_pipeline

# 1) construir a base (precisa de data/raw/ populado)
PYTHONPATH=src python scripts/build_dataset.py

# 2) rodar todos os experimentos
PYTHONPATH=src python scripts/run_experiments.py
```

Resultados (figuras, métricas, previsões) vão para `results/<experimento>/`.

## Estrutura

```
tgsi-ii/
├── configs/           configuração do pipeline (janela, fontes, alvo)
├── src/tgsi_pipeline/  pacote preservado do TGSI I: aquisição, integração,
│                       features, alvo (+ motor de baselines em modeling.py)
├── scripts/            build_dataset.py (constrói a base) · run_experiments.py
├── experiments/        holdout_12m · arima_residuals · arima_sensitivity ·
│                       cross_correlation · shock_period_analysis
├── data/               raw/ interim/ processed/   (conteúdo não versionado)
├── results/            saídas dos experimentos    (conteúdo não versionado)
├── tests/              inclui test_no_leakage.py (regressão anti-vazamento)
└── docs/               methodology.md · data_sources.md · checkpoint_history.md
```

## Reprodutibilidade

1. **Base:** `scripts/build_dataset.py --offline` reconstrói
   `features_monthly_modeling.csv` a partir de `data/raw/` — sem rede, sem
   credenciais. `--online` refaz a coleta desde as fontes.
2. **Dependências:** `requirements.txt` fixa as versões **efetivamente usadas**
   no ambiente do TGSI I (Python 3.14.3). Ver nota de ambiente em
   `docs/methodology.md` §7.2.
3. **Anti-vazamento:** `python -m pytest tests/` (ou `unittest`) —
   `test_no_leakage.py` garante que as anomalias não usam informação futura.

## Correções do checkpoint aplicadas

| # | Correção | Onde |
|---|---|---|
| 1 | Anomalias mensais agora **causais** (normal só com passado) — elimina look-ahead | `src/tgsi_pipeline/pipeline.py`, `tests/test_no_leakage.py`, `docs/methodology.md` §2 |
| 2 | Removida a modelagem em USD e a conversão USD→BRL com câmbio realizado; alvo oficial é **só BRL** | `experiments/holdout_12m.py`, `docs/methodology.md` §1.1 |
| 3 | Período e "observações perdidas" (mar/2006, abr/2026) documentados e explicados | `docs/methodology.md` §4 |
| 4 | Origem/transformação de cada fonte consolidada | `docs/data_sources.md` |
| 5 | `requirements.txt` com versões explícitas | `requirements.txt`, `pyproject.toml` |
| 6 | Base reproduzível a partir dos dados de entrada (modo offline) | `scripts/build_dataset.py` |
