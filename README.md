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

> **Os dados de entrada são versionados** em `data/raw/` (CSVs + a planilha
> `CEPEA_20260407190923.xls`, hash SHA-256 registrado em
> [`docs/data_sources.md`](docs/data_sources.md)) — é o que permite `git clone`
> → `build_dataset.py --offline` funcionar sem rede nem credenciais.
> `data/interim/` e `data/processed/` são **gerados** pelo pipeline e não são
> versionados (regenerados a cada execução).

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

**Python 3.12** (o ambiente original de desenvolvimento do TGSI I era 3.14.3;
3.12 é o recomendado por estabilidade dos pacotes científicos — ver
`docs/methodology.md` §7.2).

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .[dev]                # pacote tgsi_pipeline + pytest

# 1) construir a base (data/raw/ já vem com o repositório)
PYTHONPATH=src python scripts/build_dataset.py

# 2) rodar todos os experimentos
PYTHONPATH=src python scripts/run_experiments.py

# 3) testes (inclui anti-vazamento e pareamento date/target_month)
PYTHONPATH=src python -m pytest tests/ -v
```

Resultados (figuras, métricas, previsões) vão para `results/<experimento>/`.
`build_dataset.py` imprime, ao final, o número de linhas, o intervalo
temporal, o número de colunas, o percentual de ausentes e o hash SHA-256 do
CSV gerado — para conferência independente.

## Estrutura

```
tgsi-ii/
├── configs/           configuração do pipeline (janela, fontes, alvo)
├── src/tgsi_pipeline/  pacote preservado do TGSI I: aquisição, integração,
│                       features, alvo (+ motor de baselines em modeling.py)
├── scripts/            build_dataset.py (constrói a base) · run_experiments.py
├── experiments/        holdout_12m · arima_residuals · arima_sensitivity ·
│                       cross_correlation · shock_period_analysis
├── data/
│   ├── raw/            VERSIONADO — dados de entrada (CSVs + planilha CEPEA)
│   ├── interim/        gerado, não versionado
│   └── processed/      gerado, não versionado (inclui features_monthly_modeling.csv)
├── results/            saídas dos experimentos    (conteúdo não versionado)
├── tests/               inclui test_no_leakage.py e test_target_pairing.py
└── docs/               methodology.md · data_sources.md · checkpoint_history.md
```

## Reprodutibilidade

1. **Base:** `scripts/build_dataset.py --offline` (padrão) reconstrói
   `features_monthly_modeling.csv` a partir de `data/raw/`, que **é
   versionado** — funciona logo após `git clone`, sem rede nem credenciais.
   `--online` refaz a coleta desde as fontes originais (precisa de rede e,
   para NDVI/umidade do solo, credenciais `EARTHDATA_USERNAME`/`PASSWORD`).
2. **Dependências:** `requirements.txt` fixa as versões **efetivamente usadas**
   no ambiente do TGSI I. `pytest` é dependência de desenvolvimento
   (`pip install -e .[dev]`).
3. **Anti-vazamento e pareamento temporal:** `pytest tests/ -v` —
   `test_no_leakage.py` garante que as anomalias não usam informação futura;
   `test_target_pairing.py` garante que `target_month = date + 1 mês` mesmo
   com lacunas na série.
4. **Registro da base final:** ver a saída de `build_dataset.py` (linhas,
   período, colunas, % ausentes, hash SHA-256).

## Correções aplicadas

### Checkpoint do professor (27/08/2026)

| # | Correção | Onde |
|---|---|---|
| 1 | Anomalias mensais **causais** (normal só com passado) — elimina look-ahead | `src/tgsi_pipeline/pipeline.py`, `tests/test_no_leakage.py`, `docs/methodology.md` §2 |
| 2 | Removida a modelagem em USD e a conversão USD→BRL com câmbio realizado; alvo oficial é **só BRL** | `experiments/holdout_12m.py`, `docs/methodology.md` §1.1 |
| 3 | Período e "observações perdidas" (mar/2006, abr/2026) documentados | `docs/methodology.md` §4 |
| 4 | Origem/transformação de cada fonte consolidada | `docs/data_sources.md` |
| 5 | `requirements.txt` com versões explícitas | `requirements.txt`, `pyproject.toml` |
| 6 | Base reproduzível a partir dos dados de entrada (modo offline) | `scripts/build_dataset.py` |

### Auditoria técnica (13/09/2026)

| # | Correção | Onde |
|---|---|---|
| 7–11 | `data/raw/` passou a ser **versionado** (inclui a planilha CEPEA); `docs/methodology.md` deixou de estar no `.gitignore`; `pytest` virou dependência de desenvolvimento declarada — um `git clone` novo agora roda `build_dataset.py --offline` e `run_experiments.py` de ponta a ponta | `.gitignore`, `pyproject.toml`, `data/raw/` |
| 12 | ARIMA no holdout de 12 meses passou a usar **walk-forward de 1 passo** (mesmo protocolo do Random Forest/XGBoost), em vez de um único `forecast(steps=12)` sem atualização | `experiments/holdout_12m.py`, `docs/methodology.md` §5 |
| 13 | Classificação de período (pré-choque/choque/pós-choque) passou a usar **`target_month`** (mês previsto), não `date` (mês de entrada) | `experiments/shock_period_analysis.py` |
| 14 | Gráfico do holdout passou a usar `target_month` no eixo, não `date` | `experiments/holdout_12m.py` |
| 15 | Conjunto de features padronizado entre `holdout_12m.py` e `shock_period_analysis.py` (mesma constante `METADATA_FIELDS` de `tgsi_pipeline.modeling`) | `experiments/holdout_12m.py`, `experiments/shock_period_analysis.py` |
| 16 | Pareamento do alvo `t → t+1` passou a usar `target_month = date + 1 mês` explicitamente, em vez de "próxima linha disponível" (protege contra lacunas na série) | `src/tgsi_pipeline/pipeline.py`, `tests/test_target_pairing.py` |
| 17 | Banda de confiança da correlação cruzada (CCF) passou a usar o N de pares válidos de cada variável, não o tamanho da base completa | `experiments/cross_correlation.py` |
