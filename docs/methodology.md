# Metodologia — TGSI II

Decisões metodológicas do projeto e correções feitas em relação ao TGSI I,
conforme o checkpoint.

---

## 1. Objetivo e alvo

Prever o **preço mensal da soja** no indicador **CEPEA/ESALQ – Paranaguá**,
**um mês à frente**:

```
features do mês t   ->   preço da soja (BRL) no mês t+1
```

- **Alvo:** `soy_price_brl_bag_next_month` — R$ por saca de 60 kg.
- **Unidade analítica:** mês.
- **Escopo geográfico:** indicador de praça de referência (Paranaguá); as
  variáveis climáticas e de sensoriamento remoto são de um ponto em
  Sorriso–MT, usado como proxy de região produtora.
- **Horizonte:** t+1 (um passo à frente).

### 1.1. Alvo em BRL e a questão do USD

A planilha bruta da CEPEA traz também `À vista US$` (`soy_price_usd_bag`). No
TGSI I havia:

1. uma segunda linha de modelagem direta em USD
   (`soy_price_usd_bag_next_month`); e
2. uma conversão em duas etapas *USD → BRL* usando o **câmbio realizado** do
   período de teste.

**Ambas foram removidas no TGSI II.** Motivos:

- o câmbio futuro **não é observável** no momento da previsão; converter com o
  câmbio realizado do holdout produz um número otimista que não representa
  desempenho operacional;
- o escopo científico do trabalho é a previsão do preço **em BRL**, e manter
  duas moedas com tratamentos diferentes enfraquece a defesa metodológica.

O que mudou no código:

- `experiments/holdout_12m.py` não usa mais `soy_price_usd_bag_next_month`,
  `build_usd_to_brl_view()` nem os gráficos USD. Roda só ARIMA / RF / XGBoost
  sobre o alvo BRL.
- `soy_price_usd_bag` **continua** no dataset como variável explicativa e para
  rastreabilidade — só deixou de ser um alvo.
- `soy_price_usd_bag_next_month` ainda é gerado em
  `features_monthly_ml_target.csv` (etapa intermediária), mas **não** entra em
  `features_monthly_modeling.csv`, cujo único `target_variable` é
  `soy_price_brl_bag_next_month`.

---

## 2. Anomalias e vazamento temporal (correção prioritária)

### 2.1. Definição de anomalia

Para uma variável `x` e um mês-calendário `m` (jan…dez):

```
anomaly_x(t) = x(t) − normal_x(m, t)
normal_x(m, t) = média de { x(s) : s < t  e  mês(s) = m }
```

Ou seja, a "normal" do mês é a média histórica **daquele mês-calendário
considerando apenas observações anteriores a t**.

### 2.2. Qual era o problema

No TGSI I a normal era calculada **sobre a série inteira** (função
`_compute_month_normals`, aplicada a todas as linhas de uma vez). Assim, a
anomalia de abril/2006 usava a média de **todos os abris de 2006 a 2026** —
inclusive futuro. Numa divisão temporal treino/teste (ou walk-forward), isso
injeta informação do futuro no conjunto de treino: **vazamento temporal
(look-ahead)**.

Verificação empírica: no dataset antigo, `usd_brl_month_anomaly` já vinha
preenchido na **primeira** linha (abr/2006) — impossível sem usar abris
posteriores.

### 2.3. O que foi corrigido

`src/tgsi_pipeline/pipeline.py` → `_causal_month_normal()`: a normal é
recalculada **a cada linha**, usando somente `previous_rows` (datas
estritamente anteriores). É exatamente o que um walk-forward de janela
expansiva calcularia em cada passo.

Consequências no dataset (`features_monthly_modeling.csv`):

- as **12 primeiras linhas** (abr/2006 – mar/2007) passam a ter anomalia
  `None` — não há mês-calendário igual no passado para formar a referência;
- para `soil_moisture_m3m3` (só a partir de ~2015) a primeira anomalia
  aparece em ~mar/2016;
- **todas as outras colunas do dataset permanecem idênticas** ao arquivo do
  TGSI I (verificado célula a célula: 0 diferenças fora das 6 colunas de
  anomalia).

### 2.4. Por que isso funciona na validação temporal / walk-forward

Como cada anomalia usa apenas o passado da própria linha, ela é válida em
qualquer esquema em que o treino precede o teste:

- **Holdout final:** os 12 meses de teste têm anomalias que só dependem de
  meses anteriores a eles — nenhum valor do teste "vaza" para trás.
- **Walk-forward / janela expansiva:** no passo `i`, a anomalia da linha `i`
  já foi calculada com dados até `i-1`; recomputar dentro do fold daria o
  mesmo valor. Não é preciso refazer a anomalia por fold.

Teste de regressão: `tests/test_no_leakage.py` — verifica que (a) a primeira
ocorrência de cada mês tem anomalia `None`, (b) a anomalia é a média dos meses
iguais estritamente anteriores, e (c) **acrescentar anos futuros à série não
altera nenhuma anomalia já calculada**.

### 2.5. Outras transformações que aprendem parâmetros

| Transformação | Onde | Respeita o tempo? |
|---|---|---|
| Lags 1/2/3, média móvel 3 | `_build_monthly_ml_features` | Sim — usam só `previous_rows` |
| Anomalia mensal | idem | Sim — corrigido (2.3) |
| Imputação de faltantes (mediana) | `SimpleImputer` **dentro** do `Pipeline` sklearn nos experimentos | Sim — `fit` só no treino / no `X[:i]` do fold |
| Padronização / normalização | Não usada nos modelos atuais | — (se for adicionada, deve ficar dentro do `Pipeline`, com `fit` só no treino) |
| Ordem do ARIMA | Fixa `(1,1,0)` | Não é ajustada nos dados de teste |

---

## 3. Validação temporal

Regra única: **passado → treino, futuro → teste**. Nunca o contrário.

- **Holdout final (`experiments/holdout_12m.py`):** os **últimos 12 meses** da
  série são o teste; todo o resto é treino. Os três modelos são avaliados no
  MESMO protocolo de 1 passo: RF e XGBoost preveem cada mês do teste a partir
  das features observadas naquele mês; o ARIMA é **reajustado a cada mês**
  (janela expansiva) e faz `forecast(steps=1)` — não um único
  `forecast(steps=12)` sem atualização (correção da auditoria de 13/09/2026,
  item 12 — ver §8.3).
- **Walk-forward 1-passo (`experiments/shock_period_analysis.py`):** treino
  mínimo de 24 meses; a cada mês, treina com tudo até `i-1` e prevê `i`;
  métricas recortadas por período (pré-choque < 2020 / choque 2020–2022 /
  pós-choque ≥ 2023).
- **Sensibilidade (`experiments/arima_sensitivity.py`):** holdout fixo de 24
  meses, variando a janela de treino (36…216 meses).

k-fold aleatório **não** é usado: quebraria a ordem temporal e vazaria futuro
para o treino.

---

## 4. Período da série e observações "perdidas"

| Item | Valor |
|---|---|
| Série CEPEA mensal crua (`target_soy_cepea_monthly.csv`) | **mar/2006 → abr/2026 = 242 meses** |
| mar/2006 | Mês **parcial** — a série diária da CEPEA começa em **13/03/2006** |
| Janela do config (`date_range.start`) | **2006-04-01** (fixo; a variante 2015+ usa 2015-04-01 — mesmo padrão "1º de abril") |
| Entrada do dataset final | abr/2006 → mar/2026 (**240 meses**) |
| Alvo (t+1) | mai/2006 → abr/2026 (**240 valores**) |
| **Total: 240 observações** | |

**Por que mar/2006 não aparece:** porque `date_range.start` no config é
`2006-04-01`. `build_base_rows` só cria linhas mensais a partir dessa data.
**Não** é efeito de lag nem de janela móvel — essas apenas produzem `NaN` nas
primeiras linhas, sem removê-las.

**Contabilidade dos 242 → 240:**
- −1: mar/2006, anterior ao início configurado;
- −1: abr/2026, que não tem mês t+1 para formar o alvo
  (`_build_monthly_modeling_dataset` descarta linhas sem
  `soy_price_brl_bag_next_month`).

Isso é coerente com previsão de um passo à frente.

**A verificar / decisão para o texto:** confirmar que a exclusão de mar/2006
foi deliberada por ser mês incompleto (recomendado: manter a exclusão e
declará-la explicitamente no TCC). O README do TGSI I mencionava
`2006-03-01` — desatualizado; os `configs` reais sempre usaram `2006-04-01`.

---

## 5. Modelos

O TGSI II trabalha com quatro modelos principais:

| Modelo | Papel | Onde |
|---|---|---|
| **ARIMA(1,1,0)** | Referência temporal univariada | `holdout_12m`, `shock_period`, `arima_residuals`, `arima_sensitivity` |
| **Random Forest** | Não-linear tabular com exógenas | `holdout_12m`, `shock_period` |
| **XGBoost** | Boosting tabular com exógenas | `holdout_12m`, `shock_period` |
| **LSTM** | Sequencial (a implementar) | Ver 5.1 |

Baseline de persistência (`naive_last_value`) entra como piso de comparação no
walk-forward. O motor `tgsi-model-baselines`
(`src/tgsi_pipeline/modeling.py`) mantém, além desses, um conjunto maior de
baselines (rolling mean, seasonal naive, ridge, SARIMAX, ensemble,
`lstm_window_12`) usado para comparação ampla — foi preservado do TGSI I.

Nesta fase: **sem** novos algoritmos, **sem** tuning agressivo de
hiperparâmetros, **sem** novas features, **sem** mudança de metodologia.

### 5.1. LSTM — situação atual

Ainda **não** há um LSTM real no TGSI II. O slot está reservado:
`modeling.predict_lstm_window` implementa hoje um MLP sequencial (janela de 12
meses do alvo + exógenas), usando só `scikit-learn` — sem dependência de
TensorFlow/PyTorch. A troca por um LSTM de fato (Keras ou PyTorch) é um passo
posterior; a interface de dados (janela deslizante respeitando a ordem
temporal) já está pronta.

---

## 6. Métricas

Os experimentos reportam **MAE**, **RMSE** e **MAPE**; o holdout adiciona
**R²**. As definições não foram alteradas em relação ao TGSI I. A prioridade
desta etapa é a **infraestrutura** para repetir e comparar os experimentos de
forma consistente, não mexer nas métricas.

---

## 7. Reprodutibilidade

### 7.1. Reconstrução da base

`scripts/build_dataset.py` reconstrói `features_monthly_modeling.csv` a partir
dos dados de entrada:

- **`--offline` (padrão):** lê `data/raw/*.csv` e roda só as transformações
  (integração → agregação mensal → features → anomalias causais → alvo).
  Não acessa a rede, não precisa de credenciais. É o caminho para quem recebe
  o código + os arquivos de entrada.
- **`--online`:** re-baixa tudo das fontes originais (BCB, EIA, NASA POWER,
  INMET, AppEEARS). Exige rede e, para o sensoriamento remoto, as variáveis de
  ambiente `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` (e horas de espera no
  AppEEARS).

Verificação: o modo offline reproduz o `features_monthly_modeling.csv` do
TGSI I com **0 diferenças** fora das 6 colunas de anomalia (que mudaram de
propósito — item 2).

### 7.2. Versões de dependências

`requirements.txt` e `pyproject.toml` fixam as versões **efetivamente usadas
no ambiente do TGSI I** (virtualenv `.venv` do repositório `pipeline_tgsi`,
Python 3.14.3):

```
numpy 2.4.4 · pandas 3.0.2 · scipy 1.17.1 · scikit-learn 1.8.0
statsmodels 0.14.6 · matplotlib 3.10.9 · xgboost 3.2.0 · xlrd 2.0.2
```

Nenhuma versão foi inventada. A construção da base (`build_dataset.py`
offline) precisa apenas da biblioteca padrão + `xlrd`; `pandas`, `numpy`,
`scikit-learn`, `statsmodels`, `matplotlib` e `xgboost` são necessários só
para os experimentos em `experiments/`.

**A verificar / risco:** o `.venv` original (Python 3.14) apresentou
lentidão/trava ao importar `scikit-learn` nesta máquina. Recomenda-se recriar
o ambiente com **Python 3.12** e `pip install -r requirements.txt`. Se alguma
versão não tiver wheel para o Python escolhido, ajustar para a versão estável
mais próxima e **registrar a mudança aqui**. A auditoria de 13/09/2026
instalou com sucesso em **Python 3.13.13**.

### 7.3. Dependência de desenvolvimento (`pytest`)

`pytest` não é dependência de execução do pipeline — é usada só para rodar
`tests/`. Está declarada em `pyproject.toml` →
`[project.optional-dependencies].dev` (versão `9.1.1`, a mesma usada na
auditoria). Instalar com `pip install -e .[dev]`.

---

## 8. Auditoria técnica de 13/09/2026 — correções aplicadas

Um clone novo do repositório, em ambiente virtual novo, não conseguia
reconstruir a base nem rodar os experimentos, e a leitura do código revelou
inconsistências metodológicas. Abaixo, cada item da auditoria e a correção.

### 8.1. Cadeia de reprodutibilidade quebrada (itens 4, 7–11)

**Causa raiz:** `data/raw/` estava no `.gitignore`; um clone novo não recebia
os CSVs de entrada, então nem `build_dataset.py --offline` (faltavam os
arquivos) nem `--online` (faltava a planilha CEPEA, que também estava
ignorada via `*.xls`) conseguiam gerar `features_monthly_modeling.csv` —
e sem essa base, `run_experiments.py` não tinha o que rodar.

**Correção:** `data/raw/` passou a ser **versionado** (é a fonte da verdade
para reprodução; ver proveniência e hash em `docs/data_sources.md`).
`data/interim/` e `data/processed/` continuam gerados/ignorados — inclusive
os próprios CSVs de saída do pipeline, que não fazem mais sentido ficar
"pré-prontos" no repositório agora que podem ser reconstruídos. `docs/methodology.md`
(este arquivo) também estava incorretamente listado no `.gitignore` e passou
a ser versionado.

**Consequência da correção de vazamento (§2) sobre os resultados:** ao
corrigir a fuga de informação futura nas anomalias, os modelos de aprendizado
de máquina (RF/XGBoost) deixaram de usar uma vantagem artificial. Isso é
esperado e é exatamente o efeito que a correção deveria ter — não é sinal de
regressão.

### 8.2. `date` vs. `target_month` (itens 13 e 14)

Convenção da base (não mudou, só a *leitura* que o código fazia dela):

```
date         = t     (mes das features observadas)
target_month = t + 1 (mes ao qual o valor previsto/avaliado pertence)
```

Exemplo: `date=2019-12-01`, `target_month=2020-01-01` → previsão do preço de
janeiro/2020, usando dados de dezembro/2019.

**Problema encontrado:** `experiments/shock_period_analysis.py` indexava a
série e classificava pré-choque/choque/pós-choque por `date`, não por
`target_month` — deslocando as fronteiras da análise em um mês (o preço
avaliado como "de janeiro/2020" era na verdade rotulado pelo mês de entrada,
dezembro/2019). O mesmo deslocamento aparecia no eixo X do gráfico de
`experiments/holdout_12m.py`.

**Correção:** `shock_period_analysis.py` agora indexa a série alvo, os
walk-forwards e a classificação de período por `target_month`; a coluna
temporal do `shock_period_walkforward.csv` passou a se chamar `target_month`
(antes `date`). `holdout_12m.py` usa `target_month` no eixo X do gráfico de
previsões. `SHOCK_START`/`SHOCK_END` (2020-01-01 / 2022-12-01) agora são
lidos como fronteiras sobre o mês previsto, que é a leitura correta.

### 8.3. ARIMA fora do protocolo de 1 passo no holdout (item 12)

**Problema:** no holdout de 12 meses, RF e XGBoost já operavam em 1 passo
(recebem features de `t`, preveem `t+1`, uma vez por mês de teste — 12
previsões `X_t → P_{t+1}`). O ARIMA era ajustado **uma única vez** antes do
holdout e chamava `forecast(steps=12)`: uma previsão de 12 passos à frente,
sem atualizar o modelo com os meses reais que iam se revelando. As duas
tarefas não são equivalentes, e a diferença podia por si só explicar parte do
resultado comparativo.

**Correção:** `predict_arima_walkforward()` em `holdout_12m.py` reajusta o
ARIMA a cada um dos 12 meses de teste, usando todo o histórico disponível até
ali (janela expansiva), e prevê só o próximo mês — o mesmo protocolo de
`shock_period_analysis.py`. Os três modelos agora são avaliados no mesmo
horizonte de 1 passo.

### 8.4. Features diferentes entre `holdout_12m` e `shock_period_analysis` (item 15)

**Problema:** `holdout_12m.py` incluía `soy_price_brl_bag`/`soy_price_usd_bag`
(o preço observado no mês corrente) como feature; `shock_period_analysis.py`
excluía essas duas colunas explicitamente. "Random Forest" e "XGBoost" não
eram, portanto, a mesma especificação de modelo nos dois experimentos.

**Correção:** os dois scripts agora importam a mesma constante,
`tgsi_pipeline.modeling.METADATA_FIELDS`, para decidir o que é metadado
(excluído) vs. feature. A decisão adotada foi **manter** o preço corrente
como feature em ambos os experimentos — ele é informação legitimamente
disponível em `t` para prever `t+1` (não é vazamento), e é o que torna RF/XGB
competitivos com a persistência ingênua.

### 8.5. Pareamento `t → t+1` por posição, não por calendário (item 16)

**Problema:** `_attach_monthly_target()` pareava cada mês com o "próximo item
da lista ordenada" de meses-alvo disponíveis, não com `date + 1 mês`
explicitamente. Se a série CEPEA tivesse uma lacuna (um mês sem cotação),
esse pareamento por posição associaria incorretamente um mês ao alvo de um
mês mais distante.

**Correção:** o pareamento agora usa `_next_month_start(date)` (aritmética de
calendário) e só preenche `soy_price_brl_bag_next_month` se esse mês exato
existir na série de alvo — senão fica `None`. Na série real usada
(`data/raw/target_soy_cepea_monthly.csv`, sem lacunas, mar/2006–abr/2026), o
resultado é idêntico ao anterior; a correção é uma proteção estrutural, não
uma mudança nos 240 valores atuais. Regressão em
`tests/test_target_pairing.py`, incluindo um caso sintético com lacuna.

### 8.6. Banda de confiança da CCF com N incorreto (item 17)

**Problema:** em `experiments/cross_correlation.py`, a banda `±1.96/√N` usava
o tamanho da base completa (`len(df)`) para todas as variáveis, mesmo para
`soil_moisture_m3m3`, que só tem histórico a partir de ~2015 (N efetivo bem
menor).

**Correção:** `N` passou a ser o número de pares válidos *daquela variável*
(`len(valid)`, já calculado por feature). O script agora também imprime um
aviso explícito: em série temporal autocorrelacionada (o caso desta série
mensal), essa banda é uma referência visual, não um teste de significância
estatística formal — não deve ser citada como evidência conclusiva sem
ressalva.

### 8.7. Registro da base final (Passo 6 da auditoria)

`scripts/build_dataset.py` agora imprime e grava, ao final,
`data/processed/dataset_manifest.json` com: número de linhas, número de
colunas, intervalo de `date`, intervalo de `target_month`, `target_variable`,
percentual de ausentes por coluna e o hash SHA-256 do
`features_monthly_modeling.csv` gerado — para conferência independente sem
precisar reexecutar o pipeline.
