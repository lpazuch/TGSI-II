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

- **Holdout final (`experiments/holdout_12m.py`):** os **últimos 12 meses**
  da série são as 12 **origens de previsão**. ARIMA(1,1,0), Random Forest e
  XGBoost seguem **o mesmo protocolo — walk-forward expansivo de 1 passo**:
  ```
  para cada origem i (as 12 últimas linhas, date = abr/2025 … mar/2026):
      treina do zero com as linhas [0, i)    (janela expansiva, desde abr/2006)
      prevê a linha i                         (features de date[i] → preço de target_month[i])
      incorpora a linha i e passa à origem seguinte
  ```
  - O treino da 1ª origem tem 228 linhas (date abr/2006 → mar/2025); o da
    12ª tem 239. A última linha do treino tem como alvo o preço de `date[i]`,
    que já é conhecido na origem — nenhum alvo posterior a `date[i]` entra.
  - Os três modelos passam pela **mesma função** (`run_walk_forward`) com a
    **mesma lista de origens** (`walk_forward_origins`); só muda o
    `forecaster`: o ARIMA usa a série univariada do alvo; RF e XGBoost usam
    um `Pipeline` novo (imputação pela mediana + modelo) ajustado só no
    treino daquela origem, com as 54 features.
  - Cada modelo mantém **seus próprios hiperparâmetros**, fixos ao longo das
    12 origens de previsão (sem nova busca ou ajuste de hiperparâmetros).
  - As origens efetivamente usadas por cada modelo ficam registradas em
    `results/holdout_12m/holdout_12m_origins.csv`, e
    `tests/test_holdout_protocol.py` garante que são as mesmas 12 para os
    três (ver §9).
- **Walk-forward 1-passo (`experiments/shock_period_analysis.py`):** treino
  mínimo de 24 meses; a cada mês, treina com tudo até `i-1` e prevê `i`;
  métricas recortadas por período (pré-choque < 2020 / choque 2020–2022 /
  pós-choque ≥ 2023).
- **Sensibilidade (`experiments/arima_sensitivity.py`):** holdout fixo de 24
  meses, variando a janela de treino (36…216 meses).
- **Ablação (`experiments/ablation.py`):** mesmo walk-forward expansivo do
  holdout, em 12 e em 60 origens; ver §10.
- **Diebold-Mariano (`experiments/diebold_mariano.py`):** significância das
  diferenças de erro entre pares de modelos, no holdout (12) e no walk-forward
  da análise de choque (216); ver §11.
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
**Correção:** o ARIMA passou a ser reajustado a cada um dos 12 meses de
teste, com todo o histórico disponível até ali (janela expansiva), prevendo só
o próximo mês. Isso uniformizou o **horizonte** (1 passo), mas **não** o
protocolo de atualização: RF e XGBoost continuavam treinados uma única vez
antes do holdout. A uniformização completa está em §9.
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

---
## 9. Protocolo de atualização uniforme no holdout (item 18 — 11/10/2026)

**Problema:** após o item 12, os três modelos previam com o mesmo horizonte
(1 passo), mas com protocolos diferentes:

| Modelo | Antes |
|---|---|
| ARIMA(1,1,0) | walk-forward de 1 passo, **reajustado a cada mês** |
| Random Forest / XGBoost | 1 passo, mas **treinados uma única vez** antes dos 12 meses de holdout |
Uma comparação independente anterior (mesma base, mesmas 54 features, mesmos
hiperparâmetros por modelo, mesmos 12 meses) mostrou que a diferença tinha
efeito numérico, sobretudo no XGBoost.
**Correção:** `experiments/holdout_12m.py` foi reestruturado para que os três
modelos usem **um único protocolo — walk-forward expansivo de 1 passo**
(§3): `walk_forward_origins()` define as 12 origens uma só vez e
`run_walk_forward()` percorre essas origens para cada modelo, reajustando-o
do zero com as linhas `[0, i)` e prevendo a linha `i`. Nenhuma outra coisa
mudou (base, features, hiperparâmetros, meses de teste, métricas).
**Teste:** `tests/test_holdout_protocol.py` verifica, com dados sintéticos,
que (a) as origens são as 12 últimas linhas; (b) cada modelo é chamado 12
vezes — um reajuste por origem — e recebe como treino exatamente
`iloc[:i]` e como teste exatamente `iloc[i]`; (c) nenhum alvo do treino é
posterior a `date[i]`; (d) o registro de origens é idêntico entre ARIMA, RF e
XGBoost; (e) com os forecasters reais, RF e XGBoost instanciam um modelo novo
em cada uma das 12 origens.
**Resultado** (base com SHA-256 `692fc3ce…5e89085`; Python 3.12.6, numpy
2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, statsmodels 0.14.6):

| Modelo | Protocolo | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| ARIMA(1,1,0) | walk-forward expansivo | **2,895523** | **4,017159** | **2,169427** | **0,367305** |
| Random Forest | walk-forward expansivo | 3,268081 | 4,667835 | 2,474877 | 0,145746 |
| XGBoost | walk-forward expansivo | 3,340820 | 4,844046 | 2,543865 | 0,080032 |
| Persistência ingênua (referência) | mesmas 12 origens | 2,958251 | 3,979518 | 2,219098 | 0,379107 |
Para referência, a versão anterior (RF/XGB treinados uma vez) dava RF
MAE 3,141028 / R² 0,204389 e XGBoost MAE 4,091165 / R² −0,738026. O ranking
(ARIMA < RF < XGBoost em MAE) não mudou; a conclusão de que o ARIMA tem o
menor erro neste holdout continua válida, agora sob protocolo idêntico.

**Conferência independente posterior (13/10/2026):** na comparação inicial,
mesmo após a correção do protocolo, foram observadas diferenças nos valores de
Random Forest e XGBoost em relação a uma execução independente. Uma nova
execução, a partir de um clone do repositório, reproduziu os resultados dos
três modelos apresentados na tabela acima. Também foram conferidas as mesmas
12 origens de previsão por modelo, registradas em
`results/holdout_12m/holdout_12m_origins.csv` (36 procedimentos no total).
Portanto, a divergência numérica daquela comparação inicial foi resolvida e
não permanece como pendência de reprodutibilidade computacional do holdout.

**Reexecução para o artigo final (09/11/2026):** a base foi reconstruída
(`build_dataset.py --offline`, mesmo SHA-256 `692fc3ce…5e89085`) e todos os
experimentos foram rodados de novo em um ambiente virtual novo (Python 3.12.6,
versões fixadas em `requirements.txt`). O ARIMA reproduziu exatamente os
valores anteriores; Random Forest e XGBoost deram valores ligeiramente
diferentes dos da tabela publicada antes (RF MAE 3,273438 → 3,268081; XGBoost
3,515465 → 3,340820). A tabela acima passou a registrar essa execução, que é a
mesma de onde saem as figuras do artigo final. A ordem dos modelos (ARIMA <
persistência < RF < XGBoost em MAE) e as conclusões não mudam. A causa da
variação entre execuções de RF/XGBoost não foi identificada; uma hipótese é o
paralelismo (`n_jobs=-1`) e diferenças de plataforma/biblioteca nativa — fica
como pendência de reprodutibilidade bit a bit desses dois modelos. A linha de
persistência ingênua (previsão de t+1 = preço observado em t, nas mesmas 12
origens) foi incluída como referência mínima de comparação.

Essa conferência não substitui a verificação científica da proveniência dos
dados, da disponibilidade temporal das variáveis explicativas e de eventuais
riscos de vazamento associados às fontes e datas efetivas de publicação.

---
## 10. Seleção de variáveis e ablação (12/11/2026)

**Objetivo:** identificar as variáveis **efetivamente relevantes** para Random
Forest e XGBoost e medir a contribuição de cada uma (ablação) — pendência
registrada desde o TGSI I.

**Etapa 1 — seleção por importância de permutação (`experiments/ablation.py`):**
- As 54 colunas são agrupadas em **13 variáveis de origem** (`feature_family`):
  cada exógena com seus lags, média móvel e anomalia (ex.: câmbio = 6 colunas),
  mais preço corrente R$, preço corrente US$, temperaturas máx./mín. e
  calendário. A família inteira é permutada com a **mesma** permutação de
  linhas, para não subestimar variáveis com lags muito correlacionados.
- Treino: `date` abr/2006 → mar/2019 (156 linhas); validação: abr/2019 →
  mar/2021 (24 linhas). Essa validação é **anterior a todas as origens
  avaliadas** na etapa 2 — a seleção não vê o teste.
- 30 permutações por família (semente 42); **critério a priori:** relevante se
  `média(ΔMAE) − 2·desvio(ΔMAE) > 0`.
- Resultado (`results/ablation/importance.csv`):
  - **Random Forest:** preço R$ (ΔMAE 0,752) e câmbio (0,043).
  - **XGBoost:** preço R$ (0,780) e calendário (0,008).
  - Nenhuma variável de clima ou sensoriamento remoto foi selecionada; para a
    maioria, permutar **reduz** o erro (temperatura mínima, umidade relativa,
    precipitação, NDVI) — sinal de ajuste a ruído.
  - MAE de base na validação ≈ 35 R$/saca: o período inclui a alta de
    2020–2021, fora da faixa de treino (as árvores não extrapolam). A seleção
    depende dessa janela.

**Etapa 2 — ablação (leave-one-variable-out sobre as relevantes):**
mesmo walk-forward expansivo de 1 passo do holdout (§3, §9; funções e
hiperparâmetros importados de `holdout_12m.py`), em **12 origens** (holdout
oficial) e **60 origens** (alvos mai/2021 → abr/2026). MAE (R$/saca):

| Modelo – configuração (nº colunas) | 12m | 60m |
|---|---|---|
| RF – todas (54) | 3,268 | 7,607 |
| RF – relevantes: preço R$ + câmbio (7) | 3,594 | 7,418 |
| RF – relevantes sem preço R$ (6) | 11,311 | 18,823 |
| RF – relevantes sem câmbio (1) | 3,979 | 6,168 |
| XGB – todas (54) | 3,341 | 9,607 |
| XGB – relevantes: preço R$ + calendário (5) | 3,142 | 5,963 |
| XGB – relevantes sem preço R$ (4) | 48,225 | 74,359 |
| XGB – relevantes sem calendário (1) | 3,542 | 6,276 |
| *ref.* ARIMA(1,1,0) | 2,896 | 4,450 |
| *ref.* persistência | 2,958 | 4,771 |

**Leitura:**
- O preço corrente em R$ domina: removê-lo aumenta o MAE em +154% a +1.435%.
- Câmbio (RF) e calendário (XGB) têm efeito pequeno e de sinal instável
  (sem câmbio: +11% em 12m, −17% em 60m).
- Restringir o XGBoost às relevantes melhora muito o modelo (60m: 9,607 →
  5,963); no RF o efeito é misto.
- Nenhuma configuração de árvore supera ARIMA ou persistência nas mesmas
  janelas.

**Ressalvas:** a seleção depende da janela de validação (que coincide com o
início do choque); hiperparâmetros fixos; RF/XGBoost têm pequena variação
numérica entre execuções (§9), irrelevante diante das magnitudes acima.

---
## 11. Teste de Diebold-Mariano (14/11/2026)

**Objetivo:** verificar se as diferenças de erro entre os modelos são
estatisticamente significativas (pendência registrada no TGSI I).

**Método (`experiments/diebold_mariano.py`):**
- `d_t = L(e_A,t) − L(e_B,t)`, perda **absoluta** (principal, coerente com o
  MAE) e quadrática; horizonte h = 1.
- Correção para amostras pequenas de Harvey, Leybourne e Newbold (1997):
  `DM* = DM·sqrt((n + 1 − 2h + h(h−1)/n)/n)`, p-valor bilateral pela t com
  n − 1 g.l. **DM* < 0 ⇒ o modelo A é mais preciso.**
- Robustez: a variância de longo prazo do **diferencial de perdas d_t** é
  reestimada por Newey-West (HAC, núcleo de Bartlett, defasagem
  L = ⌊4(n/100)^(2/9)⌋: L = 2 no holdout, 4 em 216 e no pré-choque, 3 no choque e
  no pós-choque). Ela substitui a variância no denominador do DM; mantêm-se a
  correção HLN e a t com n − 1 g.l. Em todas as amostras e pares, a decisão a 5%
  é a mesma com as duas variâncias (colunas `dm_hln_nw`, `p_value_nw`).
- As duas amostras são experimentos distintos: o holdout compara as
  configurações principais; nas 216 previsões, RF/XGBoost usam as configurações
  leves da análise de choque (ARIMA e persistência são idênticos nas duas).
- Nenhuma previsão nova: usa `results/holdout_12m/holdout_12m_predictions.csv`
  (persistência = preço observado em `date`) e
  `results/shock_period/shock_period_walkforward.csv` (216 previsões, alvos
  mai/2008 → abr/2026, com recortes pré-choque/choque/pós-choque).
- Saídas: `results/diebold_mariano/dm_results.csv` / `.json`.
- Testes: `tests/test_diebold_mariano.py`.

**Resultados (perda absoluta, DM*, p-valor):**

| Par (A vs. B) | Holdout 12 | Walk-forward 216 |
|---|---|---|
| ARIMA vs. persistência | −0,18 (0,863) | −1,63 (0,106) |
| ARIMA vs. RF | −0,36 (0,723) | −6,08 (< 0,001) |
| ARIMA vs. XGB | −0,57 (0,581) | −6,14 (< 0,001) |
| Persistência vs. RF | −0,37 (0,721) | −6,16 (< 0,001) |
| Persistência vs. XGB | −0,53 (0,605) | −6,02 (< 0,001) |
| RF vs. XGB | −0,12 (0,906) | −1,48 (0,141) |

**Leitura:**
- Holdout de 12 meses: nenhuma diferença significativa — a amostra não
  distingue os modelos.
- 216 meses: ARIMA e persistência são significativamente mais precisos que
  RF e XGBoost (robusto a Newey-West e à perda quadrática); ARIMA ×
  persistência e RF × XGBoost não diferem a 5%.
- Recortes: o padrão se mantém no pré-choque (n = 140) e no choque (n = 36);
  no pós-choque (n = 40) ARIMA supera a persistência com p = 0,047 (NW 0,025),
  que não resiste a Bonferroni (6 pares).

**Ressalva:** no walk-forward de 216 meses, RF e XGBoost usam as
configurações mais leves da análise de choque (RF 200 árvores/prof. 6; XGB
200/prof. 3/lr 0,05), não as do holdout.

---
## 12. Revisão final: escolha do ARIMA sem o holdout (16/11/2026)

**Problema apontado na revisão:** a comparação de especificações do ARIMA
treinava até abr/2024 e media o MAE em mai/2024 → abr/2026 — janela que
contém os 12 meses-alvo do holdout (mai/2025 → abr/2026). Além disso, a série
começava em mar/2006 (mês parcial, excluído da base de modelagem).

**Correção (`experiments/arima_residuals.py`, testada em
`tests/test_arima_windows.py`):** série a partir de abr/2006; treino
abr/2006 → abr/2023 (205 obs); validação mai/2023 → abr/2025 (24 meses, 1 passo,
reajuste mensal); o holdout não é usado. Saída nova: `spec_comparison.csv`.

| Especificação | AIC (treino) | MAE validação |
|---|---|---|
| ARIMA(1,1,0) | **1191,79** | 3,875 |
| ARIMA(0,1,1) | 1192,06 | 3,780 |
| ARIMA(1,1,1) | 1193,62 | 3,774 |
| ARIMA(2,1,0) | 1193,70 | 3,760 |
| ARIMA(2,1,1) | 1195,46 | 3,766 |
| persistência | – | 4,491 |

Diferenças ≤ 0,115 R$/saca (≈ 3%): ARIMA(1,1,0) mantido por parcimônia.

**Ljung-Box (correção de diagnóstico):** o primeiro resíduo de um modelo com
d = 1 é o próprio nível da série (≈ 27,8) e inflava a variância, mascarando a
autocorrelação. Sem ele, p-valores 0,092 / 0,084 / 0,011 / 0,013 (lags
6/12/18/24): ruído branco não rejeitado no curto prazo, rejeitado nos lags 18
e 24 — o mesmo vale para as quatro alternativas. A ACF mostra picos isolados
(≈ 0,2) nos lags 6 e 16. Registrado como limitação no artigo; não afeta a
avaliação fora da amostra.

**Outros ajustes da revisão:** preço corrente em US$ explicitado como uma das
54 variáveis (o que foi removido foi o *alvo* em US$ e a conversão pelo câmbio
realizado); Bonferroni explícito no DM (as 4 diferenças univariados × árvores
seguem significativas, p corrigido < 0,003 mesmo com Newey-West); referências
institucionais das fontes de dados; unidade do Abstract.
