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
  para cada origem i (as 12 últimas linhas, date = mar/2025 … fev/2026):
      treina do zero com as linhas [0, i)    (janela expansiva, desde abr/2006)
      prevê a linha i                         (features de date[i] → preço de target_month[i])
      incorpora a linha i e passa à origem seguinte
  ```
  - O treino da 1ª origem tem 227 linhas (date abr/2006 → fev/2025); o da
    12ª tem 238. A última linha do treino tem como alvo o preço de `date[i]`,
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
| mar/2006 | Mês **parcial**: a série diária da CEPEA começa em **13/03/2006** |
| abr/2026 | Mês **parcial**: a planilha foi exportada em **07/04/2026** (`CEPEA_20260407190923.xls`), então a média de abril tem só 4 pregões, de 1 a 7/4 |
| Janela do config (`date_range`) | **2006-04-01 → 2026-04-07** (data real da extração) |
| Meses completos usados como preço/alvo | abr/2006 → mar/2026 (**240 meses**) |
| Entrada do dataset final | abr/2006 → fev/2026 (**239 meses**) |
| Alvo (t+1) | mai/2006 → mar/2026 (**239 valores**) |
| **Total: 239 observações** | |

**Regra (24/11/2026, revisão do orientador, itens 6 e 7):** só entram como
preço corrente ou alvo os meses **inteiramente cobertos** pela janela de
extração `[date_range.start, date_range.end]`
(`tgsi_pipeline.pipeline.complete_month_bounds` / `_complete_target_months`,
aplicados no modo offline e no online). A regra também vale para os scripts
que leem a série crua (`arima_residuals.py`, `arima_sensitivity.py`).
Testes: `tests/test_target_pairing.py` e `tests/test_arima_windows.py`.
O arquivo cru `target_soy_cepea_monthly.csv` não foi alterado: ele continua
sendo a agregação fiel da planilha, e o corte acontece no uso.

**Contabilidade dos 242 → 239:**
- −1: mar/2006, parcial (anterior ao início configurado);
- −1: abr/2026, parcial (exportação em 07/04/2026); sai como alvo, e por isso
  mar/2026 fica sem par;
- −1: mar/2026 como **entrada**, por não ter alvo t+1 completo
  (`_build_monthly_modeling_dataset` descarta linhas sem
  `soy_price_brl_bag_next_month`).

**Histórico:** até 23/11/2026 a base tinha 240 pares, com alvos até
abr/2026. O alvo de abril/2026 era a média parcial de 1 a 7/4 e entrava como
última previsão do holdout, das 216 origens e da janela de 60 meses. Os
resultados anteriores a essa data usam essa base antiga.

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

**Conferência independente posterior (13/10/2026):** registro do autor:
uma nova execução, a partir de um clone do repositório, reproduziu os valores
da tabela acima, com as mesmas 12 origens por modelo
(`results/holdout_12m/holdout_12m_origins.csv`). Os logs dessa execução não
foram preservados.

**Reexecução para o artigo final (09/11/2026):** a base foi reconstruída
(`build_dataset.py --offline`, mesmo SHA-256 `692fc3ce…5e89085`) e todos os
experimentos foram rodados de novo em um ambiente virtual novo (Python 3.12.6,
versões fixadas em `requirements.txt`). As previsões individuais dos três
modelos são **idênticas** às de 11/10. Os valores 3,273438 (RF) e 3,515465
(XGBoost), que apareceram nesta tabela entre 24/10 e 16/11, são da execução
independente do orientador, e não de uma execução do autor. Ver §13.4
(REV-04). A linha de
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

---
## 13. Verificação de 24/11/2026 — REV-01 a REV-04 (respostas de 25/11/2026)

Os códigos REV-xx seguem o registro de acompanhamento do orientador de
24/11/2026. Nesta seção, **"origem da previsão"** é o instante em que a
previsão do mês t+1 é feita. O protocolo usa as variáveis do mês t para prever
a **média** do preço em t+1; a origem nominal adotada aqui é, portanto, o
**primeiro dia de t+1**, antes de qualquer pregão de t+1. Nenhum experimento
foi reexecutado para esta seção. As verificações abaixo são documentais
(VD), de código (VC) e numéricas sobre arquivos existentes (VN).

### 13.1. REV-01 — Disponibilidade temporal das variáveis explicativas

**Tabela completa (54 linhas):**
[`docs/verification_evidence/rev01_disponibilidade_54_variaveis.csv`](verification_evidence/rev01_disponibilidade_54_variaveis.csv).
Resumo por família (as 13 famílias somam as 54 colunas: 4 + 1 + 1 + 1 + 1 +
6 × 4 + 5 × 2 + 6 × 2):

| Família (nº colunas) | Fonte | Período de referência do valor de t | Quando fica disponível | Na origem (1º dia de t+1) | Revisões posteriores |
|---|---|---|---|---|---|
| Calendário (4) | data da linha | mês t | determinístico | sim | não se aplica |
| Preço R$ (1) | CEPEA, "À vista R$" | média dos pregões de t | diário; publicação no próprio dia **não confirmada** (site do CEPEA bloqueou a consulta automática) | sim, se a publicação diária se confirmar | não documentado |
| Preço US$ (1) | CEPEA, "À vista US$" (conversão do próprio CEPEA) | idem | idem | idem | não documentado |
| Câmbio (6) | BCB, PTAX venda | média dos dias úteis de t | diário; horário de divulgação **não confirmado** nesta revisão | sim | não documentado nesta revisão |
| Brent (6) | EIA, RBRTE | média dos dias de t | a tabela da EIA é atualizada em lote semanal (página consultada em 25/11/2026 indicava próxima atualização em uma semana) | **parcial**: os últimos dias de t podem sair só na semana seguinte | sem nota de revisão na página |
| Precipitação, temp. média (6 cada) | NASA POWER | soma / média de t | GEOS-IT "generally ready within about two days"; depois sobrescrito por MERRA-2 | **parcial**: faltariam ~2 dias finais de t | **sim**: "any GEOS-IT data are over-written" por MERRA-2, "typically updated every several months" |
| Temp. máx., temp. mín. (1 cada) | NASA POWER | máx./mín. diário de t | idem | parcial | sim |
| Umidade relativa, vento (5 cada) | NASA POWER | média de t | idem | parcial | sim |
| NDVI (6) | MODIS MOD13Q1.061 (AppEEARS) | média das composições de 16 dias **cuja data de início** cai em t | só depois do fim da composição; latência de produção não confirmada | **NÃO** (ver abaixo) | coleção 6.1 é reprocessamento; extensão não confirmada |
| Umidade do solo (6) | SMAP SPL3SMP_E.006 (AppEEARS) | média das passagens PM de t (~11 dias/mês) | latência média ~1 dia (NSIDC: média de "1 day 05:34:25", tabela de out/2025) | parcial: falta o último ~1 dia | versões novas reprocessam; não confirmado para a v6 |

Defasagens (`_lag_1/2/3`) e média móvel (`_rolling_mean_3`, média de t−1 a
t−3, **sem** t — `_rolling_mean` em `pipeline.py`) referem-se a meses
anteriores a t. Para todas as famílias, exceto NDVI, estão disponíveis na
origem. As anomalias usam o valor de t e herdam a disponibilidade dele.

**Problema identificado — NDVI (exige mudança de código):** a data de cada
registro do MOD13Q1 no arquivo é o **início** da composição. Todas as 461
datas caem nos dias 1 + 16k do ano, o calendário de início das composições
de 16 dias (VN em `data/raw/remote_ndvi_sorriso_mt.csv`). A agregação mensal
(`_build_monthly_features`, `resample_rows(..., aggregations={"ndvi": "mean"})`)
atribui cada composição ao mês do início. Com isso, **225 das 461
composições terminam no mês seguinte**, ou seja, o NDVI do mês t inclui até
15 dias de t+1, que é o mês-alvo. Isso atinge `ndvi` e `ndvi_month_anomaly`.
`ndvi_lag_1` e `ndvi_rolling_mean_3` dependem da latência de produção, que
não foi confirmada. A correção prevista é atribuir cada composição pelo
**último dia** do período, mais a latência de publicação, e só então agregar.
Ela muda a base e exige reexecução, que fica para discussão com o orientador
antes de ser feita.

**Limitações registradas, sem correção nesta etapa:**
- Brent, NASA POWER e SMAP têm defasagem de dias. Na origem nominal, os
  últimos dias de t poderiam não estar publicados. A base usa o mês completo.
- Os dados externos foram obtidos depois do período de referência (ver 13.2),
  então refletem a **versão disponível na data de download**, e não a que
  existia na época de cada previsão. Isso vale principalmente para o NASA
  POWER, cujo trecho GEOS-IT é sobrescrito por MERRA-2.
- Não foi possível comprovar, por documentação oficial, o horário de
  divulgação do CEPEA e da PTAX, nem a política de revisão de CEPEA, PTAX e
  EIA. Esses pontos ficam marcados como **não comprovados**.

Fontes consultadas em 25/11/2026: `power.larc.nasa.gov/docs/methodology/data/sources_meteorology/`;
`nsidc.org/data/spl3smp_e/versions/6` e
`nsidc.org/data/user-resources/help-center/what-are-latencies-smap-radiometer-data-sets`;
`eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RBRTE&f=D`;
`earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061`. As páginas do CEPEA
(HTTP 403) e do BCB (conteúdo carregado por JavaScript) não puderam ser lidas.

### 13.2. REV-02 — Proveniência e integridade dos dados

**CEPEA (alvo e preço corrente):**
- **Arquivo original:** `data/raw/CEPEA_20260407190923.xls`, SHA-256
  `4e0cb593df0f3d29ac23413dd7187f9bba0003b47610e9998df72b1c42f18c40`
  (conferido em 25/11/2026).
- **Identificação no próprio arquivo:** aba "Plan 1"; cabeçalho
  "INDICADOR DA SOJA CEPEA/ESALQ - PARANAGUÁ", "Fonte: Cepea"; colunas
  `Data`, `À vista R$`, `À vista US$`; 5.003 pregões, de 13/03/2006 a
  07/04/2026, sem valores ausentes nem datas duplicadas (VN).
- **Unidade:** o arquivo não traz unidade. A convenção do indicador (R$ e US$
  por saca de 60 kg, à vista, posto Paranaguá) **precisa ser confirmada na
  página do CEPEA**, que não pôde ser lida automaticamente.
- **Obtenção:** exportação manual pela consulta pública do site, em
  07/04/2026 às 19:09:23 (data embutida no nome do arquivo). Não há API.
- **Tratamento:** leitura com `xlrd` (`sources/cepea.py`,
  `ignore_workbook_corruption=True`, porque o `.xls` exportado tem um
  cabeçalho OLE irregular que o `xlrd` rejeita no modo estrito). A leitura
  foi conferida contra `target_soy_cepea_daily.csv`: diferença máxima 0,0
  nas duas colunas.
- **Agregação:** média aritmética mensal dos pregões (`cepea.resample_monthly`)
  → `target_soy_cepea_monthly.csv`. Recalculada a partir do `.xls`, a
  diferença máxima é 0,0 nos 242 meses.
- **Meses usados:** só os completos (§4). Março/2006 tem 15 pregões a partir
  de 13/03, e abril/2026 tem 4 pregões, até 07/04. Ambos ficam fora.
- **Conversões:** nenhuma no pipeline. A coluna US$ é a publicada pelo CEPEA.
- **Observação de código:** `resample_monthly` trata um preço ausente como 0
  e o conta na média. Como a série não tem ausentes, isso não afeta a base
  atual, mas é um defeito latente a corrigir.
- **Restrição de redistribuição:** **não verificada.** O `.xls` está em um
  repositório público. Se os termos do CEPEA restringirem a redistribuição,
  o arquivo deve sair do repositório, e um terceiro deverá reproduzir a
  exportação manual pela consulta pública do Indicador Soja CEPEA/ESALQ
  Paranaguá, com período a partir de 13/03/2006, e conferir o hash acima.

**Demais arquivos de `data/raw/`** (SHA-256 conferidos em 25/11/2026):

| Arquivo | SHA-256 | Origem |
|---|---|---|
| `climate_combined_sorriso_mt.csv` | `7101a850…250028db` | NASA POWER (100% dos dias) |
| `climate_nasa_power_sorriso_mt.csv` | `7101a850…250028db` (idêntico ao anterior) | NASA POWER |
| `economics_brent_daily.csv` | `aa785461…96f03` | EIA RBRTE |
| `economics_fx_bcb.csv` | `47c2478b…a86b4` | BCB PTAX |
| `remote_ndvi_sorriso_mt.csv` | `2f4c8828…f6ca` | MOD13Q1.061 / AppEEARS |
| `remote_soil_moisture_sorriso_mt.csv` | `5453a877…36f6` | SPL3SMP_E.006 / AppEEARS |
| `target_soy_cepea_daily.csv` | `7be89893…2e` | derivado do `.xls` |
| `target_soy_cepea_monthly.csv` | `49c27b4f…ea0` | derivado do `.xls` |

**Data de obtenção dos arquivos externos:** não há log de download. Os mesmos
arquivos, com hashes idênticos, estão na execução original do projeto do
TGSI I (`pipeline_tgsi/data/run_cepea_hybrid_2006plus_long_remote/raw/`),
com data de modificação de **21–22/08/2026**, e `run_report.json` sem avisos
nem erros. A data de download é, portanto, **provavelmente** 21–22/08/2026,
não comprovada. A janela pedida às fontes foi 2006-04-01 → 2026-04-07.

### 13.3. REV-03 — Umidade do solo

- **Disponibilidade original:** o SPL3SMP_E v6 começa em 31/03/2015 (NSIDC).
  O arquivo tem 1.413 registros diários, de 31/03/2015 a 06/04/2026, todos da
  passagem PM e sem valores ausentes. Todos os meses de mar/2015 a abr/2026
  têm ao menos um registro. Há em média 11 dias por mês (mínimo 1 em
  mar/2015, 2 em ago/2022 e 3 em set/2022 e jul/2019), por causa da órbita.
- **Na base mensal:** as 107 linhas de abr/2006 a fev/2015 não têm umidade do
  solo, nem as defasagens e a média móvel correspondentes. A anomalia só
  existe a partir do segundo ano de cada mês-calendário.
- **Tratamento:** não há interpolação, preenchimento a partir de outra fonte
  nem imputação na construção da base: os valores ficam ausentes em
  `features_monthly_modeling.csv`. A imputação acontece só nos experimentos,
  dentro de um `Pipeline` do scikit-learn com `SimpleImputer(strategy="median")`
  ajustado **apenas no treino de cada origem** (`experiments/holdout_12m.py:119`;
  `experiments/shock_period_analysis.py:109`; `experiments/ablation.py` usa o
  forecaster de `holdout_12m.py`).
- **Holdout (12 origens) e ablação (12 e 60 origens):** todo treino inclui
  meses de 2015 em diante. Os meses anteriores recebem a mediana dos valores
  de umidade do solo **do próprio treino**, que são todos anteriores à
  origem. Não há uso de informação futura, mas 107 meses recebem um valor
  constante.
- **Walk-forward da análise de choque:** nas origens cujo treino termina
  antes de mar/2015 (as ~84 primeiras), a coluna está inteiramente vazia no
  treino. O `SimpleImputer` do scikit-learn 1.8, com o padrão
  `keep_empty_features=False`, **descarta** essas colunas. Nessas origens, os
  modelos de árvore usam 48 colunas, não 54. O texto do artigo que fala em
  "mesmo conjunto de 54 variáveis" nessa análise precisa ser corrigido.
- **Informação futura:** nenhuma etapa usa valores posteriores à origem. A
  mediana é ajustada no treino. As defasagens, a média móvel e a anomalia são
  causais (`_build_monthly_ml_features`, `_causal_month_normal`).

### 13.4. REV-04 — Divergências numéricas entre execuções

**O que os registros mostram (VN):**

| Execução | Base (SHA-256) | RF MAE | XGB MAE | Origem do registro |
|---|---|---|---|---|
| 11/10/2026 (autor) | `692fc3ce…5e89085` | 3,268081 | 3,340820 | `docs/verification_evidence/rev04/2026-10-11_*` (cópia de `results/` da worktree do item 18) |
| Execução independente do orientador (out/2026) | não registrado aqui | 3,273438 | 3,515465 | citada em `methodology.md` no commit `2ac453a` (11/10) |
| 09/11/2026 (autor) | `692fc3ce…5e89085` | 3,268081 | 3,340820 | `docs/verification_evidence/rev04/2026-11-09_*` (cópia de `results/` do checkout principal) |
| 24/11/2026 (autor, base nova sem abr/2026) | `596fba0f…` | — | — | 11 previsões em comum com 09/11 |

- As previsões individuais de 11/10 e de 09/11 são **idênticas** nos 12 meses
  e nos três modelos. A diferença máxima é 0,0 no ARIMA e no XGBoost e
  2,8×10⁻¹⁴ no RF, nível de arredondamento de ponto flutuante.
- Na execução de 24/11, com a base nova, as 11 previsões dos meses-alvo em
  comum (mai/2025 a mar/2026) também são idênticas às de 09/11 (mesma
  tolerância). Para esses meses, o treino de cada origem é o mesmo.
- Todas as execuções do autor usaram Python 3.12.6, numpy 2.4.4, pandas
  3.0.2, scikit-learn 1.8.0, scipy 1.17.1, statsmodels 0.14.6 e xgboost
  3.2.0, em macOS (Darwin arm64). Os hiperparâmetros e a semente
  (`random_state=42`) não mudaram entre 11/10 e 09/11 (código de
  `holdout_12m.py` sem alteração nesses parâmetros). A ordem das colunas vem
  da lista fixa de features e a ordem das linhas é por `date`.

**Origem da confusão de números na documentação:** o commit `4aff63c`
(24/10/2026, "Ajuste das métricas de performance dos modelos") trocou na
tabela da §9 os valores do autor (3,268081 e 3,340820) pelos da execução
independente (3,273438 e 3,515465). O commit `9d41e63` (03/11) manteve essa
troca, e o `cdbf2b6` (16/11) voltou aos valores do autor. Por isso, a
redação anterior da §9, que falava em diferença "entre execuções" do autor,
estava **incorreta**: as execuções do autor são reproduzíveis entre si, e a
divergência é entre o ambiente do autor e o da execução independente.

**Causa:** **não determinada.** Não há, neste repositório, as previsões
individuais nem a descrição do ambiente da execução independente. Sem elas,
não é possível dizer se a diferença vem da base, das versões de bibliotecas,
da plataforma ou do procedimento. A hipótese de paralelismo (`n_jobs=-1`),
registrada antes, **não é sustentada**: no ambiente do autor, o resultado é
bit a bit idêntico entre execuções com o mesmo `n_jobs`. Para fechar o item,
são necessárias as previsões por origem da execução independente e o
ambiente usado (Python, bibliotecas, sistema operacional e arquitetura).

**Efeito nas conclusões:** nas duas versões, ARIMA e persistência ficam
abaixo de RF e XGBoost no holdout, e nenhuma diferença do holdout é
estatisticamente significativa (§11). A divergência não altera as
conclusões, mas a reprodutibilidade numérica de RF e XGBoost **entre
ambientes** continua como limitação.
