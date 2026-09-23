# Fontes de dados — TGSI II

Documentação da origem e das transformações de cada variável usada na base
`data/processed/features_monthly_modeling.csv`.

Base geográfica: ponto único **Sorriso, MT** (lat −12,5425 / lon −55,7211),
estação INMET de referência **A904**. Janela de coleta configurada:
**2006-04-01 → 2026-04-07** (`configs/dataset.hybrid_2006plus.json`).

Legenda: *(a verificar)* = não determinável só pelo código/arquivos atuais;
precisa ser confirmado antes da escrita final do TCC.

> **Proveniência exata (auditoria 13/09/2026, Passo 1).** `data/raw/` é
> versionado no Git — é o material que qualquer clone recebe. A planilha bruta
> da CEPEA está em `data/raw/CEPEA_20260407190923.xls`:
> - **URL/fonte:** consulta pública do Indicador CEPEA/ESALQ — Soja,
>   Paranaguá (`https://www.cepea.esalq.usp.br/br/indicador/soja.aspx`);
> - **Data do download:** 07/04/2026 (embutida no nome do arquivo);
> - **Procedimento:** exportação manual da série completa pela interface do
>   site (não há endpoint de API público documentado pela CEPEA);
> - **Nome esperado:** `CEPEA_20260407190923.xls`;
> - **SHA-256:** `4e0cb593df0f3d29ac23413dd7187f9bba0003b47610e9998df72b1c42f18c40`.
>
> Os demais arquivos de `data/raw/` (câmbio, Brent, clima, NDVI, umidade do
> solo, alvo diário/mensal) foram gerados pelo próprio pipeline
> (`tgsi_pipeline.pipeline.run_pipeline`, modo `--online`) na janela
> 2006-04-01 → 2026-04-07, a partir das fontes descritas abaixo — não são
> download manual, são reprodutíveis via `scripts/build_dataset.py --online`.

---

## 1. Alvo — preço da soja CEPEA/ESALQ

| Campo | Valor |
|---|---|
| Origem | CEPEA (Centro de Estudos Avançados em Economia Aplicada) / ESALQ-USP — Indicador da Soja, praça **Paranaguá** |
| Referência | Planilha `.xls` exportada manualmente do site do CEPEA (`data/raw/CEPEA_20260407190923.xls`; download em **07/04/2026**). Página: consulta pública do Indicador Soja CEPEA/ESALQ Paranaguá |
| Variáveis | `À vista R$` → `soy_price_brl_bag` · `À vista US$` → `soy_price_usd_bag` |
| Unidade | R$ (e US$) por **saca de 60 kg** |
| Frequência original | Diária (dias úteis). Série começa em **13/03/2006** e vai até 07/04/2026 |
| Transformação | Parse do `.xls` (`xlrd`) → série diária (`target_soy_cepea_daily.csv`) |
| Agregação | **Média aritmética mensal** dos dias com cotação (`cepea.resample_monthly`) → `target_soy_cepea_monthly.csv` |
| Alvo de modelagem | `soy_price_brl_bag_next_month` = preço BRL do **mês seguinte** (deslocamento +1 mês em `_attach_monthly_target`). Ver `methodology.md`. `soy_price_usd_bag` fica só como variável explicativa / rastreabilidade — **não há experimento oficial em USD** |

## 2. Câmbio USD/BRL

| Campo | Valor |
|---|---|
| Origem | **Banco Central do Brasil — PTAX**, API Olinda |
| Referência | `https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(...)` — moeda `USD` |
| Variável | `cotacaoVenda` (fallback `cotacaoCompra`) → `usd_brl` |
| Unidade | BRL por 1 USD |
| Frequência original | Diária (dias úteis) |
| Período | 2006-04-01 → 2026-04-07 |
| Transformação | Nenhuma sobre o valor |
| Agregação | **Média mensal** (`resample_rows`, modo `mean`) |

## 3. Petróleo Brent

| Campo | Valor |
|---|---|
| Origem | **U.S. Energy Information Administration (EIA)** — série RBRTE (Europe Brent Spot Price FOB). Fallback: **World Bank "Pink Sheet"** (Commodity Markets), planilha `CMO-Historical-Data-Monthly.xlsx`, coluna *Crude oil, Brent* |
| Referência | EIA: `eia.gov/dnav/pet/hist/LeafHandler.ashx?f=d&n=PET&s=RBRTE` (tabela HTML) · World Bank: página "Pink Sheet" → workbook mensal |
| Variável | `brent_usd_bbl` |
| Unidade | USD por barril |
| Frequência original | Diária (EIA) ou mensal (fallback World Bank) |
| Período | 2006-04-01 → 2026-04-07 |
| Transformação | Scraping de tabela HTML → série diária (`economics_brent_daily.csv`) |
| Agregação | **Média mensal** |
| Fonte usada nesta run | **100% EIA** (5060/5060 linhas com `oil_source=EIA`; fallback World Bank não foi acionado) — verificado em `data/raw/economics_brent_daily.csv` |

## 4. Clima (precipitação, temperatura, umidade, vento)

| Campo | Valor |
|---|---|
| Origem primária | **INMET** — Dados Históricos por estação (ZIP anual `portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip`). Estação **A904** |
| Fallback | **NASA POWER** — API `power.larc.nasa.gov/api/temporal/daily/point`, community `AG`, parâmetros `PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, RH2M, WS2M` |
| Variáveis / unidade | `precipitation_mm` (mm) · `temperature_mean_c` / `_max_c` / `_min_c` (°C) · `relative_humidity_pct` (%) · `wind_speed_ms` (m/s). `climate_source` registra `INMET` / `NASA_POWER` / `INMET+NASA_POWER` por dia |
| Frequência original | INMET: horária → agregada a diária (precip. = soma; temp. média/umidade/vento = média; máx/mín = máx/mín). NASA POWER: diária |
| Período | 2006-04-01 → 2026-04-07 |
| Agregação mensal | Precipitação = **soma**; temperatura média, umidade, vento = **média**; temp. máx = **máx**; temp. mín = **mín** (`_build_monthly_features`) |
| Fonte usada nesta run | **100% NASA_POWER** (7312/7312 dias com `climate_source=NASA_POWER`, verificado em `data/raw/climate_combined_sorriso_mt.csv`). Não existe `climate_inmet_*.csv` em `data/raw/` — a estação INMET **A904** não retornou nenhuma linha usável na janela 2006-04→2026-04 (fetch falhou ou a estação não tem dados históricos publicados nesse período; o pipeline só grava esse arquivo `if inmet_rows:`) |
| *(a verificar)* | Causa exata da ausência de dados INMET/A904 (estação sem histórico publicado vs. falha pontual de rede/parsing no dia da coleta) |

## 5. NDVI (vigor da vegetação)

| Campo | Valor |
|---|---|
| Origem | **NASA LP DAAC** via **AppEEARS** (autenticação NASA Earthdata) — produto **MOD13Q1.061** (MODIS/Terra Vegetation Indices, composição de 16 dias, 250 m) |
| Referência | `appeears.earthdatacloud.nasa.gov/api` — extração `point` no ponto de Sorriso; layer escolhida automaticamente (prioridade `NDVI` / `_250m_16_days_NDVI`) |
| Variável / unidade | `ndvi` — adimensional, −1 a 1 (fator de escala aplicado se o CSV vier em inteiros) |
| Frequência original | Composição de 16 dias |
| Período | MODIS desde 2000; nesta run, alinhado à janela até 2026-04 |
| Agregação | Valor do ponto → **média mensal** |
| Layer usada nesta run | **`_250m_16_days_NDVI`** do produto `MOD13Q1.061` (461/461 linhas com `ndvi_source=AppEEARS:MOD13Q1.061:_250m_16_days_NDVI`, verificado em `data/raw/remote_ndvi_sorriso_mt.csv`) |

## 6. Umidade do solo

| Campo | Valor |
|---|---|
| Origem | **NASA NSIDC DAAC** via **AppEEARS** — produto **SPL3SMP_E.006** (SMAP Enhanced L3 Radiometer Global Daily 9 km Soil Moisture) |
| Referência | `appeears.earthdatacloud.nasa.gov/api` — layers `soil_moisture_am` / `soil_moisture_pm` (média das disponíveis) |
| Variável / unidade | `soil_moisture_m3m3` — m³/m³ |
| Frequência original | Diária (composições AM/PM) |
| Período | **A partir de 2015** (missão SMAP lançada em jan/2015). No dataset final há valores só a partir de ~fev/2015 → as ~107 primeiras linhas ficam sem umidade do solo. **Limitação conhecida, não é erro** |
| Agregação | **Média mensal** |
| Layer usada nesta run | **`Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm`** do produto `SPL3SMP_E.006` (1413/1413 linhas com `soil_moisture_source=AppEEARS:SPL3SMP_E.006:Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm`, verificado em `data/raw/remote_soil_moisture_sorriso_mt.csv`) — só a passagem **PM**; a passagem AM não foi retornada/selecionada nesta coleta |

## 7. Variáveis derivadas do calendário / da própria série

Geradas em `_build_monthly_ml_features` a partir das colunas acima — nenhuma
fonte externa:

| Variável | Definição |
|---|---|
| `month`, `quarter` | Mês (1–12) e trimestre da linha |
| `soy_planting_window` | 1 se mês ∈ {9,10,11,12}, senão 0 |
| `soy_harvest_window` | 1 se mês ∈ {1,2,3,4}, senão 0 |
| `<var>_lag_1/2/3` | Valor de `<var>` 1, 2 ou 3 meses antes (somente passado) |
| `<var>_rolling_mean_3` | Média móvel dos 3 meses anteriores (somente passado) |
| `<var>_month_anomaly` | `valor(t) − média dos meses-calendário iguais **anteriores** a t`. **Causal** (sem look-ahead) — ver `methodology.md` |

Aplicadas a: `precipitation_mm`, `temperature_mean_c`, `relative_humidity_pct`,
`wind_speed_ms`, `ndvi`, `soil_moisture_m3m3`, `usd_brl`, `brent_usd_bbl`
(anomalia: as 6 primeiras exceto `relative_humidity_pct` e `wind_speed_ms`).

## 8. Reproduzindo a coleta de sensoriamento remoto (NDVI e umidade do solo)

Só é necessário para o modo `--online` (o modo `--offline`, padrão, usa os
CSVs já versionados em `data/raw/` e não toca nisso). Procedimento:

1. **Criar conta NASA Earthdata:** `https://urs.earthdata.nasa.gov/users/new`
   (gratuita).
2. **Autorizar o app AppEEARS** na conta (feito automaticamente no primeiro
   login em `https://appeears.earthdatacloud.nasa.gov/`, ou manualmente em
   Earthdata → Applications → Authorized Apps → adicionar "AppEEARS").
3. **Configurar as credenciais como variáveis de ambiente** (nunca commitar):
   ```bash
   export EARTHDATA_USERNAME="seu_usuario"
   export EARTHDATA_PASSWORD="sua_senha"
   ```
4. **Produtos e layers consultados** pelo pipeline
   (`src/tgsi_pipeline/sources/remote_sensing.py`, via API REST do AppEEARS,
   `task_type: "point"`):
   - NDVI: produto `MOD13Q1.061`, layer `_250m_16_days_NDVI` (a que esta run
     efetivamente retornou — seleção é automática por prioridade, ver
     `_pick_ndvi_layers`);
   - Umidade do solo: produto `SPL3SMP_E.006`, layer
     `Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm` (idem, ver
     `_pick_soil_moisture_layers`).
5. **Filtro espacial:** ponto único, lat `-12.5425` / lon `-55.7211`
   (Sorriso–MT), definido em `configs/dataset.hybrid_2006plus.json` →
   `locations[0]`.
6. **Filtro temporal:** `date_range` do mesmo config (`2006-04-01` a
   `2026-04-07`).
7. **Escalas aplicadas:** `scale_factor`/`add_offset` vêm dos metadados do
   próprio produto AppEEARS; o pipeline só os aplica quando detecta que o
   valor bruto ainda não veio escalado (`_needs_scaling` em
   `remote_sensing.py` — evita escalar duas vezes um valor que o AppEEARS já
   devolveu em unidade física).
8. **Tempo de espera:** o AppEEARS processa a extração de forma assíncrona
   (fila de tasks); o config usa `task_timeout_seconds: 21600` (6h) e
   `poll_interval_seconds: 30`. Na prática o tempo varia de minutos a horas
   conforme a fila do serviço.
