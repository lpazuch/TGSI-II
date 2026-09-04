# Fontes de dados — TGSI II

Documentação da origem e das transformações de cada variável usada na base
`data/processed/features_monthly_modeling.csv`.

Base geográfica: ponto único **Sorriso, MT** (lat −12,5425 / lon −55,7211),
estação INMET de referência **A904**. Janela de coleta configurada:
**2006-04-01 → 2026-04-07** (`configs/dataset.hybrid_2006plus.json`).

Legenda: *(a verificar)* = não determinável só pelo código/arquivos atuais;
precisa ser confirmado antes da escrita final do TCC.

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
| *(a verificar)* | Se, nesta run, algum trecho veio do fallback World Bank (checar `oil_source` no CSV) |

## 4. Clima (precipitação, temperatura, umidade, vento)

| Campo | Valor |
|---|---|
| Origem primária | **INMET** — Dados Históricos por estação (ZIP anual `portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip`). Estação **A904** |
| Fallback | **NASA POWER** — API `power.larc.nasa.gov/api/temporal/daily/point`, community `AG`, parâmetros `PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, RH2M, WS2M` |
| Variáveis / unidade | `precipitation_mm` (mm) · `temperature_mean_c` / `_max_c` / `_min_c` (°C) · `relative_humidity_pct` (%) · `wind_speed_ms` (m/s). `climate_source` registra `INMET` / `NASA_POWER` / `INMET+NASA_POWER` por dia |
| Frequência original | INMET: horária → agregada a diária (precip. = soma; temp. média/umidade/vento = média; máx/mín = máx/mín). NASA POWER: diária |
| Período | 2006-04-01 → 2026-04-07 |
| Agregação mensal | Precipitação = **soma**; temperatura média, umidade, vento = **média**; temp. máx = **máx**; temp. mín = **mín** (`_build_monthly_features`) |
| *(a verificar)* | Qual fonte de fato preencheu esta run (inspecionar `climate_source` em `data/raw/climate_combined_sorriso_mt.csv`); cobertura real da estação A904 antes de ~2008 |

## 5. NDVI (vigor da vegetação)

| Campo | Valor |
|---|---|
| Origem | **NASA LP DAAC** via **AppEEARS** (autenticação NASA Earthdata) — produto **MOD13Q1.061** (MODIS/Terra Vegetation Indices, composição de 16 dias, 250 m) |
| Referência | `appeears.earthdatacloud.nasa.gov/api` — extração `point` no ponto de Sorriso; layer escolhida automaticamente (prioridade `NDVI` / `_250m_16_days_NDVI`) |
| Variável / unidade | `ndvi` — adimensional, −1 a 1 (fator de escala aplicado se o CSV vier em inteiros) |
| Frequência original | Composição de 16 dias |
| Período | MODIS desde 2000; nesta run, alinhado à janela até 2026-04 |
| Agregação | Valor do ponto → **média mensal** |
| *(a verificar)* | Nome exato da layer retornada pelo AppEEARS (registrar de `data/raw/remote_ndvi_sorriso_mt.csv`, coluna `ndvi_source`) |

## 6. Umidade do solo

| Campo | Valor |
|---|---|
| Origem | **NASA NSIDC DAAC** via **AppEEARS** — produto **SPL3SMP_E.006** (SMAP Enhanced L3 Radiometer Global Daily 9 km Soil Moisture) |
| Referência | `appeears.earthdatacloud.nasa.gov/api` — layers `soil_moisture_am` / `soil_moisture_pm` (média das disponíveis) |
| Variável / unidade | `soil_moisture_m3m3` — m³/m³ |
| Frequência original | Diária (composições AM/PM) |
| Período | **A partir de 2015** (missão SMAP lançada em jan/2015). No dataset final há valores só a partir de ~fev/2015 → as ~107 primeiras linhas ficam sem umidade do solo. **Limitação conhecida, não é erro** |
| Agregação | **Média mensal** |
| *(a verificar)* | Layer(s) exata(s) retornada(s) pelo AppEEARS |

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
