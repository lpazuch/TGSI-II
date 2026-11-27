# Dados do CEPEA/ESALQ — licença e atribuição

**Fonte:** CEPEA – Centro de Estudos Avançados em Economia Aplicada, ESALQ/USP.
Indicador da Soja CEPEA/ESALQ – Paranaguá, obtido pela consulta pública em
`https://www.cepea.org.br/br/indicador/soja.aspx`.

**Licença:** Creative Commons Atribuição-NãoComercial 4.0 Internacional
(**CC BY-NC 4.0**), <https://creativecommons.org/licenses/by-nc/4.0/deed.pt-br>.

**Declaração do CEPEA:** página "Licença de uso de dados"
(<https://www.cepea.org.br/br/licenca-de-uso-de-dados.aspx>, consultada em
26/11/2026). Segundo a página, todo o conteúdo do site, incluindo os dados,
segue essa licença. O uso comercial é proibido, salvo autorização expressa do
CEPEA (contato informado na página: onetoone@cepea.org.br).

Os dados podem ser redistribuídos e adaptados nas seguintes condições:

- **Atribuição:** cite o CEPEA/ESALQ-USP como fonte;
- **Licença:** indique a licença CC BY-NC 4.0;
- **Modificações:** informe as alterações feitas;
- **Uso não comercial:** não use os dados para fins comerciais.

A licença do CEPEA vale só para os dados abaixo, não para o código deste
repositório.

## Arquivos cobertos

| Arquivo | Relação com o original |
|---|---|
| `data/raw/CEPEA_20260407190923.xls` | Arquivo original, sem modificação. Exportado em 07/04/2026. SHA-256 `4e0cb593df0f3d29ac23413dd7187f9bba0003b47610e9998df72b1c42f18c40` |
| `data/raw/target_soy_cepea_daily.csv` | **Adaptação:** as colunas `À vista R$` e `À vista US$` foram renomeadas para `soy_price_brl_bag` e `soy_price_usd_bag`, e a data foi convertida para ISO |
| `data/raw/target_soy_cepea_monthly.csv` | **Adaptação:** média aritmética mensal dos pregões |
| `docs/verification_evidence/rev04/*_predictions.csv` | **Adaptação:** a coluna `actual` é a média mensal acima |
| `data/processed/*` (gerados localmente, fora do Git) | **Adaptação:** preço corrente e alvo t+1 derivados das médias mensais |

## Atribuição sugerida

> Dados: CEPEA/ESALQ-USP, Indicador da Soja CEPEA/ESALQ – Paranaguá
> (consulta em 07/04/2026), licença CC BY-NC 4.0. Médias mensais calculadas
> neste trabalho.
