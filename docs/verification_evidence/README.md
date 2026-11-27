# Evidências de verificação (REV-10)

Inventário organizado em 26/11/2026. Nenhuma execução nova de aprendizado de
máquina foi feita para montá-lo: foram reunidos os registros que já
existiam, e o que não existe mais está identificado.

**Limites de redistribuição:**
- **CEPEA:** as previsões trazem a coluna `actual`, que é uma adaptação da
  série CEPEA, sob CC BY-NC 4.0 (`data/raw/LICENSE-CEPEA.md`).
- **Fontes da NASA e da EIA:** os dados são de domínio público.
- **O que não está aqui:** nenhum arquivo contém credenciais (Earthdata)
  nem informações pessoais. Os caminhos locais foram substituídos por
  `<repo>` no log dos testes.

## Conteúdo

| Pasta / arquivo | O que é | Commit / base |
|---|---|---|
| `rev01_disponibilidade_54_variaveis.csv` | disponibilidade de cada variável na origem (§13.1) | `9fc99e1` |
| `rev01/dataset_manifest_apos_rev01.json` | manifesto da base corrigida pela regra de disponibilidade (linhas, período, % de ausentes, SHA-256) | commit das correções de 26/11; base `a686e62b…` |
| `rev01/comparacao_colunas_antes_depois.csv` | linhas alteradas e ausentes por coluna, base `596fba0f…` → `a686e62b…` | idem |
| `rev03/variaveis_efetivas_por_origem.csv` | nº de variáveis efetivas por origem do walk-forward, nas duas bases | idem |
| `rev04/2026-10-11_*` | previsões e métricas do holdout do autor, 11/10/2026 | worktree do item 18 (protocolo uniforme do holdout, §9); base com abr/2026, sha `692fc3ce…5e89085` |
| `rev04/2026-11-09_*` | previsões, métricas e origens do holdout do autor, 09/11/2026 | checkout principal; mesma base e mesmos hiperparâmetros |
| `2026-11-24_provisorio_sem_abril/` | **resultados intermediários** sem abr/2026 (holdout, choque, DM, ablação, ARIMA, CCF), CSV/JSON | código de `9fc99e1`; base `596fba0f…3c43e9b` (manifesto incluído). Anteriores à correção do REV-01, sem uso no artigo |
| `ambiente/2026-11-26_ambiente_local.txt` | versão do Python, SO e arquitetura, `pip freeze` completo | ambiente de todas as execuções do autor desde out/2026 |
| `ambiente/2026-11-26_sha256_data_raw.txt` | SHA-256 de todos os arquivos de `data/raw/` | — |
| `ambiente/2026-11-26_pytest.txt` | log dos 47 testes automatizados | commit das correções de 26/11 |

## Configurações e sementes

Estão no próprio código, inalterado nas execuções acima:
- **holdout e ablação:** `experiments/holdout_12m.py`, RF e XGB com
  `random_state=42`;
- **análise de choque:** `experiments/shock_period_analysis.py`;
- **permutações da ablação:** semente 42.

Comandos usados nas execuções:

    PYTHONPATH=src python scripts/build_dataset.py
    PYTHONPATH=src python scripts/run_experiments.py
    PYTHONPATH=src python -m pytest

## O que não está disponível

- **Execuções independentes do orientador em outubro de 2026:** previsões
  por origem e ambiente. O orientador vai recuperá-los (REV-04).
- **Execuções do autor anteriores a 11/10/2026:** não foram guardadas
  (`results/` não é versionado). Ficaram só os valores citados nos commits e
  na documentação da época.
- **Figuras (`.png`) das execuções:** não foram preservadas aqui. São
  regeneradas pelos scripts.
- **Execução com a base corrigida (`a686e62b…`):** ainda não existe. Depende
  da conferência do commit pelo orientador (REV-09).
