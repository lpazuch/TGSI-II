# Alterações pendentes no artigo (REV-03, REV-05 a REV-08)

O arquivo `TGSI II - Artigo - Final.docx` **ainda não foi alterado**. As
mudanças numéricas dependem da rodada definitiva com a base corrigida
(REV-01). As mudanças só de texto estão abaixo, com o local e a redação. Os
textos completos estão em `docs/methodology.md` §14.

| Local no artigo | Item | Alteração |
|---|---|---|
| Resumo/Abstract e §4.1 ("240 pares", "abril de 2026") | REV-02/08 | 239 pares, alvos de mai/2006 a mar/2026; abr/2026 excluído como mês parcial |
| §4.1, parágrafo das fontes | REV-01 | incluir a regra de disponibilidade: linha de t só com dados publicados até o fim de t; latências (NDVI fim da composição + 16 dias; Brent 10; NASA POWER e SMAP 3) |
| §4.1, parágrafo das 54 variáveis ("as mesmas ... em todas as análises") | REV-03 | 54 no holdout e na ablação; 48 a 54 no walk-forward de choque (85 origens com 48, porque a umidade do solo é descartada quando inteiramente vazia no treino); valores imputados antes de abr/2015 não são observações reais |
| §4.1, parágrafo do ARIMA ("teve o menor AIC") | REV-05 | trocar pela justificativa de diferenças desprezíveis e parcimônia (sem abr/2026, (0,1,1) tem AIC 1176,87 contra 1177,00) |
| §4.3, Diebold-Mariano (holdout) | REV-07 | texto do orientador: ausência de significância não demonstra equivalência |
| §4.3, Diebold-Mariano (amostra longa) | REV-07 | 216 → 215; listar as quatro diferenças em relação ao holdout (hiperparâmetros, variáveis efetivas, treino mínimo, período); declarar que holdout e amostra longa não foram corrigidos entre si |
| §4.4, último parágrafo da ablação | REV-06 | texto do orientador sobre o condicionamento do resultado; dependência da janela de validação; critério heurístico; ablação só sobre as selecionadas; janelas de 12 e 60 sobrepostas |
| §4.5, limitações | REV-01/05 | substituir a terceira limitação (sem data de publicação) pela de versões consolidadas e latências atuais aplicadas retroativamente; incluir o texto do REV-05 sobre o conjunto de teste não ser historicamente intocado |
| Tabelas 2, 3 e 4 e números do texto | REV-08 | substituir pelos resultados da rodada definitiva, com o rótulo da base e do commit |
