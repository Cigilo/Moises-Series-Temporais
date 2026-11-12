# Ablation de variáveis climáticas

As três especificações usam a mesma família (GLM Binomial Negativa), as mesmas 24 origens expansivas, os mesmos alvos e o mesmo horizonte 1–4 semanas. Em cada origem, treino contém apenas linhas até a origem; as features de clima respeitam o horizonte. Não houve embaralhamento.

## Especificações

- **Sem clima:** casos defasados (horizonte e 52 semanas), calendário e exposição populacional.
- **Só temperatura:** especificação anterior + temperatura, com lag mínimo 2 e lag pelo menos igual ao horizonte.
- **Três variáveis climáticas:** especificação anterior + temperatura, precipitação e umidade, com lags preferidos 2, 3 e 4 semanas (limitados pelo horizonte).

## Resultado e decisão

Menor RMSE: **Só temperatura**, RMSE=54.68, MAE=26.91, MASE=0.250. Diferença de RMSE da melhor especificação contra Sem clima: +2.40% (positivo significa erro menor). Veja cobertura e largura dos intervalos antes de preferir um modelo.

A especificação final do modelo de produção deve equilibrar RMSE, MAE, MASE e cobertura dos intervalos, não apenas correlações. Esta comparação avalia inclusão de clima no modelo NB selecionado; não substitui as comparações entre famílias de modelos. As métricas detalhadas por horizonte e capital estão nas tabelas CSV ao lado.

Origens: 21/01/2024 a 27/09/2026; uma origem por ponto espaçado no período recente. O conjunto de previsões foi salvo para auditoria.
