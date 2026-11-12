# Modelagem e validação — nove capitais do Nordeste

## Dados e origem de previsão

A base de casos do InfoDengue foi atualizada em 09/11/2026 e contém observações até a SE 202643 (25/10/2026, início da semana epidemiológica), com **9 capitais** e **5,553 semanas-capital**. Clima diário NASA POWER foi agregado a semanas iniciadas no domingo; população vem do último ano conhecido no painel DATASUS, sem usar anos futuros. O corte local do modelo usa casos prováveis `casprov` e rótulo oficial `nivel_inc_mem == 2` para incidência acima do limiar MEM.

NASA POWER é uma fonte meteorológica em grade, não uma estação observacional local. O modelo usa NASA POWER de forma consistente em todo o período para temperatura (°C), precipitação acumulada semanal (mm) e umidade relativa (%). A EDA exploratória anterior usou reanálise ERA5 do pacote Mosqlimate; correlações dela são candidatas, enquanto a seleção final de variáveis será confirmada no backtest com as variáveis NASA. Fontes, períodos, unidades e transformações estão em `tables/fontes_dados_modelagem.csv` e `tables/dicionario_modelagem.csv`.

## Validação temporal

Foram usadas 24 origens expansivas entre 21/01/2024 e 27/09/2026, espaçadas ao longo dos últimos anos. Em cada origem, o treino usa somente semanas até aquela origem; nenhuma divisão aleatória é usada. Cada modelo prevê diretamente horizontes de 1 a 4 semanas, e as explicativas climáticas têm defasagem suficiente para estarem disponíveis no corte.

## Modelos e intervalos

1. Baseline sazonal: repete casos observados 52 semanas antes.
2. Binomial Negativa global: GLM de contagem com efeitos fixos de capital, exposição populacional e dispersão NB2 fixada em α=1; intervalos 90% pela distribuição binomial negativa.
3. SARIMAX sazonal local: ARMA(1,1) sobre log(1+casos), com dois pares de harmônicos Fourier anuais como regressoras sazonais; intervalos 90% do estado-espaço transformados para casos.
4. Random Forest global: árvores com efeitos de capital; intervalos 90% pelo pool empírico de valores nas folhas das árvores (aproximação de floresta quantílica).
5. LightGBM Poisson global: perda Poisson; intervalos 90% por regressões quantílicas nos níveis 5% e 95%.

Após uma comparação de especificações no mesmo walk-forward, todos os modelos globais usam temperatura como único preditor meteorológico. Para NB, entra um único lag por horizonte (mínimo 2 semanas); RF e LightGBM recebem lags de temperatura observados e disponíveis na origem. Precipitação e umidade foram excluídas da especificação principal porque não melhoraram o RMSE da NB e elevaram redundância entre preditores. A cobertura e largura dos intervalos são verificadas no walk-forward; a cobertura observada pode ficar abaixo dos 90% nominais.

## Resultados

O menor RMSE agregado foi de **Binomial Negativa global** (54.68 casos/semana); o segundo menor foi **SARIMAX sazonal** (59.23). A tabela completa com MAE, RMSE, MAPE apenas em semanas positivas, cobertura do MAPE, sMAPE e MASE está em `tables/metricas_modelos.csv`, por horizonte em `tables/metricas_por_horizonte.csv` e por capital em `tables/metricas_por_capital.csv`.

Como dengue tem zeros e contagens pequenas em parte das semanas, MAPE é reportado apenas onde casos observados >0 e vem acompanhado da cobertura; sMAPE e MASE completam a avaliação.

A ablation climática pareada está em `tables/comparacao_especificacoes_clima.csv`: NB com temperatura apenas teve RMSE 54,68; sem clima, 56,03; com temperatura, chuva e umidade, 55,55. O ganho é modesto, por isso temperatura fica como covariável candidata e a exclusão dos outros climas é mantida por parcimônia, não como prova de ausência de associação.

### Precisão de eventos MEM

O classificador LightGBM de cruzamento MEM (probabilidade ≥0,5) obteve precisão=0.635, recall=0.810 e F1=0.712 no conjunto de origens. O desbalanceamento e a escolha do limiar afetam esses números; a tabela também separa os quatro horizontes.

### Diebold–Mariano

Os dois modelos comparados são Binomial Negativa global e SARIMAX sazonal, escolhidos pela RMSE de validação, não por desempenho no treino. O teste usa diferença de perda quadrática regional, agregada por semana-alvo e horizonte, com variância HAC até lag 3 e correção de amostra finita. Resultados por horizonte estão em `tables/diebold_mariano.csv`; p>0,05 significa evidência insuficiente de diferença, não equivalência.

### SARIMAX

Ajustes SARIMAX concluídos: 216; falhas documentadas: 0. Ver `tables/diagnostico_sarimax.csv`; não foram substituídos silenciosamente por dados inventados.

## Previsão a partir do dado mais recente

O modelo selecionado por RMSE (Binomial Negativa global) gera previsões para **01/11/2026–22/11/2026**, correspondentes aos horizontes 1–4 a partir de 25/10/2026. O InfoDengue tem atraso de divulgação: na data de execução (09/11/2026), as primeiras semanas deste horizonte já podem ter começado ou terminado, embora a série pública ainda pare na SE 202643. Portanto, são previsões condicionadas à última semana publicada, não previsões em tempo real para quatro semanas completas a partir de hoje.

Os arquivos `tables/previsao_4_semanas_por_modelo.csv` e `tables/resumo_previsao_por_capital.csv` incluem casos, intervalo 90%, incidência por 100 mil habitantes e probabilidade estimada de ultrapassar o limiar MEM. Para previsão de incidência, o modelo não inventa um corte futuro: usa o classificador treinado no rótulo oficial MEM.

## Limitações e próximo passo

As origens de validação são 24 semanas escolhidas ao longo dos últimos ~145 períodos; a amostra temporal ainda limita o teste DM. A chuva da NASA é acumulada semanal em mm, enquanto a EDA ERA5 anterior usou taxa média em mm/h. O pipeline mantém fontes e unidades distintas documentadas.

Não se deve reutilizar previsões dos rascunhos antigos: esta análise foi recalculada para nove capitais com casos atualizados em novembro de 2026.
