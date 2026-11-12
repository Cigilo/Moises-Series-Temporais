# Correlação e seleção de defasagens — nove capitais do Nordeste

## Como ler esta análise

Pearson mede associação linear; Spearman mede associação monotônica por postos e é menos sensível a valores extremos. As matrizes brutas misturam cidades, níveis e sazonalidade e devem ser lidas como descrição, não como efeito causal. As matrizes de anomalias subtraem, em cada capital, a média histórica daquela semana epidemiológica; isso reduz a influência do ciclo anual e das diferenças de clima entre capitais.

A CCF associa casos na semana t ao clima em t−k, para k=2…8. A anomalia de casos é log1p(casos) menos a climatologia da capital e da semana do ano. Granger pergunta se lags climáticos acrescentam informação preditiva além da sequência de casos; não prova causalidade. Os p-valores de Granger foram ajustados por Benjamini–Hochberg (BH) sobre todas as combinações de capital, clima e lag.

## CCF e defasagens candidatas

A figura `ccf_anomalias_por_lag.png` mostra uma curva por clima: cada ponto é a mediana da correlação entre capitais e a faixa sombreada contém os 50% centrais das capitais. Valores perto de zero indicam associação linear pequena. A figura `comparacao_series_clima_dengue.png` coloca as anomalias padronizadas ao longo do tempo na mesma escala; ela ajuda a comparar quando cada série ficou acima ou abaixo do seu padrão sazonal, mas não prova causalidade.

- **Precipitação:** maior associação mediana em valor absoluto no lag 8 semanas (Pearson mediano=0.001, sinal positiva; 5/9 capitais compartilham o sinal majoritário). Esta é uma defasagem candidata, não uma relação causal nem uma seleção automática.
- **Temperatura:** maior associação mediana em valor absoluto no lag 8 semanas (Pearson mediano=0.140, sinal positiva; 5/9 capitais compartilham o sinal majoritário). Esta é uma defasagem candidata, não uma relação causal nem uma seleção automática.
- **Umidade relativa:** maior associação mediana em valor absoluto no lag 7 semanas (Pearson mediano=0.002, sinal positiva; 5/9 capitais compartilham o sinal majoritário). Esta é uma defasagem candidata, não uma relação causal nem uma seleção automática.

## Granger

Nenhuma combinação capital–variável–lag permaneceu significativa a 5% após o ajuste BH. A evidência não sustenta retirar ou incluir uma variável apenas pelo teste; mantemos o clima como candidato e decidimos pelo desempenho temporal fora da amostra.

## Multicolinearidade (VIF)

Entre 40 preditores candidatos, 22 têm VIF acima de 5 e 22 acima de 10. VIF alto indica que vários lags repetem informação; por isso não se deve colocar todos os lags climáticos simultaneamente numa regressão interpretável. Os VIFs e o gráfico `vif_top20.png` identificam redundância entre lags de casos, clima e calendário.

## Decisão para a modelagem

A conclusão para a especificação parcimoniosa é manter lags de casos recentes e de 52 semanas, além da sazonalidade anual de calendário. Entre as variáveis meteorológicas, somente temperatura mostra associação consistente, ainda que fraca (Pearson mediano perto de 0,13); proponho temperatura no lag 2 como candidato mínimo para horizontes curtos e aumentar o lag conforme o horizonte para garantir que o dado já estivesse observado. Precipitação e umidade têm associações próximas de zero, não passam Granger após ajuste BH e devem ficar fora da especificação parcimoniosa inicial. A comparação das séries temporais é descritiva; a decisão final sobre incluir clima depende do desempenho no walk-forward. O primeiro backtest foi executado antes desta análise formal e incluía chuva e umidade nos modelos globais; ele deve ser tratado como referência. Na próxima revisão, comparar no mesmo walk-forward: (a) casos/calendário, (b) casos/calendário + temperatura, e (c) todos os climas. VIF alto (incluindo 22 variáveis acima de 10) recomenda não incluir vários lags redundantes simultaneamente. A decisão definitiva é pelo desempenho temporal fora da amostra, com os recursos limitados ao que existia na origem.

As matrizes, coeficientes e testes foram calculados sobre o histórico completo apenas para descrição e geração de hipóteses. Eles não alteram o corte temporal do backtest. Correlação e Granger não estabelecem causalidade.
