# Análise exploratória — dengue nas capitais do Nordeste

## Base e escopo

A análise filtra a base pública InfoDengue/Mosqlimate arquivada localmente para as nove capitais. O recorte tem **9 capitais**, **5.256 linhas** e período de **04/01/2015 a 08/03/2026**. A unidade é capital × semana epidemiológica. Há 0 chaves capital-semana duplicadas.

O alvo é `casprov`, casos prováveis semanais. Incidência por 100 mil habitantes é apresentada como medida complementar. O rótulo `nivel_inc_mem == 2` indica incidência acima do limiar epidêmico do Moving Epidemics Method (MEM), conforme a documentação do InfoDengue/Mosqlimate. Os dados deste arquivo encerram em 8 de março de 2026; as análises não descrevem semanas posteriores.

### Origem dos dados

O arquivo público de dengue documenta casos prováveis como suspeitos menos descartados; o clima vem da reanálise ERA5/Copernicus organizada pelo Mosqlimate, e a população anual é de DATASUS/SVS. Links e períodos por fonte estão em `tables/fontes_e_periodos.csv`; as variáveis e unidades estão em `tables/dicionario_variaveis.csv`. A documentação oficial da base está em [Data documentation — 3rd IMDC](https://sprint.mosqlimate.org/data/), e a consulta municipal InfoDengue em [serviços da API](https://info.dengue.mat.br/services/api).

## Estatísticas descritivas

A proporção agregada de semanas com zero casos é 0,6%. Entre as semanas com valor positivo, mediana=47,0, P90=275,0 e P99=1.143,9 casos. As estatísticas por capital, incluindo dispersão, máximos, zeros, população e clima, estão em `tables/estatisticas_por_capital.csv`.

**Leitura para a modelagem:** as séries têm assimetria e picos; a mediana, os quantis e a proporção de zeros devem ser considerados junto com a média. Contagens e incidência respondem a perguntas diferentes, então as previsões de casos serão convertidas em incidência para analisar o limiar.

## Séries, agregado e sazonalidade

`figures/01_series_por_capital.png` mostra a trajetória semanal de cada capital. `figures/02_agregado_casos_incidencia.png` compara o total de casos e a incidência agregada, calculada como soma de casos dividida pela soma da população. `figures/03_sazonalidade_ano_capital.png` sobrepõe os perfis por semana epidemiológica e ano para distinguir um padrão anual recorrente de anos atípicos.

**Leitura para a modelagem:** picos sazonais recorrentes justificam um baseline sazonal de 52 semanas e variáveis de calendário. Diferenças entre capitais e entre anos apoiam modelos globais com identificador de cidade e validação temporal; não se deve pressupor um único perfil sazonal idêntico.

## Decomposição STL

A STL usa `log(1 + casos)` e período de 52 semanas para cada capital. Cada arquivo `figures/stl_<geocódigo>.png` separa tendência, sazonalidade anual e resíduo. A transformação reduz a influência visual de picos muito grandes, mas não os remove da base.

**Interpretação simples:** tendência resume mudanças graduais; sazonalidade resume o padrão que tende a reaparecer a cada ano; resíduo contém flutuações e eventos que esse padrão não explica. Um pico no resíduo pode ser um surto real. A decisão é comparar modelos com sazonalidade anual e manter semanas de pico para o treino e a avaliação.

## Estacionariedade: ADF e KPSS

Na série log(1+casos), ADF e KPSS concordam em estacionariedade em 8/9 capitais; concordam em não estacionariedade em 0/9. Nas demais, os resultados são discordantes/inconclusivos. Veja os p-valores e decisões por capital na tabela de testes.

ADF tem como hipótese nula uma raiz unitária; p<0,05 é evidência contra essa hipótese. KPSS tem como hipótese nula estacionariedade; p<0,05 é evidência contra estacionariedade. Por isso, a leitura conjunta ajuda a separar concordância de discordância. A tabela `tables/testes_estacionariedade.csv` inclui estatísticas, defasagens usadas, p-valores, interpretação e decisão para `log1p_nivel` e `log1p_diferenca_52` em cada capital.

**Decisão para a modelagem:** a necessidade de diferenciação não será decidida olhando a série de teste. Transformações e diferenças serão ajustadas dentro de cada janela de treino; modelos também serão comparados sem diferença quando os testes discordarem.

## Correlação, CCF, Granger e VIF

As matrizes Pearson e Spearman são calculadas separadamente dentro de cada capital; as figuras mostram a mediana entre cidades e as tabelas mantêm os resultados individuais. A CCF compara log(1+casos) na semana atual com clima observado 2–8 semanas antes. O Granger usa diferenças sazonais de 52 semanas e testa se o clima melhora a previsão bivariada de casos. Os p-valores de Granger não têm correção por múltiplos testes e não demonstram causalidade.

As figuras `06_matriz_pearson.png`, `06_matriz_spearman.png`, `07_ccf_clima_defasagens.png` e `08_vif_clima.png` e as tabelas `correlacoes_por_capital.csv`, `ccf_por_capital.csv`, `granger_por_capital.csv`, `vif_por_capital.csv` e `decisao_variaveis_defasagens.csv` documentam essas análises.

Candidatas que resultam da triagem (uma por variável climática):

- `precip_med`: Pearson mediano 0,201 e Spearman mediano 0,246; lag CCF candidato 3 semanas (mediana 0,282); 8 testes Granger com p<0,05 sem ajuste; VIF mediano 2,2.
- `rel_humid_med`: Pearson mediano 0,323 e Spearman mediano 0,319; lag CCF candidato 4 semanas (mediana 0,394); 11 testes Granger com p<0,05 sem ajuste; VIF mediano 5,1.
- `temp_med`: Pearson mediano -0,206 e Spearman mediano -0,202; lag CCF candidato 2 semanas (mediana -0,159); 9 testes Granger com p<0,05 sem ajuste; VIF mediano 9,6.

**Conclusão para a modelagem:** as defasagens climáticas com maior |CCF| mediana ficam registradas como candidatas, junto com o número de testes Granger significativos e o VIF mediano. Isso não autoriza escolher variáveis olhando o teste. A seleção final ocorre dentro do treino walk-forward; para um horizonte h, clima só entra se o lag for ≥ h ou se houver previsão climática disponível na origem.

### Diferença entre versões públicas de casos

A série `casprov` da extração InfoDengue e o arquivo histórico público têm 183 diferenças em 5256 semanas-capital (3,48%); maior diferença absoluta=159 casos. A auditoria por cidade está em `tables/auditoria_versoes_casos.csv`. Mantive `casprov` como alvo por consistência com a extração usada no alerta MEM e registrarei a data da extração; as diferenças refletem revisão/versão, não dados sintéticos.

## ACF e PACF

Os arquivos `figures/acf_pacf_<geocódigo>.png` mostram a autocorrelação (ACF) e a autocorrelação parcial (PACF) de `log(1 + casos)` até 52 semanas. Picos próximos a 52 semanas são compatíveis com dependência anual; picos nas primeiras semanas sugerem persistência de curto prazo. ACF/PACF orientam defasagens candidatas, mas não selecionam o modelo sozinhas.

**Decisão para a modelagem:** testar lags de casos de 1 a 8 semanas e 52 semanas como candidatos; a validação walk-forward decide quais ajudam. Para previsão de 4 semanas, recursos só entram quando seriam conhecidos na origem da previsão.

## Faltantes e outliers

Há 0 valores faltantes no alvo. Nas variáveis climáticas avaliadas, há 0 células ausentes (0,000% das células clima-capital-semana). A tabela `tables/faltantes_por_variavel_capital.csv` localiza ausências e distingue faltantes de zeros observados. Semanas ausentes não serão convertidas automaticamente em zero; se houver faltante climático, serão avaliados indicadores de ausência e imputação temporal/por série apenas no treino, com análise de sensibilidade.

A regra IQR sinalizou 575 valores altos para inspeção; 538 deles também coincidem com o indicador de surto MEM quando disponível. A lista está em `tables/picos_sinalizados_para_revisao.csv`. A regra estatística é apenas uma triagem: **nenhum pico foi removido ou winsorizado**. Só corrigiremos um registro se houver evidência verificável de erro de origem ou unidade. Assim, surtos observados permanecem na análise.

O gráfico `figures/05_picos_para_revisao.png` identifica os pontos sinalizados e os mantém desenhados. Extremos meteorológicos também serão conferidos com os intervalos plausíveis e com os valores da fonte antes de qualquer correção; não será feita exclusão automática.

## Limitações que afetam a leitura

A base é histórica e pode ser revisada pela fonte; termina em março de 2026. Casos prováveis são suspeitos menos descartados, agregados pela semana epidemiológica de início dos sintomas. Clima é reanálise horária ERA5 resumida por município e semana, não observação independente de uma estação por capital. `precip_med` é a taxa média semanal de precipitação em mm/h; ela não é precipitação total semanal. População vem da série municipal anual do DATASUS, carregada para frente sem usar anos futuros.

A documentação do pacote público de dados 3rd IMDC 2026 especifica os arquivos de dengue, clima ERA5 e população; a documentação da API Mosqlimate indica chave de usuário para a consulta ao datastore. A API antiga `alertcity` respondeu a uma consulta de teste para São Luís, mas a EDA usa os dados já baixados e mantém todos os arquivos originais no projeto.

## Arquivos

- Tabelas: `tables/fontes_e_periodos.csv`, `tables/dicionario_variaveis.csv`, `tables/estatisticas_por_capital.csv`, `tables/faltantes_por_variavel_capital.csv`, `tables/picos_sinalizados_para_revisao.csv`, `tables/testes_estacionariedade.csv`, `tables/stl_interpretacao.csv`, `tables/serie_agregada_capitais.csv`, `tables/perfil_sazonal_ano_semana.csv`, `tables/interpretacao_graficos.csv`, `tables/correlacoes_por_capital.csv`, `tables/ccf_por_capital.csv`, `tables/granger_por_capital.csv`, `tables/vif_por_capital.csv`, `tables/decisao_variaveis_defasagens.csv` e `tables/auditoria_versoes_casos.csv`.
- Figuras: séries por cidade, agregado, sazonalidade, mapa de faltantes, triagem de picos, matrizes Pearson/Spearman, CCF, VIF, nove STL e nove painéis ACF/PACF.