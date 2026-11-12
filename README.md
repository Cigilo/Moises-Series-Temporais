# Previsão semanal de dengue nas capitais do Nordeste

Projeto final da disciplina **Séries Temporais e Análise Preditiva**. O trabalho estima casos prováveis semanais e a probabilidade de ultrapassar o nível epidêmico oficial MEM, com horizonte de quatro semanas.

## Conteúdo da entrega

- `notebooks/`: quatro notebooks na ordem de execução indicada abaixo.
- `data/processed/`: painéis tratados e dicionário de variáveis.
- `src/`: scripts Python usados pelos notebooks.
- `reports/`: interpretações, tabelas e figuras geradas.
- `deliverables/`: apresentação utilizada pelo grupo e relatório técnico.
- `requirements.txt`: dependências.

## Ordem de execução

1. `01_analise_exploratoria.ipynb` — estatísticas e gráficos das séries, sazonalidade, STL, faltantes, extremos e testes ADF/KPSS e ACF/PACF. Usa o painel exploratório com ERA5.
2. `02_correlacao_clima.ipynb` — correlações, CCF, Granger com ajuste BH e VIF. Examina associações sem interpretá-las como causalidade.
3. `03_comparacao_clima.ipynb` — compara especificações sem clima, com temperatura e com variáveis climáticas, nas mesmas origens temporais.
4. `04_modelagem_e_validacao.ipynb` — compara modelos, avalia a classificação do limiar MEM e produz previsões de quatro semanas por validação walk-forward.

Os notebooks chamam scripts em `src/` e salvam resultados em `reports/`. As bases tratadas estão incluídas; executar os notebooks não exige coleta online.

## Ambiente e execução

Requer Python 3.10 ou superior. No PowerShell, a partir da pasta do repositório:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name dengue-ne --display-name "Python (Dengue NE)"
.\.venv\Scripts\python.exe -m jupyter lab
```

Abra os notebooks na sequência acima e execute as células de cima para baixo. A validação é temporal, expansiva e sem embaralhamento: cada treino utiliza somente informações disponíveis até sua data de corte.

## Dados, períodos e fontes

| Arquivo | Conteúdo | Período coberto |
|---|---|---|
| `data/processed/painel_capitais_eda.parquet` | 5.256 linhas, nove capitais, casos, população e clima ERA5/Mosqlimate | 04/01/2015 a 08/03/2026 |
| `data/processed/painel_capitais_modelagem.parquet` | 5.553 linhas, nove capitais, casos, nível MEM, população e clima NASA POWER | 04/01/2015 a 25/10/2026 |
| `data/dicionario_variaveis.csv` | Fontes, unidades, frequência e período de cada variável | Consulte antes de interpretar as colunas |

Os painéis têm períodos e fontes climáticas diferentes: a EDA usa ERA5 e termina em março de 2026; a modelagem usa uma extração posterior e NASA POWER até outubro de 2026. As análises finais de modelagem devem ser lidas junto com o respectivo período e fonte. Isso mantém claro o que foi usado em cada etapa e permite reproduzir os resultados.

- Casos prováveis e indicador MEM: [InfoDengue](https://info.dengue.mat.br/services/api/) e [Mosqlimate](https://sprint.mosqlimate.org/data/).
- Clima na EDA: ERA5/Copernicus, organizado pelo Mosqlimate.
- Clima na modelagem: [NASA POWER Daily](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), estimativa em grade, não leitura de estação local.
- População: DATASUS/SVS, conforme o dicionário de variáveis.

## Resultados resumidos

A Binomial Negativa teve o menor RMSE agregado (54,68 casos por capital-semana); o SARIMAX teve o menor MAE (26,50). O teste Diebold-Mariano não mostrou evidência suficiente para afirmar superioridade entre os dois em nenhum horizonte. A inclusão de temperatura trouxe ganho modesto no RMSE. A previsão é condicionada à última semana publicada e deve ser interpretada considerando o atraso de divulgação.
