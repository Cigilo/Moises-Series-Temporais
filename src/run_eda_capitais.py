"""Exploratory analysis of dengue and climate in the nine Northeast capitals.

Input is the locally archived, public InfoDengue/Mosqlimate 2026 sprint panel
at data/processed/painel_capitais_eda.parquet. This script filters that real panel
to the nine capitals and writes slide/report-ready tables, figures, and notes.
It never removes epidemiological peaks identified by statistical rules.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "painel_capitais_eda.parquet"
OUT = ROOT / "reports" / "eda_capitais"
FIG = OUT / "figures"
TABLE = OUT / "tables"

CAPITALS = {
    "2111300": "São Luís",
    "2211001": "Teresina",
    "2304400": "Fortaleza",
    "2408102": "Natal",
    "2507507": "João Pessoa",
    "2611606": "Recife",
    "2704302": "Maceió",
    "2800308": "Aracaju",
    "2927408": "Salvador",
}
CASE_COL = "casprov"
CLIMATE_COLS = ["temp_med", "precip_med", "rel_humid_med", "rainy_days"]
CLIMATE_LABELS = {
    "temp_med": "Temperatura média (°C)",
    "precip_med": "Precipitação média semanal (mm/h)",
    "rel_humid_med": "Umidade relativa média (%)",
    "rainy_days": "Dias chuvosos (dias/semana)",
}
PALETTE = "tab10"
SEED = 20261109


def load_panel(path: Path = INPUT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Base não encontrada: {path}\n"
            "Baixe o arquivo público dengue.csv.gz, junte às séries semanais de clima e população "
            "da pasta data_imdc_2026 do InfoDengue/Mosqlimate e gere o painel tratado em "
            "data/processed/painel_capitais_eda.parquet. Consulte a documentação em "
            "https://sprint.mosqlimate.org/data/. Não substitua por dados sintéticos."
        )
    frame = pd.read_parquet(path)
    required = {"geocode", "date", CASE_COL, "population", "incidencia_100mil", "municipio_nome"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no painel: {sorted(missing)}")
    frame["geocode"] = frame["geocode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[CASE_COL] = pd.to_numeric(frame[CASE_COL], errors="coerce")
    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
    frame["incidencia_100mil"] = pd.to_numeric(frame["incidencia_100mil"], errors="coerce")
    frame = frame.loc[frame["geocode"].isin(CAPITALS)].copy()
    if frame.empty:
        raise ValueError("Nenhuma das nove capitais foi encontrada no painel.")
    frame["municipio"] = frame["geocode"].map(CAPITALS)
    frame["epiweek_year"] = frame["date"].dt.isocalendar().year.astype(int)
    frame["epiweek_number"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["log1p_casos"] = np.log1p(frame[CASE_COL].clip(lower=0))
    frame = frame.sort_values(["geocode", "date"])
    return frame


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date")
        y = g[CASE_COL]
        row = {
            "geocode_ibge": code,
            "municipio": name,
            "inicio": g.date.min(),
            "fim": g.date.max(),
            "n_semanas": int(g.date.nunique()),
            "casos_media": y.mean(),
            "casos_mediana": y.median(),
            "casos_desvio_padrao": y.std(),
            "casos_p90": y.quantile(.90),
            "casos_p95": y.quantile(.95),
            "casos_p99": y.quantile(.99),
            "casos_maximo": y.max(),
            "data_maximo": g.loc[y.idxmax(), "date"] if y.notna().any() else pd.NaT,
            "proporcao_zeros": (y == 0).mean(),
            "incidencia_media_100mil": g.incidencia_100mil.mean(),
            "populacao_mediana": g.population.median(),
        }
        for col in CLIMATE_COLS:
            if col in g:
                row[f"{col}_media"] = g[col].mean()
                row[f"{col}_desvio_padrao"] = g[col].std()
        records.append(row)
    result = pd.DataFrame(records)
    return result


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CASE_COL, "incidencia_100mil", "population"] + [c for c in CLIMATE_COLS if c in df]
    rows = []
    for col in columns:
        for code, name in CAPITALS.items():
            s = df.loc[df.geocode == code, col]
            rows.append({
                "variavel": col,
                "municipio": name,
                "n": len(s),
                "faltantes": int(s.isna().sum()),
                "pct_faltantes": 100 * s.isna().mean() if len(s) else np.nan,
                "zeros_observados": int((s == 0).sum()) if col in [CASE_COL, "incidencia_100mil", "rainy_days"] else np.nan,
                "decisao_modelagem": ("não imputar; tratar zero como observação" if s.isna().sum() == 0 else
                                      "identificar causa; não converter ausência em zero; imputar somente dentro do treino e marcar a imputação"),
            })
    return pd.DataFrame(rows)


def variable_dictionary() -> pd.DataFrame:
    rows = [
        ("geocode", "Código municipal IBGE", "IBGE/InfoDengue", "código municipal", "município", "Identificador da capital"),
        ("date", "Início da semana epidemiológica (domingo)", "InfoDengue/Mosqlimate", "data", "semanal", "Chave temporal"),
        (CASE_COL, "Casos prováveis semanais; suspeitos menos descartados", "InfoDengue (SINAN/IBGE)", "casos", "semanal", "Alvo principal"),
        ("casos_arquivo", "Casos prováveis da série histórica pública para auditoria", "InfoDengue/Mosqlimate, dengue.csv.gz", "casos", "semanal", "Comparação de versões, não substitui silenciosamente o alvo"),
        ("incidencia_100mil", "Casos por 100 mil habitantes", "Calculada: casos/população × 100.000", "casos/100 mil hab.", "semanal", "Medida auxiliar para pergunta de limiar"),
        ("nivel_inc_mem", "Nível MEM de incidência; 2 acima do limiar epidêmico", "InfoDengue/MEM", "nível categórico", "semanal", "Indicador de contexto/evento"),
        ("surto_mem", "Indicador de incidência acima do limiar MEM", "InfoDengue/MEM", "0/1", "semanal", "1 quando nível MEM é 2"),
        ("population", "População municipal estimada, último ano conhecido até a semana", "DATASUS/SVS, pacote público 3rd IMDC", "habitantes", "anual, carregada para frente", "Não usa população futura"),
        ("temp_med", "Temperatura média semanal", "ERA5/Copernicus, organizada por Mosqlimate", "°C", "semanal", "Média de dados horários resumidos por semana"),
        ("precip_med", "Taxa média semanal de precipitação", "ERA5/Copernicus, organizada por Mosqlimate", "mm/h", "semanal", "Não é precipitação total semanal"),
        ("rel_humid_med", "Umidade relativa média semanal", "ERA5/Copernicus, organizada por Mosqlimate", "%", "semanal", "Média semanal"),
        ("rainy_days", "Dias da semana em que precipitação total excedeu 0,03", "ERA5/Copernicus, organizada por Mosqlimate", "dias/semana", "semanal", "Contagem de dias chuvosos"),
    ]
    return pd.DataFrame(rows, columns=["variavel", "descricao", "fonte", "unidade", "frequencia", "uso_observacoes"])


def source_register() -> pd.DataFrame:
    rows = [
        ("casos e atributos epidemiológicos", "SINAN/IBGE, organizado por InfoDengue/Mosqlimate", "semanal por município", "2010–2026 (base do pacote); esta análise 2015-01-04 a 2026-03-08", "https://sprint.mosqlimate.org/data/", "Pacote público 3rd IMDC 2026; `casprov` usado como alvo e comparado com o arquivo histórico"),
        ("casos prováveis e níveis MEM", "InfoDengue", "semanal por município", "2015–2026 na extração local", "https://info.dengue.mat.br/services/api", "Consulta pública por código IBGE; `nivel_inc == 2` identifica incidência acima do limiar epidêmico MEM"),
        ("variáveis climáticas", "ERA5/Copernicus, organizado por Mosqlimate", "semanal por município", "1999–2026 (pacote); recorte da análise 2015-01-04 a 2026-03-08", "https://sprint.mosqlimate.org/data/", "Reanálise horária agregada por dia e depois por semana; precipitação média em mm/h"),
        ("população", "DATASUS/SVS", "anual por município", "2001–2025; último ano conhecido carregado para frente", "https://sprint.mosqlimate.org/data/", "Usada para calcular incidência sem recorrer a população futura"),
        ("API datastore", "Mosqlimate", "consulta paginada", "acesso verificado em 2026-11-09", "https://api.mosqlimate.org/docs/uid-key", "Exige chave de usuário; não foi necessária para executar esta EDA, que usa arquivos públicos arquivados"),
    ]
    return pd.DataFrame(rows, columns=["conjunto", "origem", "frequencia", "periodo", "documentacao_acesso", "observacoes"])


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flag high IQR values for review only; never classify or remove them as errors."""
    rows = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code, ["date", CASE_COL, "surto_mem", "nivel_inc_mem"]].copy()
        q1, q3 = g[CASE_COL].quantile([.25, .75])
        upper = q3 + 1.5 * (q3 - q1)
        flagged = g.loc[g[CASE_COL] > upper]
        for _, r in flagged.iterrows():
            rows.append({
                "geocode_ibge": code,
                "municipio": name,
                "data_semana": r.date,
                "casos_provaveis": r[CASE_COL],
                "limite_superior_iqr_apenas_triagem": upper,
                "nivel_inc_mem": r.get("nivel_inc_mem", np.nan),
                "surto_mem": r.get("surto_mem", np.nan),
                "decisao": "manter; valor sinalizado para conferência, não é erro por ser alto",
            })
    return pd.DataFrame(rows)


def case_version_table(df: pd.DataFrame) -> pd.DataFrame:
    """Audit the API case series against the public historical bulk file."""
    if "casos_arquivo" not in df:
        return pd.DataFrame([{"observacao": "coluna casos_arquivo ausente; comparação não executada"}])
    rows = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code]
        delta = (pd.to_numeric(g[CASE_COL], errors="coerce") -
                 pd.to_numeric(g["casos_arquivo"], errors="coerce")).abs()
        rows.append({
            "geocode_ibge": code, "municipio": name, "n_comparavel": int(delta.notna().sum()),
            "linhas_diferentes": int((delta > 0).sum()),
            "pct_linhas_diferentes": 100 * (delta > 0).mean(),
            "diferenca_absoluta_maxima": delta.max(),
            "soma_casprov_api": g[CASE_COL].sum(),
            "soma_arquivo_historico": g["casos_arquivo"].sum(),
            "decisao": "manter casprov como alvo; registrar que versões públicas são revisadas e auditar novamente ao atualizar a base",
        })
    return pd.DataFrame(rows)


def correlation_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Within-capital correlations, lagged CCF, Granger and climate VIF diagnostics."""
    corr_rows, ccf_rows, granger_rows, vif_rows = [], [], [], []
    feature_labels = {"temp_med": "temperatura", "precip_med": "precipitação", "rel_humid_med": "umidade"}
    matrix_data = {"pearson": [], "spearman": []}
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date").copy()
        values = pd.DataFrame({
            "log1p_casos": np.log1p(g[CASE_COL].clip(lower=0)),
            "temp_med": g["temp_med"], "precip_med": g["precip_med"], "rel_humid_med": g["rel_humid_med"],
        }).dropna()
        for method in ("pearson", "spearman"):
            mat = values.corr(method=method)
            matrix_data[method].append(mat)
            for first in mat.columns:
                for second in mat.columns:
                    if first < second:
                        corr_rows.append({"municipio": name, "metodo": method, "variavel_1": first,
                                          "variavel_2": second, "correlacao": mat.loc[first, second],
                                          "n_pares": len(values),
                                          "interpretacao": "associação dentro da capital; Pearson resume relação linear e Spearman relação monotônica",
                                          "decisao_modelagem": "usar como triagem, não como evidência causal; verificar redundância e validar recursos só no treino"})

        target = np.log1p(g[CASE_COL].clip(lower=0)).reset_index(drop=True)
        for col, label in feature_labels.items():
            climate = pd.to_numeric(g[col], errors="coerce").reset_index(drop=True)
            for lag in range(2, 9):
                pair = pd.concat([target.rename("target"), climate.shift(lag).rename("climate_lag")], axis=1).dropna()
                ccf_rows.append({"geocode_ibge": code, "municipio": name, "variavel_climatica": col,
                                 "rotulo": label, "defasagem_semanas": lag,
                                 "correlacao_cruzada": pair.target.corr(pair.climate_lag), "n_pares": len(pair),
                                 "interpretacao": "associação entre log(1+casos) nesta semana e clima observado na semana indicada; não implica causalidade",
                                 "decisao_modelagem": "considerar como candidata se houver sinal recorrente e se o lag estiver disponível na origem do horizonte; validar no treino"})

            # Seasonal differences reduce annual pattern before the bivariate Granger diagnostic.
            gc = pd.DataFrame({"cases": target.diff(52), "climate": climate.diff(52)}).dropna()
            gc = gc.replace([np.inf, -np.inf], np.nan).dropna()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = grangercausalitytests(gc[["cases", "climate"]], maxlag=8)
                for lag in range(2, 9):
                    fstat, pval, df_denom, df_num = results[lag][0]["ssr_ftest"]
                    granger_rows.append({"geocode_ibge": code, "municipio": name, "variavel_climatica": col,
                                         "defasagem_maxima_semanas": lag, "F_ssr": fstat, "p_valor": pval,
                                         "n_observacoes": len(gc), "significativo_5pct_sem_ajuste": pval < .05,
                                         "interpretacao": "p<0,05 sugere informação preditiva incremental no teste bivariado, sem provar causalidade; são testes exploratórios múltiplos",
                                         "decisao_modelagem": "se p<0,05, manter como candidata exploratória e confirmar por validação temporal; não selecionar por p-valor isolado" if pval < .05 else "não selecionar por este teste isolado; considerar outros sinais e validar no treino"})
            except Exception as exc:
                granger_rows.append({"geocode_ibge": code, "municipio": name, "variavel_climatica": col,
                                     "defasagem_maxima_semanas": np.nan, "F_ssr": np.nan, "p_valor": np.nan,
                                     "n_observacoes": len(gc), "significativo_5pct_sem_ajuste": False,
                                     "interpretacao": f"teste não calculado: {type(exc).__name__}: {exc}",
                                     "decisao_modelagem": "não tomar decisão de modelo até corrigir a falha do teste"})

        # Check multicollinearity on candidate climate inputs; omit the first 8 rows only
        # to align lagged variables. VIF is a diagnostic, not a reason to discard a variable alone.
        predictors = pd.DataFrame(index=g.index)
        for col in feature_labels:
            for lag in (2, 4, 6, 8):
                predictors[f"{col}_lag_{lag}"] = g[col].shift(lag)
        predictors = predictors.replace([np.inf, -np.inf], np.nan).dropna()
        if len(predictors) > predictors.shape[1] + 2:
            x = predictors.astype(float).reset_index(drop=True)
            for i, col in enumerate(x.columns):
                try:
                    vif = float(variance_inflation_factor(x.to_numpy(), i))
                except Exception:
                    vif = np.inf
                vif_rows.append({"geocode_ibge": code, "municipio": name, "variavel": col,
                                 "VIF": vif, "n_observacoes": len(x),
                                 "interpretacao": "VIF acima de 5 é alerta exploratório de redundância; defasagens climáticas próximas tendem a ser correlacionadas",
                                 "decisao_modelagem": "se VIF>5, reduzir lags redundantes em modelos interpretáveis ou regularizar; conferir novamente apenas no treino" if vif > 5 else "sem alerta VIF>5 neste ajuste; ainda verificar seleção no treino"})

    corr = pd.DataFrame(corr_rows)
    ccf = pd.DataFrame(ccf_rows)
    granger = pd.DataFrame(granger_rows)
    vif = pd.DataFrame(vif_rows)
    median_matrices = {}
    for method, matrices in matrix_data.items():
        median_matrices[method] = pd.concat(matrices).groupby(level=0).median().reindex(index=matrices[0].index, columns=matrices[0].columns)
    for method, matrix in median_matrices.items():
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        sns.heatmap(matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="vlag",
                    square=True, cbar_kws={"label": "Correlação"}, ax=ax)
        ax.set_title(f"Matriz {method.title()} — mediana das correlações dentro de cada capital")
        ax.set_xlabel("Variável")
        ax.set_ylabel("Variável")
        fig.tight_layout()
        savefig(fig, f"06_matriz_{method}.png")

    ccf_summary = (ccf.groupby(["variavel_climatica", "rotulo", "defasagem_semanas"], as_index=False)
                   .agg(mediana_correlacao=("correlacao_cruzada", "median"),
                        media_correlacao=("correlacao_cruzada", "mean"),
                        capitais_com_correlacao=("municipio", "nunique")))
    ccf_summary["abs_mediana"] = ccf_summary.mediana_correlacao.abs()
    ccf_summary["melhor_lag_por_mediana"] = False
    for col in ccf_summary.variavel_climatica.unique():
        idx = ccf_summary.loc[ccf_summary.variavel_climatica == col, "abs_mediana"].idxmax()
        ccf_summary.loc[idx, "melhor_lag_por_mediana"] = True
    best_lags = ccf_summary.loc[ccf_summary.melhor_lag_por_mediana].set_index("variavel_climatica").defasagem_semanas.to_dict()
    ccf["melhor_lag_mediano_entre_capitais"] = ccf.apply(
        lambda r: int(r.defasagem_semanas) == int(best_lags.get(r.variavel_climatica, -1)), axis=1
    )
    return corr, ccf, granger, vif, ccf_summary


def correlation_decisions(ccf_summary: pd.DataFrame, granger: pd.DataFrame, vif: pd.DataFrame,
                          corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, sub in ccf_summary.groupby("variavel_climatica"):
        best = sub.loc[sub.abs_mediana.idxmax()]
        hits = int(((granger.variavel_climatica == col) & (granger.p_valor < .05)).sum()) if not granger.empty else 0
        pair = corr.loc[((corr.variavel_1 == "log1p_casos") & (corr.variavel_2 == col)) |
                        ((corr.variavel_2 == "log1p_casos") & (corr.variavel_1 == col))]
        pearson = pair.loc[pair.metodo == "pearson", "correlacao"].median()
        spearman = pair.loc[pair.metodo == "spearman", "correlacao"].median()
        rows.append({"variavel": col, "defasagem_candidata_CCF": int(best.defasagem_semanas),
                     "correlacao_mediana_CCF": best.mediana_correlacao,
                     "pearson_mediana_com_casos": pearson,
                     "spearman_mediana_com_casos": spearman,
                     "testes_Granger_p_menor_0_05_sem_ajuste": hits,
                     "VIF_mediano_nos_lags_2_4_6_8": float(vif.loc[vif.variavel.str.startswith(col + "_"), "VIF"].median()) if not vif.empty else np.nan,
                     "interpretacao_simples": f"O lag {int(best.defasagem_semanas)} tem a maior magnitude de correlação mediana entre as capitais; isso é uma pista, não prova de efeito causal.",
                     "decisao_modelagem": "manter esta defasagem como candidata; selecionar dentro do treino walk-forward. Para horizonte h, usar apenas lag >= h ou clima previsto disponível na origem."})
    return pd.DataFrame(rows)


def plot_correlation_diagnostics(ccf: pd.DataFrame, vif: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes, feature_labels_for_plot().items()):
        part = ccf.loc[ccf.variavel_climatica == col]
        for city, group in part.groupby("municipio"):
            ax.plot(group.defasagem_semanas, group.correlacao_cruzada, marker="o", ms=3, lw=.8, alpha=.55)
        median = part.groupby("defasagem_semanas").correlacao_cruzada.median()
        ax.plot(median.index, median.values, color="#b42318", lw=2.5, marker="o", label="Mediana das capitais")
        ax.axhline(0, color="#667085", lw=.8)
        ax.set_title(label)
        ax.set_xlabel("Defasagem do clima (semanas)")
        ax.set_xticks(range(2, 9))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Correlação com log(1 + casos)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("CCF exploratória por capital: clima anterior e casos atuais", fontsize=13)
    fig.tight_layout()
    savefig(fig, "07_ccf_clima_defasagens.png")

    if not vif.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        order = ["temp_med_lag_2", "temp_med_lag_4", "temp_med_lag_6", "temp_med_lag_8",
                 "precip_med_lag_2", "precip_med_lag_4", "precip_med_lag_6", "precip_med_lag_8",
                 "rel_humid_med_lag_2", "rel_humid_med_lag_4", "rel_humid_med_lag_6", "rel_humid_med_lag_8"]
        values = vif.assign(VIF_plot=vif.VIF.clip(upper=100))
        sns.boxplot(data=values, x="variavel", y="VIF_plot", order=order, color="#84caff", ax=ax)
        ax.axhline(5, color="#d92d20", ls="--", lw=1, label="VIF = 5 (alerta exploratório)")
        ax.set_title("VIF das variáveis climáticas candidatas (corte visual em 100)")
        ax.set_xlabel("Variável e defasagem")
        ax.set_ylabel("VIF (valores acima de 100 mostrados como 100)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        savefig(fig, "08_vif_clima.png")


def feature_labels_for_plot() -> dict[str, str]:
    return {"temp_med": "Temperatura média", "precip_med": "Precipitação média", "rel_humid_med": "Umidade relativa"}


def run_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date")
        series = g[CASE_COL].astype(float).dropna()
        if len(series) < 110:
            continue
        transforms = {
            "log1p_nivel": np.log1p(series.clip(lower=0)),
            "log1p_diferenca_52": np.log1p(series.clip(lower=0)).diff(52).dropna(),
        }
        for transform, values in transforms.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf_stat, adf_p, adf_lags, adf_n, *_ = adfuller(values, autolag="AIC")
            except Exception as exc:
                adf_stat, adf_p, adf_lags, adf_n = np.nan, np.nan, np.nan, len(values)
                adf_error = str(exc)
            else:
                adf_error = ""
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_stat, kpss_p, kpss_lags, _ = kpss(values, regression="c", nlags="auto")
            except Exception as exc:
                kpss_stat, kpss_p, kpss_lags = np.nan, np.nan, np.nan
                kpss_error = str(exc)
            else:
                kpss_error = ""
            adf_stationary = bool(adf_p < .05) if pd.notna(adf_p) else None
            kpss_stationary = bool(kpss_p >= .05) if pd.notna(kpss_p) else None
            if adf_stationary is True and kpss_stationary is True:
                result = "evidência conjunta de estacionariedade"
                decision = "a transformação pode ser usada sem diferença adicional para estacionariedade; manter a sazonalidade explícita"
            elif adf_stationary is False and kpss_stationary is False:
                result = "evidência conjunta de não estacionariedade"
                decision = "avaliar diferença (incluindo sazonal de 52 semanas) dentro de cada janela de treino"
            else:
                result = "testes discordam ou inconclusivos"
                decision = "não diferenciar automaticamente; inspecionar sazonalidade e comparar modelos em validação temporal"
            rows.append({
                "geocode_ibge": code, "municipio": name, "transformacao": transform,
                "n": len(values), "ADF_estatistica": adf_stat, "ADF_pvalor": adf_p,
                "ADF_lags": adf_lags, "ADF_n": adf_n,
                "KPSS_estatistica": kpss_stat, "KPSS_pvalor": kpss_p, "KPSS_lags": kpss_lags,
                "ADF_rejeita_raiz_unitaria_5pct": adf_stationary,
                "KPSS_nao_rejeita_estacionariedade_5pct": kpss_stationary,
                "leitura_simples": result, "decisao_modelagem": decision,
                "erro_ADF": adf_error, "erro_KPSS": kpss_error,
            })
    return pd.DataFrame(rows)


def interpretation_for_test(row: pd.Series) -> str:
    if row.ADF_rejeita_raiz_unitaria_5pct is True and row.KPSS_nao_rejeita_estacionariedade_5pct is True:
        return "Os dois testes são compatíveis com uma série estacionária após esta transformação."
    if row.ADF_rejeita_raiz_unitaria_5pct is False and row.KPSS_nao_rejeita_estacionariedade_5pct is False:
        return "Os dois testes apontam não estacionariedade; avaliar diferenças apenas usando dados de treino."
    return "ADF e KPSS discordam ou não concluíram; não decidir diferenciação por um teste isolado."


def plot_city_series(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for ax, (code, name) in zip(axes.flat, CAPITALS.items()):
        g = df.loc[df.geocode == code]
        ax.plot(g.date, g[CASE_COL], lw=.8, color="#b42318")
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("Casos prováveis/semana")
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Dengue por capital: casos prováveis semanais (2015–base disponível)", fontsize=15)
    fig.supxlabel("Semana epidemiológica (início da semana)")
    fig.tight_layout()
    savefig(fig, "01_series_por_capital.png")


def plot_aggregate(df: pd.DataFrame) -> None:
    agg = df.groupby("date", as_index=False).agg(casos=(CASE_COL, "sum"), populacao=("population", "sum"))
    agg["incidencia_100mil"] = agg.casos / agg.populacao * 100_000
    agg.to_csv(TABLE / "serie_agregada_capitais.csv", index=False, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    ax1.plot(agg.date, agg.casos, color="#b42318", lw=1, label="Casos prováveis")
    ax1.set_ylabel("Casos prováveis/semana", color="#b42318")
    ax1.tick_params(axis="y", labelcolor="#b42318")
    ax1.grid(alpha=.2)
    ax2 = ax1.twinx()
    ax2.plot(agg.date, agg.incidencia_100mil, color="#175cd3", lw=1, alpha=.8, label="Incidência")
    ax2.set_ylabel("Incidência por 100 mil hab./semana", color="#175cd3")
    ax2.tick_params(axis="y", labelcolor="#175cd3")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_title("Série agregada das nove capitais")
    ax1.set_xlabel("Semana epidemiológica")
    fig.tight_layout()
    savefig(fig, "02_agregado_casos_incidencia.png")


def plot_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby(["geocode", "municipio", "epiweek_year", "epiweek_number"], as_index=False)[CASE_COL].mean()
    profile.to_csv(TABLE / "perfil_sazonal_ano_semana.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=False)
    years = sorted(profile.epiweek_year.unique())
    colors = dict(zip(years, sns.color_palette("viridis", len(years))))
    for ax, (code, name) in zip(axes.flat, CAPITALS.items()):
        sub = profile.loc[profile.geocode == code]
        for year, gy in sub.groupby("epiweek_year"):
            label = f"{year}*" if year == max(years) and gy.epiweek_number.max() < 52 else str(year)
            ax.plot(gy.epiweek_number, gy[CASE_COL], lw=.9, alpha=.8, color=colors[year], label=label)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Semana epidemiológica do ano")
        ax.set_ylabel("Casos/semana")
        ax.set_xlim(1, 53)
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Ano", loc="center right", bbox_to_anchor=(1.04, .5), ncol=1, fontsize=8)
    fig.suptitle("Sazonalidade da dengue por ano e capital", fontsize=15)
    fig.text(.42, .005, "* 2026 incompleto (dados até SE 10)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .025, .96, .96))
    savefig(fig, "03_sazonalidade_ano_capital.png")
    return profile


def plot_stl(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date")
        y = np.log1p(g.set_index("date")[CASE_COL].astype(float)).asfreq("W-SUN")
        if y.isna().any() or len(y) < 104:
            rows.append({"geocode_ibge": code, "municipio": name, "status": "não executado: série irregular ou curta"})
            continue
        result = STL(y, period=52, robust=True).fit()
        fig = result.plot()
        fig.set_size_inches(13, 7)
        fig.suptitle(f"STL de log(1 + casos): {name} (período sazonal = 52 semanas)", y=1.02, fontsize=13)
        for ax, label in zip(fig.axes, ["Casos transformados", "Tendência", "Sazonalidade anual", "Resíduo"]):
            ax.set_ylabel(label)
            ax.grid(alpha=.15)
        fig.tight_layout()
        savefig(fig, f"stl_{code}.png")
        trend_start = float(result.trend.iloc[:26].mean())
        trend_end = float(result.trend.iloc[-26:].mean())
        seasonal_week = int(pd.Series(result.seasonal.to_numpy(), index=y.index).groupby(y.index.isocalendar().week).mean().idxmax())
        resid_sd = float(result.resid.std())
        resid_extreme_count = int((result.resid.abs() > 3 * resid_sd).sum()) if resid_sd > 0 else 0
        if trend_end > trend_start * 1.05:
            trend_read = "tendência recente acima do início da série"
        elif trend_end < trend_start * .95:
            trend_read = "tendência recente abaixo do início da série"
        else:
            trend_read = "tendência média inicial e final de magnitude parecida"
        rows.append({
            "geocode_ibge": code, "municipio": name, "status": "ok",
            "observacoes": len(y), "periodo_semanas": 52,
            "media_tendencia_primeiras_26_sem": trend_start,
            "media_tendencia_ultimas_26_sem": trend_end,
            "leitura_tendencia": trend_read,
            "semana_pico_componente_sazonal": seasonal_week,
            "residuos_acima_de_3dp": resid_extreme_count,
            "interpretacao": "A curva de tendência resume mudanças lentas; a sazonalidade de 52 semanas mostra o padrão anual; resíduos retêm variações semanais e picos não explicados. Picos residuais não são automaticamente erros.",
            "decisao_modelagem": "comparar baseline sazonal de 52 semanas e modelos com termos de calendário; manter eventos reais e diferenças locais entre anos",
        })
    result = pd.DataFrame(rows)
    result.to_csv(TABLE / "stl_interpretacao.csv", index=False, encoding="utf-8-sig")
    return result


def plot_acf_pacf(df: pd.DataFrame) -> None:
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date")
        y = np.log1p(g[CASE_COL].astype(float))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
        plot_acf(y, lags=52, ax=axes[0], zero=False, title="ACF — log(1 + casos)", alpha=.05)
        plot_pacf(y, lags=52, ax=axes[1], zero=False, method="ywm", title="PACF — log(1 + casos)", alpha=.05)
        for ax in axes:
            ax.set_xlabel("Defasagem (semanas)")
            ax.grid(alpha=.18)
        fig.suptitle(f"Dependência temporal: {name}", fontsize=13)
        fig.tight_layout()
        savefig(fig, f"acf_pacf_{code}.png")


def graph_interpretations(df: pd.DataFrame, profile: pd.DataFrame,
                          stl: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Machine-readable plain-language note and modeling decision for each figure."""
    from statsmodels.tsa.stattools import acf, pacf

    rows = []
    for code, name in CAPITALS.items():
        g = df.loc[df.geocode == code].sort_values("date")
        y = g[CASE_COL].astype(float)
        peak = g.loc[y.idxmax()]
        rows.append({
            "figura": "01_series_por_capital.png", "municipio": name,
            "leitura_simples": f"A série vai de {int(y.min())} a {int(y.max())} casos por semana; o maior valor observado foi {int(peak[CASE_COL])} em {peak.date:%d/%m/%Y}. Semanas sem casos representam zero observado.",
            "decisao_modelagem": "manter picos observados; incluir defasagens de casos e comparar previsões em escala original e log1p",
        })
        seas = profile.loc[profile.geocode == code]
        peaks = seas.loc[seas.groupby("epiweek_year")[CASE_COL].idxmax()]
        peak_weeks = peaks.epiweek_number.astype(int).tolist()
        median_week = int(np.median(peak_weeks)) if peak_weeks else np.nan
        rows.append({
            "figura": "03_sazonalidade_ano_capital.png", "municipio": name,
            "leitura_simples": f"O pico anual de casos ocorreu em uma semana mediana {median_week}; as semanas dos picos variaram entre os anos: {', '.join(map(str, peak_weeks))}.",
            "decisao_modelagem": "testar sazonalidade anual de 52 semanas e fatores de semana epidemiológica; a variação de pico entre anos impede fixar uma única semana como regra",
        })
        stlrow = stl.loc[stl.geocode_ibge == code]
        if not stlrow.empty and stlrow.iloc[0].get("status") == "ok":
            s = stlrow.iloc[0]
            rows.append({
                "figura": f"stl_{code}.png", "municipio": name,
                "leitura_simples": f"Na escala log1p, a tendência indica {s.leitura_tendencia}; o componente sazonal atinge seu máximo na semana {int(s.semana_pico_componente_sazonal)}. {int(s.residuos_acima_de_3dp)} resíduos excedem 3 desvios-padrão como triagem, sem serem classificados como erro.",
                "decisao_modelagem": "comparar componente sazonal anual e tendência; inspecionar resíduos extremos sem removê-los automaticamente",
            })
        z = np.log1p(y.clip(lower=0))
        acf_values = acf(z, nlags=52, fft=True)[1:]
        pacf_values = pacf(z, nlags=52, method="ywm")[1:]
        threshold = 1.96 / np.sqrt(len(z))
        acf_lags = [i + 1 for i, v in enumerate(acf_values) if abs(v) > threshold]
        pacf_lags = [i + 1 for i, v in enumerate(pacf_values) if abs(v) > threshold]
        rows.append({
            "figura": f"acf_pacf_{code}.png", "municipio": name,
            "leitura_simples": f"Entre as 52 semanas examinadas, ACF excede a banda aproximada em {acf_lags[:12]}; PACF em {pacf_lags[:12]}. A banda é exploratória e não corrige testes múltiplos.",
            "decisao_modelagem": "candidatas iniciais: lags curtos 1–8 e o lag anual 52; confirmar apenas com walk-forward e sem vazamento",
        })
    total = df.groupby("date", as_index=False).agg(casos=(CASE_COL, "sum"), populacao=("population", "sum"))
    total["incidencia_100mil"] = 100_000 * total.casos / total.populacao
    p = total.loc[total.casos.idxmax()]
    rows.extend([
        {"figura": "02_agregado_casos_incidencia.png", "municipio": "Nove capitais (agregado)",
         "leitura_simples": f"O pico agregado foi de {int(p.casos)} casos em {p.date:%d/%m/%Y}; a incidência usa total de casos dividido pela população total, evitando média simples das taxas municipais.",
         "decisao_modelagem": "relatar casos e incidência separadamente; modelar cada cidade e somar previsões para o total regional"},
        {"figura": "04_faltantes_heatmap.png", "municipio": "Nove capitais",
         "leitura_simples": "A cor mostra o percentual ausente por variável e capital; zero observado não conta como ausente.",
         "decisao_modelagem": "na base atual, manter zeros; preencher lacunas apenas após localizar causa e usando treino, com indicador de ausência quando necessário"},
        {"figura": "05_picos_para_revisao.png", "municipio": "Nove capitais",
         "leitura_simples": f"O corte IQR sinalizou {len(flags)} observações altas. Trata-se de uma triagem estatística; surtos podem gerar picos legítimos.",
         "decisao_modelagem": "nenhum pico foi removido; corrigir apenas registro com evidência externa de erro de unidade ou captura"},
        {"figura": "06_matriz_pearson.png", "municipio": "Mediana dentro das capitais",
         "leitura_simples": "Cada correlação é calculada dentro de uma capital; a matriz mostra a mediana entre capitais, evitando que a diferença de escala entre cidades domine a associação.",
         "decisao_modelagem": "usar como triagem de relações lineares; não interpretar correlação como causalidade"},
        {"figura": "06_matriz_spearman.png", "municipio": "Mediana dentro das capitais",
         "leitura_simples": "Spearman resume associação monotônica e é menos sensível a escala e valores extremos que Pearson.",
         "decisao_modelagem": "comparar o sinal e a magnitude com Pearson; qualquer variáveis redundantes serão avaliadas pelo VIF e no treino"},
        {"figura": "07_ccf_clima_defasagens.png", "municipio": "Nove capitais",
         "leitura_simples": "As linhas finas mostram cada capital e a linha destacada mostra a mediana; cada ponto compara clima passado com casos atuais.",
         "decisao_modelagem": "usar o lag de maior magnitude mediana como candidato; confirmar por walk-forward e respeitar a informação disponível na origem"},
        {"figura": "08_vif_clima.png", "municipio": "Nove capitais",
         "leitura_simples": "VIF alto indica que uma variável climática pode ser explicada por outras defasagens climáticas; a caixa resume as cidades.",
         "decisao_modelagem": "reduzir defasagens redundantes em modelos interpretáveis ou usar regularização; validar seleção sem consultar o período de teste"},
    ])
    return pd.DataFrame(rows)


def plot_missingness(missing: pd.DataFrame) -> None:
    pivot = missing.pivot(index="variavel", columns="municipio", values="pct_faltantes").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "% faltantes"}, ax=ax)
    ax.set_title("Percentual de valores faltantes por variável e capital")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Variável")
    fig.tight_layout()
    savefig(fig, "04_faltantes_heatmap.png")


def plot_outliers(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for ax, (code, name) in zip(axes.flat, CAPITALS.items()):
        g = df.loc[df.geocode == code]
        ax.plot(g.date, g[CASE_COL], color="#667085", lw=.7, label="Observado")
        f = flags.loc[flags.geocode_ibge == code] if not flags.empty else flags
        if not f.empty:
            ax.scatter(f.data_semana, f.casos_provaveis, color="#d92d20", s=15, label="Acima do corte IQR (revisar)", zorder=3)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Casos/semana")
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Triagem de valores altos: picos são mantidos, não removidos", fontsize=14)
    fig.supxlabel("Semana epidemiológica")
    fig.tight_layout()
    savefig(fig, "05_picos_para_revisao.png")


def write_report(df: pd.DataFrame, desc: pd.DataFrame, missing: pd.DataFrame,
                 flags: pd.DataFrame, tests: pd.DataFrame, stl: pd.DataFrame,
                 corr_decisions: pd.DataFrame, version_audit: pd.DataFrame) -> None:
    def fmt_pt(value: float, decimals: int = 2) -> str:
        return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    dates = f"{df.date.min():%d/%m/%Y} a {df.date.max():%d/%m/%Y}"
    n_muni = df.geocode.nunique()
    n_rows = len(df)
    unique = df.groupby(["geocode", "date"]).size()
    duplicates = int((unique > 1).sum())
    week_gaps = []
    for code, g in df.groupby("geocode"):
        expected = pd.date_range(g.date.min(), g.date.max(), freq="W-SUN")
        absent = expected.difference(pd.DatetimeIndex(g.date))
        if len(absent):
            week_gaps.append((CAPITALS[code], len(absent)))
    total_missing_cases = int(df[CASE_COL].isna().sum())
    total_missing_climate = int(df[[c for c in CLIMATE_COLS if c in df]].isna().sum().sum())
    all_zeros = (df[CASE_COL] == 0).mean()
    peak_weeks = int(flags.shape[0])
    overlap = (flags.surto_mem == 1).sum() if "surto_mem" in flags else 0
    # Robustly compute how many high IQR flags align with the official MEM indicator.
    if not flags.empty and "surto_mem" in flags:
        overlap = int((pd.to_numeric(flags.surto_mem, errors="coerce") == 1).sum())
    nonzero = df.loc[df[CASE_COL] > 0, CASE_COL]
    if len(nonzero):
        nonzero_q = nonzero.quantile([.5, .9, .99]).to_dict()
    else:
        nonzero_q = {0.5: np.nan, 0.9: np.nan, 0.99: np.nan}
    test_rows = []
    if not tests.empty:
        level = tests.loc[tests.transformacao == "log1p_nivel"]
        stable = int(((level.ADF_rejeita_raiz_unitaria_5pct == True) & (level.KPSS_nao_rejeita_estacionariedade_5pct == True)).sum())
        nonstable = int(((level.ADF_rejeita_raiz_unitaria_5pct == False) & (level.KPSS_nao_rejeita_estacionariedade_5pct == False)).sum())
        test_rows.append(f"Na série log(1+casos), ADF e KPSS concordam em estacionariedade em {stable}/{len(level)} capitais; concordam em não estacionariedade em {nonstable}/{len(level)}. Nas demais, os resultados são discordantes/inconclusivos. Veja os p-valores e decisões por capital na tabela de testes.")
    else:
        test_rows.append("Os testes não produziram resultados suficientes; confira a tabela de testes.")
    climate_missing_pct = 100 * total_missing_climate / max(1, len(df) * len([c for c in CLIMATE_COLS if c in df]))
    lines = [
        "# Análise exploratória — dengue nas capitais do Nordeste",
        "",
        "## Base e escopo",
        "",
        f"A análise filtra a base pública InfoDengue/Mosqlimate arquivada localmente para as nove capitais. O recorte tem **{n_muni} capitais**, **{n_rows:,} linhas** e período de **{dates}**. A unidade é capital × semana epidemiológica. Há {duplicates} chaves capital-semana duplicadas.".replace(f"{n_rows:,}", f"{n_rows:,}".replace(",", ".")),
        "",
        "O alvo é `casprov`, casos prováveis semanais. Incidência por 100 mil habitantes é apresentada como medida complementar. O rótulo `nivel_inc_mem == 2` indica incidência acima do limiar epidêmico do Moving Epidemics Method (MEM), conforme a documentação do InfoDengue/Mosqlimate. Os dados deste arquivo encerram em 8 de março de 2026; as análises não descrevem semanas posteriores.",
        "",
        "### Origem dos dados",
        "",
        "O arquivo público de dengue documenta casos prováveis como suspeitos menos descartados; o clima vem da reanálise ERA5/Copernicus organizada pelo Mosqlimate, e a população anual é de DATASUS/SVS. Links e períodos por fonte estão em `tables/fontes_e_periodos.csv`; as variáveis e unidades estão em `tables/dicionario_variaveis.csv`. A documentação oficial da base está em [Data documentation — 3rd IMDC](https://sprint.mosqlimate.org/data/), e a consulta municipal InfoDengue em [serviços da API](https://info.dengue.mat.br/services/api).",
        "",
        "## Estatísticas descritivas",
        "",
        f"A proporção agregada de semanas com zero casos é {fmt_pt(100 * all_zeros, 1)}%. Entre as semanas com valor positivo, mediana={fmt_pt(nonzero_q[.5], 1)}, P90={fmt_pt(nonzero_q[.9], 1)} e P99={fmt_pt(nonzero_q[.99], 1)} casos. As estatísticas por capital, incluindo dispersão, máximos, zeros, população e clima, estão em `tables/estatisticas_por_capital.csv`.",
        "",
        "**Leitura para a modelagem:** as séries têm assimetria e picos; a mediana, os quantis e a proporção de zeros devem ser considerados junto com a média. Contagens e incidência respondem a perguntas diferentes, então as previsões de casos serão convertidas em incidência para analisar o limiar.",
        "",
        "## Séries, agregado e sazonalidade",
        "",
        "`figures/01_series_por_capital.png` mostra a trajetória semanal de cada capital. `figures/02_agregado_casos_incidencia.png` compara o total de casos e a incidência agregada, calculada como soma de casos dividida pela soma da população. `figures/03_sazonalidade_ano_capital.png` sobrepõe os perfis por semana epidemiológica e ano para distinguir um padrão anual recorrente de anos atípicos.",
        "",
        "**Leitura para a modelagem:** picos sazonais recorrentes justificam um baseline sazonal de 52 semanas e variáveis de calendário. Diferenças entre capitais e entre anos apoiam modelos globais com identificador de cidade e validação temporal; não se deve pressupor um único perfil sazonal idêntico.",
        "",
        "## Decomposição STL",
        "",
        "A STL usa `log(1 + casos)` e período de 52 semanas para cada capital. Cada arquivo `figures/stl_<geocódigo>.png` separa tendência, sazonalidade anual e resíduo. A transformação reduz a influência visual de picos muito grandes, mas não os remove da base.",
        "",
        "**Interpretação simples:** tendência resume mudanças graduais; sazonalidade resume o padrão que tende a reaparecer a cada ano; resíduo contém flutuações e eventos que esse padrão não explica. Um pico no resíduo pode ser um surto real. A decisão é comparar modelos com sazonalidade anual e manter semanas de pico para o treino e a avaliação.",
        "",
        "## Estacionariedade: ADF e KPSS",
        "",
        *test_rows,
        "",
        "ADF tem como hipótese nula uma raiz unitária; p<0,05 é evidência contra essa hipótese. KPSS tem como hipótese nula estacionariedade; p<0,05 é evidência contra estacionariedade. Por isso, a leitura conjunta ajuda a separar concordância de discordância. A tabela `tables/testes_estacionariedade.csv` inclui estatísticas, defasagens usadas, p-valores, interpretação e decisão para `log1p_nivel` e `log1p_diferenca_52` em cada capital.",
        "",
        "**Decisão para a modelagem:** a necessidade de diferenciação não será decidida olhando a série de teste. Transformações e diferenças serão ajustadas dentro de cada janela de treino; modelos também serão comparados sem diferença quando os testes discordarem.",
        "",
        "## Correlação, CCF, Granger e VIF",
        "",
        "As matrizes Pearson e Spearman são calculadas separadamente dentro de cada capital; as figuras mostram a mediana entre cidades e as tabelas mantêm os resultados individuais. A CCF compara log(1+casos) na semana atual com clima observado 2–8 semanas antes. O Granger usa diferenças sazonais de 52 semanas e testa se o clima melhora a previsão bivariada de casos. Os p-valores de Granger não têm correção por múltiplos testes e não demonstram causalidade.",
        "",
        "As figuras `06_matriz_pearson.png`, `06_matriz_spearman.png`, `07_ccf_clima_defasagens.png` e `08_vif_clima.png` e as tabelas `correlacoes_por_capital.csv`, `ccf_por_capital.csv`, `granger_por_capital.csv`, `vif_por_capital.csv` e `decisao_variaveis_defasagens.csv` documentam essas análises.",
        "",
        "Candidatas que resultam da triagem (uma por variável climática):",
        "",
        *[f"- `{r.variavel}`: Pearson mediano {fmt_pt(r.pearson_mediana_com_casos, 3)} e Spearman mediano {fmt_pt(r.spearman_mediana_com_casos, 3)}; lag CCF candidato {int(r.defasagem_candidata_CCF)} semanas (mediana {fmt_pt(r.correlacao_mediana_CCF, 3)}); {int(r.testes_Granger_p_menor_0_05_sem_ajuste)} testes Granger com p<0,05 sem ajuste; VIF mediano {fmt_pt(r.VIF_mediano_nos_lags_2_4_6_8, 1)}." for r in corr_decisions.itertuples()],
        "",
        "**Conclusão para a modelagem:** as defasagens climáticas com maior |CCF| mediana ficam registradas como candidatas, junto com o número de testes Granger significativos e o VIF mediano. Isso não autoriza escolher variáveis olhando o teste. A seleção final ocorre dentro do treino walk-forward; para um horizonte h, clima só entra se o lag for ≥ h ou se houver previsão climática disponível na origem.",
        "",
        "### Diferença entre versões públicas de casos",
        "",
        f"A série `casprov` da extração InfoDengue e o arquivo histórico público têm {int((df[CASE_COL] != df['casos_arquivo']).sum())} diferenças em {len(df)} semanas-capital ({fmt_pt(100 * (df[CASE_COL] != df['casos_arquivo']).mean(), 2)}%); maior diferença absoluta={float((df[CASE_COL] - df['casos_arquivo']).abs().max()):.0f} casos. A auditoria por cidade está em `tables/auditoria_versoes_casos.csv`. Mantive `casprov` como alvo por consistência com a extração usada no alerta MEM e registrarei a data da extração; as diferenças refletem revisão/versão, não dados sintéticos.",
        "",
        "## ACF e PACF",
        "",
        "Os arquivos `figures/acf_pacf_<geocódigo>.png` mostram a autocorrelação (ACF) e a autocorrelação parcial (PACF) de `log(1 + casos)` até 52 semanas. Picos próximos a 52 semanas são compatíveis com dependência anual; picos nas primeiras semanas sugerem persistência de curto prazo. ACF/PACF orientam defasagens candidatas, mas não selecionam o modelo sozinhas.",
        "",
        "**Decisão para a modelagem:** testar lags de casos de 1 a 8 semanas e 52 semanas como candidatos; a validação walk-forward decide quais ajudam. Para previsão de 4 semanas, recursos só entram quando seriam conhecidos na origem da previsão.",
        "",
        "## Faltantes e outliers",
        "",
        f"Há {total_missing_cases} valores faltantes no alvo. Nas variáveis climáticas avaliadas, há {total_missing_climate} células ausentes ({fmt_pt(climate_missing_pct, 3)}% das células clima-capital-semana). A tabela `tables/faltantes_por_variavel_capital.csv` localiza ausências e distingue faltantes de zeros observados. Semanas ausentes não serão convertidas automaticamente em zero; se houver faltante climático, serão avaliados indicadores de ausência e imputação temporal/por série apenas no treino, com análise de sensibilidade.",
        "",
        f"A regra IQR sinalizou {peak_weeks} valores altos para inspeção; {overlap} deles também coincidem com o indicador de surto MEM quando disponível. A lista está em `tables/picos_sinalizados_para_revisao.csv`. A regra estatística é apenas uma triagem: **nenhum pico foi removido ou winsorizado**. Só corrigiremos um registro se houver evidência verificável de erro de origem ou unidade. Assim, surtos observados permanecem na análise.",
        "",
        "O gráfico `figures/05_picos_para_revisao.png` identifica os pontos sinalizados e os mantém desenhados. Extremos meteorológicos também serão conferidos com os intervalos plausíveis e com os valores da fonte antes de qualquer correção; não será feita exclusão automática.",
        "",
        "## Limitações que afetam a leitura",
        "",
        "A base é histórica e pode ser revisada pela fonte; termina em março de 2026. Casos prováveis são suspeitos menos descartados, agregados pela semana epidemiológica de início dos sintomas. Clima é reanálise horária ERA5 resumida por município e semana, não observação independente de uma estação por capital. `precip_med` é a taxa média semanal de precipitação em mm/h; ela não é precipitação total semanal. População vem da série municipal anual do DATASUS, carregada para frente sem usar anos futuros.",
        "",
        "A documentação do pacote público de dados 3rd IMDC 2026 especifica os arquivos de dengue, clima ERA5 e população; a documentação da API Mosqlimate indica chave de usuário para a consulta ao datastore. A API antiga `alertcity` respondeu a uma consulta de teste para São Luís, mas a EDA usa os dados já baixados e mantém todos os arquivos originais no projeto.",
        "",
        "## Arquivos",
        "",
        "- Tabelas: `tables/fontes_e_periodos.csv`, `tables/dicionario_variaveis.csv`, `tables/estatisticas_por_capital.csv`, `tables/faltantes_por_variavel_capital.csv`, `tables/picos_sinalizados_para_revisao.csv`, `tables/testes_estacionariedade.csv`, `tables/stl_interpretacao.csv`, `tables/serie_agregada_capitais.csv`, `tables/perfil_sazonal_ano_semana.csv`, `tables/interpretacao_graficos.csv`, `tables/correlacoes_por_capital.csv`, `tables/ccf_por_capital.csv`, `tables/granger_por_capital.csv`, `tables/vif_por_capital.csv`, `tables/decisao_variaveis_defasagens.csv` e `tables/auditoria_versoes_casos.csv`.",
        "- Figuras: séries por cidade, agregado, sazonalidade, mapa de faltantes, triagem de picos, matrizes Pearson/Spearman, CCF, VIF, nove STL e nove painéis ACF/PACF.",
    ]
    (OUT / "interpretacao_eda.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT, help="Caminho para o painel parquet real tratado")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.9)
    plt.rcParams["font.family"] = "DejaVu Sans"

    df = load_panel(args.input)
    if df.duplicated(["geocode", "date"]).any():
        raise ValueError("Há município-semana duplicado na base de entrada; corrigir antes da EDA.")
    available = set(df.geocode.unique())
    missing_caps = set(CAPITALS) - available
    if missing_caps:
        raise ValueError(f"Capitais ausentes no painel: {sorted(missing_caps)}")

    desc = descriptive_table(df)
    missing = missingness_table(df)
    flags = outlier_table(df)
    tests = run_stationarity(df)
    version_audit = case_version_table(df)
    corr, ccf, granger, vif, ccf_summary = correlation_eda(df)
    corr_decisions = correlation_decisions(ccf_summary, granger, vif, corr)
    desc.to_csv(TABLE / "estatisticas_por_capital.csv", index=False, encoding="utf-8-sig")
    variable_dictionary().to_csv(TABLE / "dicionario_variaveis.csv", index=False, encoding="utf-8-sig")
    source_register().to_csv(TABLE / "fontes_e_periodos.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(TABLE / "faltantes_por_variavel_capital.csv", index=False, encoding="utf-8-sig")
    flags.to_csv(TABLE / "picos_sinalizados_para_revisao.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(TABLE / "testes_estacionariedade.csv", index=False, encoding="utf-8-sig")
    version_audit.to_csv(TABLE / "auditoria_versoes_casos.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(TABLE / "correlacoes_por_capital.csv", index=False, encoding="utf-8-sig")
    ccf.to_csv(TABLE / "ccf_por_capital.csv", index=False, encoding="utf-8-sig")
    ccf_summary.to_csv(TABLE / "ccf_resumo_medianas.csv", index=False, encoding="utf-8-sig")
    granger.to_csv(TABLE / "granger_por_capital.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(TABLE / "vif_por_capital.csv", index=False, encoding="utf-8-sig")
    corr_decisions.to_csv(TABLE / "decisao_variaveis_defasagens.csv", index=False, encoding="utf-8-sig")

    plot_city_series(df)
    plot_aggregate(df)
    profile = plot_seasonality(df)
    stl = plot_stl(df)
    plot_acf_pacf(df)
    plot_missingness(missing)
    plot_outliers(df, flags)
    plot_correlation_diagnostics(ccf, vif)
    graph_interpretations(df, profile, stl, flags).to_csv(
        TABLE / "interpretacao_graficos.csv", index=False, encoding="utf-8-sig"
    )
    write_report(df, desc, missing, flags, tests, stl, corr_decisions, version_audit)

    print(f"Painel: {len(df):,} linhas; {df.geocode.nunique()} capitais; {df.date.min():%Y-%m-%d} a {df.date.max():%Y-%m-%d}")
    print(f"Alvo: {CASE_COL} (casos prováveis por semana); zeros observados: {(df[CASE_COL] == 0).mean():.2%}")
    print(f"Faltantes no alvo: {df[CASE_COL].isna().sum()}; faltantes clima: {df[[c for c in CLIMATE_COLS if c in df]].isna().sum().sum()}")
    print(f"Picos sinalizados apenas para revisão e mantidos: {len(flags)}")
    print(f"Matrizes/correlações: {len(corr)} células cidade-método; CCF: {len(ccf)} combinações; Granger: {len(granger)} testes")
    print(f"Resultados salvos em: {OUT}")


if __name__ == "__main__":
    main()
