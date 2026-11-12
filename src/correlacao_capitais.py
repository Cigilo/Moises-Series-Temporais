"""Análises de correlação para o painel semanal de dengue nas capitais do NE."""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "painel_capitais_modelagem.parquet"
OUT = ROOT / "reports" / "correlacao_capitais"
TABLE = OUT / "tables"
FIG = OUT / "figures"
CLIMATE = {
    "Temperatura": ("temp_med", "°C", False),
    "Precipitação": ("precip_tot_mm", "mm/semana", True),
    "Umidade relativa": ("rel_humid_med", "%", False),
}
CITY_NAMES = {
    "2111300": "São Luís", "2211001": "Teresina", "2304400": "Fortaleza",
    "2408102": "Natal", "2507507": "João Pessoa", "2611606": "Recife",
    "2704302": "Maceió", "2800308": "Aracaju", "2927408": "Salvador",
}


def prepare() -> pd.DataFrame:
    d = pd.read_parquet(DATA).sort_values(["geocode", "date"]).copy()
    d["casos_log"] = np.log1p(d.casprov.clip(lower=0))
    d["chuva_log"] = np.log1p(d.precip_tot_mm.clip(lower=0))
    # Remoção de climatologia semanal por capital para controlar o ciclo anual
    # nas análises associativas. É descritiva, não uma transformação do modelo.
    transforms = {"casprov": "casos_log", "temp_med": "temp_med",
                  "precip_tot_mm": "chuva_log", "rel_humid_med": "rel_humid_med"}
    for name, source in transforms.items():
        climatology = d.groupby(["geocode", "epiweek_number"])[source].transform("mean")
        d[f"anom_{name}"] = d[source] - climatology
    return d


def corr_matrices(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = pd.DataFrame({
        "log1p(casos)": d.casos_log,
        "incidência/100 mil": d.incidencia_100mil,
        "temperatura (°C)": d.temp_med,
        "log1p(chuva mm)": d.chuva_log,
        "umidade (%)": d.rel_humid_med,
        "dias chuvosos": d.rainy_days,
    })
    anomalies = pd.DataFrame({
        "anomalia log1p(casos)": d.anom_casprov,
        "anomalia temperatura": d.anom_temp_med,
        "anomalia log1p(chuva)": d.anom_precip_tot_mm,
        "anomalia umidade": d.anom_rel_humid_med,
    })
    return {
        "pearson_bruta": raw.corr(method="pearson"),
        "spearman_bruta": raw.corr(method="spearman"),
        "pearson_anomalias": anomalies.corr(method="pearson"),
        "spearman_anomalias": anomalies.corr(method="spearman"),
    }


def ccf_table(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for city, g in d.groupby("geocode", sort=False):
        for label, (col, unit, log_rain) in CLIMATE.items():
            x_name = "anom_" + col
            y = g.anom_casprov.to_numpy(float)
            x = g[x_name].to_numpy(float)
            for lag in range(2, 9):
                # Correlação entre casos na semana t e clima observado em t-lag.
                yy, xx = y[lag:], x[:-lag]
                mask = np.isfinite(yy) & np.isfinite(xx)
                yy, xx = yy[mask], xx[mask]
                if len(yy) < 30 or np.std(xx) == 0 or np.std(yy) == 0:
                    pr = sr = np.nan
                else:
                    pr = float(pearsonr(xx, yy).statistic)
                    sr = float(spearmanr(xx, yy).statistic)
                rows.append({"geocode": city, "capital": CITY_NAMES.get(city, city),
                             "variavel_climatica": label, "defasagem_semanas": lag,
                             "pearson_anomalias": pr, "spearman_anomalias": sr,
                             "n_pares": len(yy), "unidade_clima": unit})
    table = pd.DataFrame(rows)
    summary = table.groupby(["variavel_climatica", "defasagem_semanas"], as_index=False).agg(
        pearson_mediano_capitais=("pearson_anomalias", "median"),
        pearson_medio_capitais=("pearson_anomalias", "mean"),
        abs_pearson_mediano_capitais=("pearson_anomalias", lambda x: x.abs().median()),
        spearman_mediano_capitais=("spearman_anomalias", "median"),
        capitais_mesmo_sinal=("pearson_anomalias", lambda x: max((x > 0).sum(), (x < 0).sum())),
        n_capitais=("capital", "nunique"),
    )
    return table, summary


def granger_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in d.groupby("geocode", sort=False):
        for label, (col, _, _) in CLIMATE.items():
            pair = g[["anom_casprov", "anom_" + col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 100:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tests = grangercausalitytests(pair, maxlag=8)
                for lag, result in tests.items():
                    fstat, pval, _, _ = result[0]["ssr_ftest"]
                    rows.append({"geocode": city, "capital": CITY_NAMES.get(city, city),
                                 "variavel_climatica": label, "defasagem_maxima": lag,
                                 "estatistica_F": float(fstat), "p_valor": float(pval),
                                 "n_semanas": len(pair)})
            except (ValueError, np.linalg.LinAlgError) as exc:
                rows.append({"geocode": city, "capital": CITY_NAMES.get(city, city),
                             "variavel_climatica": label, "defasagem_maxima": np.nan,
                             "estatistica_F": np.nan, "p_valor": np.nan,
                             "n_semanas": len(pair), "erro": str(exc)[:200]})
    result = pd.DataFrame(rows)
    result["p_ajustado_BH"] = np.nan
    valid = result.p_valor.notna()
    if valid.any():
        result.loc[valid, "p_ajustado_BH"] = multipletests(result.loc[valid, "p_valor"], method="fdr_bh")[1]
    result["significativo_5pct_BH"] = result.p_ajustado_BH < 0.05
    return result


def vif_table(d: pd.DataFrame) -> pd.DataFrame:
    x = d[["geocode", "date", "casprov", "temp_med", "precip_tot_mm", "rel_humid_med",
           "epiweek_number"]].copy()
    x["casos_log"] = np.log1p(x.casprov.clip(lower=0))
    for lag in range(1, 9):
        x[f"casos_lag_{lag}"] = x.groupby("geocode").casos_log.shift(lag)
    x["casos_lag_52"] = x.groupby("geocode").casos_log.shift(52)
    for col in ["temp_med", "precip_tot_mm", "rel_humid_med"]:
        values = np.log1p(x[col].clip(lower=0)) if col == "precip_tot_mm" else x[col]
        for lag in range(2, 9):
            x[f"{col}_lag_{lag}"] = values.groupby(x.geocode).shift(lag)
    x["semana_sin"] = np.sin(2 * np.pi * x.epiweek_number / 52.18)
    x["semana_cos"] = np.cos(2 * np.pi * x.epiweek_number / 52.18)
    features = [c for c in x.columns if c.startswith("casos_lag_") or
                any(c.startswith(k + "_lag_") for k in ["temp_med", "precip_tot_mm", "rel_humid_med"])] + ["semana_sin", "semana_cos"]
    dummies = pd.get_dummies(x.geocode, prefix="capital", drop_first=True, dtype=float)
    mat = pd.concat([x[features].astype(float), dummies], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    # Centralizar/escalar facilita a estabilidade numérica; VIF não muda com escala linear.
    mat = (mat - mat.mean()) / mat.std(ddof=0).replace(0, 1)
    rows = []
    for i, col in enumerate(mat.columns):
        try:
            value = float(variance_inflation_factor(mat.to_numpy(), i))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            value = np.inf
        rows.append({"variavel": col, "VIF": value,
                     "classificacao": "alto (>10)" if value > 10 else ("moderado (5–10)" if value > 5 else "baixo (≤5)"),
                     "n_linhas": len(mat)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def save_figures(matrices: dict[str, pd.DataFrame], ccf: pd.DataFrame,
                 ccf_summary: pd.DataFrame, granger: pd.DataFrame, vif: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font_scale=.85)
    for key in ["pearson_bruta", "spearman_bruta", "pearson_anomalias", "spearman_anomalias"]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(matrices[key], cmap="vlag", center=0, vmin=-1, vmax=1, annot=True,
                    fmt=".2f", square=True, ax=ax, cbar_kws={"label": "Coeficiente de correlação"})
        ax.set_title(key.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(FIG / f"{key}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # CCF is shown as small multiples with the capital-to-capital spread.
    # This makes both the direction and consistency of the association visible.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    palette = {"Temperatura": "#175cd3", "Precipitação": "#039855", "Umidade relativa": "#7f56d9"}
    for ax, name in zip(axes, CLIMATE):
        subset = ccf.loc[ccf.variavel_climatica == name]
        stats = subset.groupby("defasagem_semanas").pearson_anomalias.agg(
            mediana="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
        color = palette[name]
        ax.fill_between(stats.index, stats.q25, stats.q75, color=color, alpha=.18,
                        label="Faixa central de 50% das capitais")
        ax.plot(stats.index, stats.mediana, color=color, marker="o", lw=2.5,
                label="Mediana das capitais")
        ax.axhline(0, color="#344054", lw=1)
        ax.set_title(name)
        ax.set_xlabel("Clima observado quantas semanas antes")
        ax.set_xticks(range(2, 9))
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Correlação com os casos (−1 a +1)")
    axes[-1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("Clima anterior e dengue depois: associação por intervalo", y=1.03, fontsize=14)
    fig.text(.5, -.03,
             "Cada ponto compara anomalias: semanas mais quentes/chuvosas/úmidas que o normal e casos acima do normal.\n"
             "A faixa mostra a variação entre capitais; associação não demonstra causa nem melhora da previsão.",
             ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "ccf_anomalias_por_lag.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # Direct time-series comparison on one understandable scale. Each city is
    # seasonally adjusted and standardized first; lines are city medians and
    # four-week trailing means are only a visual aid, never model inputs.
    d = prepare()
    climate_series = {
        "Temperatura": ("anom_temp_med", "°C"),
        "Precipitação": ("anom_precip_tot_mm", "log(1 + mm/semana)"),
        "Umidade relativa": ("anom_rel_humid_med", "pontos percentuais"),
    }
    city_scaled = pd.DataFrame({"date": d.date, "geocode": d.geocode,
                                "Dengue": d.anom_casprov})
    city_scaled["Dengue"] = city_scaled.groupby("geocode").Dengue.transform(
        lambda s: s / s.std(ddof=0) if s.std(ddof=0) else s)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, sharey=True)
    for ax, (label, (col, unit)) in zip(axes, climate_series.items()):
        pair = d[["date", "geocode", col]].copy()
        pair["clima"] = pair.groupby("geocode")[col].transform(
            lambda s: s / s.std(ddof=0) if s.std(ddof=0) else s)
        climate_median = pair.groupby("date").clima.median().rolling(4, min_periods=1).mean()
        dengue_median = city_scaled.groupby("date").Dengue.median().rolling(4, min_periods=1).mean()
        ax.plot(dengue_median.index, dengue_median, color="#b42318", lw=1.5,
                label="Dengue (anomalia padronizada)")
        ax.plot(climate_median.index, climate_median, color=palette[label], lw=1.5,
                label=f"{label} (anomalia padronizada)")
        ax.axhline(0, color="#667085", lw=.8)
        ax.set_title(label)
        ax.set_ylabel("Desvios do padrão\n(z-score)")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Semana epidemiológica")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Como clima e dengue variam juntos ao longo do tempo", y=1.01, fontsize=14)
    fig.text(.5, -.005,
             "Cada linha resume a mediana das nove capitais. Zero = padrão sazonal típico; +1 = um desvio-padrão acima do normal.\n"
             "Média móvel de 4 semanas apenas para facilitar a leitura; não é usada como dado de modelagem.",
             ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "comparacao_series_clima_dengue.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    gr = granger.copy()
    gr["score"] = -np.log10(gr.p_ajustado_BH.clip(lower=1e-12))
    gr["linha"] = gr.capital + " — " + gr.variavel_climatica
    pivot = gr.pivot_table(index="linha", columns="defasagem_maxima", values="score")
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(pivot, cmap="mako", ax=ax, cbar_kws={"label": "−log10(p ajustado BH)"})
    ax.set(title="Granger: força da evidência após ajuste por testes múltiplos",
           xlabel="Defasagem máxima testada (semanas)", ylabel="Capital e variável climática")
    fig.tight_layout(); fig.savefig(FIG / "granger_p_ajustado.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    top = vif.head(20).sort_values("VIF")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.variavel, np.minimum(top.VIF, 100), color="#175cd3")
    ax.axvline(5, color="#f79009", ls="--", label="VIF = 5")
    ax.axvline(10, color="#d92d20", ls=":", label="VIF = 10")
    ax.set(title="20 maiores VIFs entre lags candidatos", xlabel="VIF (valores acima de 100 limitados no gráfico)", ylabel="Variável")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "vif_top20.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def write_interpretation(ccf_summary: pd.DataFrame, granger: pd.DataFrame, vif: pd.DataFrame,
                         matrices: dict[str, pd.DataFrame], ccf: pd.DataFrame) -> None:
    best = ccf_summary.loc[ccf_summary.groupby("variavel_climatica").abs_pearson_mediano_capitais.idxmax()]
    lines = [
        "# Correlação e seleção de defasagens — nove capitais do Nordeste", "",
        "## Como ler esta análise", "",
        "Pearson mede associação linear; Spearman mede associação monotônica por postos e é menos sensível a valores extremos. As matrizes brutas misturam cidades, níveis e sazonalidade e devem ser lidas como descrição, não como efeito causal. As matrizes de anomalias subtraem, em cada capital, a média histórica daquela semana epidemiológica; isso reduz a influência do ciclo anual e das diferenças de clima entre capitais.", "",
        "A CCF associa casos na semana t ao clima em t−k, para k=2…8. A anomalia de casos é log1p(casos) menos a climatologia da capital e da semana do ano. Granger pergunta se lags climáticos acrescentam informação preditiva além da sequência de casos; não prova causalidade. Os p-valores de Granger foram ajustados por Benjamini–Hochberg (BH) sobre todas as combinações de capital, clima e lag.", "",
        "## CCF e defasagens candidatas", "",
        "A figura `ccf_anomalias_por_lag.png` mostra uma curva por clima: cada ponto é a mediana da correlação entre capitais e a faixa sombreada contém os 50% centrais das capitais. Valores perto de zero indicam associação linear pequena. A figura `comparacao_series_clima_dengue.png` coloca as anomalias padronizadas ao longo do tempo na mesma escala; ela ajuda a comparar quando cada série ficou acima ou abaixo do seu padrão sazonal, mas não prova causalidade.", "",
    ]
    for _, r in best.iterrows():
        label = r.variavel_climatica
        sign = "positiva" if r.pearson_mediano_capitais >= 0 else "negativa"
        consistent = int(r.capitais_mesmo_sinal)
        lines.append(f"- **{label}:** maior associação mediana em valor absoluto no lag {int(r.defasagem_semanas)} semanas (Pearson mediano={r.pearson_mediano_capitais:.3f}, sinal {sign}; {consistent}/9 capitais compartilham o sinal majoritário). Esta é uma defasagem candidata, não uma relação causal nem uma seleção automática.")
    significant = granger[granger.significativo_5pct_BH]
    lines += ["", "## Granger", ""]
    if significant.empty:
        lines.append("Nenhuma combinação capital–variável–lag permaneceu significativa a 5% após o ajuste BH. A evidência não sustenta retirar ou incluir uma variável apenas pelo teste; mantemos o clima como candidato e decidimos pelo desempenho temporal fora da amostra.")
    else:
        lines.append(f"{len(significant)} combinações capital–variável–lag foram significativas após BH. A concentração e os lags aparecem na tabela `granger_causalidade.csv`; resultados localizados não justificam extrapolar o efeito a todas as capitais.")
    lines += ["", "## Multicolinearidade (VIF)", ""]
    n5 = int((vif.VIF > 5).sum()); n10 = int((vif.VIF > 10).sum())
    lines.append(f"Entre {len(vif)} preditores candidatos, {n5} têm VIF acima de 5 e {n10} acima de 10. VIF alto indica que vários lags repetem informação; por isso não se deve colocar todos os lags climáticos simultaneamente numa regressão interpretável. Os VIFs e o gráfico `vif_top20.png` identificam redundância entre lags de casos, clima e calendário.")
    lines += ["", "## Decisão para a modelagem", ""]
    lines.append("A conclusão para a especificação parcimoniosa é manter lags de casos recentes e de 52 semanas, além da sazonalidade anual de calendário. Entre as variáveis meteorológicas, somente temperatura mostra associação consistente, ainda que fraca (Pearson mediano perto de 0,13); proponho temperatura no lag 2 como candidato mínimo para horizontes curtos e aumentar o lag conforme o horizonte para garantir que o dado já estivesse observado. Precipitação e umidade têm associações próximas de zero, não passam Granger após ajuste BH e devem ficar fora da especificação parcimoniosa inicial. A comparação das séries temporais é descritiva; a decisão final sobre incluir clima depende do desempenho no walk-forward. O primeiro backtest foi executado antes desta análise formal e incluía chuva e umidade nos modelos globais; ele deve ser tratado como referência. Na próxima revisão, comparar no mesmo walk-forward: (a) casos/calendário, (b) casos/calendário + temperatura, e (c) todos os climas. VIF alto (incluindo 22 variáveis acima de 10) recomenda não incluir vários lags redundantes simultaneamente. A decisão definitiva é pelo desempenho temporal fora da amostra, com os recursos limitados ao que existia na origem.")
    lines += ["", "As matrizes, coeficientes e testes foram calculados sobre o histórico completo apenas para descrição e geração de hipóteses. Eles não alteram o corte temporal do backtest. Correlação e Granger não estabelecem causalidade.", ""]
    (OUT / "interpretacao_correlacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Painel ausente: execute src/fetch_capitais_model_data.py primeiro ({DATA})")
    TABLE.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    d = prepare()
    matrices = corr_matrices(d)
    for name, matrix in matrices.items():
        matrix.to_csv(TABLE / f"matriz_{name}.csv", encoding="utf-8-sig")
    ccf, ccf_summary = ccf_table(d)
    granger = granger_table(d)
    vif = vif_table(d)
    ccf.to_csv(TABLE / "ccf_por_capital.csv", index=False, encoding="utf-8-sig")
    ccf_summary.to_csv(TABLE / "ccf_resumo_por_lag.csv", index=False, encoding="utf-8-sig")
    granger.to_csv(TABLE / "granger_causalidade.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(TABLE / "vif_lags_candidatos.csv", index=False, encoding="utf-8-sig")
    save_figures(matrices, ccf, ccf_summary, granger, vif)
    write_interpretation(ccf_summary, granger, vif, matrices, ccf)
    print(f"Painel: {d.geocode.nunique()} capitais, {d.date.nunique()} semanas, {len(d):,} linhas")
    print("Defasagens CCF com maior |Pearson mediano|:")
    print(ccf_summary.loc[ccf_summary.groupby("variavel_climatica").abs_pearson_mediano_capitais.idxmax(),
                          ["variavel_climatica", "defasagem_semanas", "pearson_mediano_capitais", "capitais_mesmo_sinal"]].to_string(index=False))
    print(f"Granger significativo após BH: {int(granger.significativo_5pct_BH.sum())} / {granger.p_valor.notna().sum()}")
    print(f"VIF > 5: {int((vif.VIF > 5).sum())}/{len(vif)}; VIF > 10: {int((vif.VIF > 10).sum())}/{len(vif)}")
    print(f"Saídas: {OUT}")


if __name__ == "__main__":
    main()
