"""Compara especificações climáticas da Binomial Negativa no mesmo walk-forward."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model_capitais as mc  # noqa: E402

OUT = ROOT / "reports" / "modelagem_capitais"
TABLE = OUT / "tables"
FIG = OUT / "figures"
SPECS = {"Sem clima": "none", "Só temperatura": "temperature", "Três variáveis climáticas": "all"}


def run(n_origins: int) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    panel = mc.load_panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    if len(dates) < 250:
        raise ValueError("Histórico menor que 250 semanas.")
    start = max(104, len(dates) - 145)
    end = len(dates) - 5
    origins = list(dates[np.unique(np.linspace(start, end, n_origins, dtype=int))])
    rows = []
    total_steps = len(origins) * 4
    step = 0
    for oi, origin in enumerate(origins, 1):
        available = panel.loc[panel.date <= origin]
        for h in range(1, 5):
            step += 1
            train = available.loc[
                available.date.le(origin)
                & available[f"casos_lag_{h}"].notna()
                & available.casos_lag_52.notna()
            ].copy()
            target_date = origin + pd.Timedelta(weeks=h)
            test = panel.loc[panel.date.eq(target_date)].copy()
            if len(test) != len(mc.CAPITALS):
                raise ValueError(f"Alvo incompleto na origem={origin}, horizonte={h}.")
            scales = mc.train_mase_scale(train)
            for name, climate_mode in SPECS.items():
                point, lower, upper = mc.fit_nb(train, test, h, climate_mode=climate_mode)
                p = test[["geocode", "municipio", "uf", "date", "casprov"]].copy()
                p = p.rename(columns={"date": "target_date", "casprov": "y_true"})
                p["origin"] = origin
                p["horizon"] = h
                p["specification"] = name
                p["y_pred"] = np.maximum(point, 0)
                p["lower_90"] = np.minimum(np.maximum(lower, 0), p.y_pred.to_numpy())
                p["upper_90"] = np.maximum(upper, p.y_pred.to_numpy())
                p["mase_scale"] = p.geocode.map(scales)
                rows.append(p)
        print(f"Origem {oi}/{len(origins)} concluída: {origin.date()}")
    return pd.concat(rows, ignore_index=True), origins


def metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_h, rows_all, rows_city = [], [], []
    def calculate(part: pd.DataFrame) -> dict:
        y = part.y_true.to_numpy(float); p = part.y_pred.to_numpy(float); e = p - y
        nz = y > 0; den = np.abs(y) + np.abs(p)
        mase = np.abs(e) / part.mase_scale.replace(0, np.nan).to_numpy(float)
        return {
            "MAE_casos_semana": float(np.mean(np.abs(e))),
            "RMSE_casos_semana": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_pct_y_positivo": float(100 * np.mean(np.abs(e[nz] / y[nz]))) if nz.any() else np.nan,
            "cobertura_MAPE_pct": float(100 * nz.mean()),
            "sMAPE_pct": float(100 * np.mean(np.divide(2 * np.abs(e), den, out=np.zeros_like(e), where=den != 0))),
            "MASE": float(np.nanmean(mase)),
            "cobertura_intervalo90_pct": float(100 * np.mean((y >= part.lower_90) & (y <= part.upper_90))),
            "largura_intervalo90_media": float(np.mean(part.upper_90 - part.lower_90)),
            "n_previsoes": len(part),
        }
    for (spec, h), g in pred.groupby(["specification", "horizon"]):
        rows_h.append({"especificacao": spec, "horizonte_semanas": h, **calculate(g)})
    for spec, g in pred.groupby("specification"):
        rows_all.append({"especificacao": spec, **calculate(g)})
    for (spec, city, h), g in pred.groupby(["specification", "municipio", "horizon"]):
        rows_city.append({"especificacao": spec, "capital": city, "horizonte_semanas": h, **calculate(g)})
    return pd.DataFrame(rows_all).sort_values("RMSE_casos_semana"), pd.DataFrame(rows_h), pd.DataFrame(rows_city)


def save(pred: pd.DataFrame, overall: pd.DataFrame, by_h: pd.DataFrame,
         origins: list[pd.Timestamp]) -> None:
    TABLE.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    pred.to_csv(TABLE / "previsoes_ablation_clima.csv.gz", index=False, compression="gzip")
    overall.to_csv(TABLE / "comparacao_especificacoes_clima.csv", index=False, encoding="utf-8-sig")
    by_h.to_csv(TABLE / "comparacao_especificacoes_clima_por_horizonte.csv", index=False, encoding="utf-8-sig")
    city = metrics(pred)[2]
    city.to_csv(TABLE / "comparacao_especificacoes_clima_por_capital.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for spec, g in by_h.groupby("especificacao"):
        ax.plot(g.horizonte_semanas, g.RMSE_casos_semana, marker="o", label=spec)
    ax.set(title="Binomial Negativa: comparação de clima no walk-forward",
           xlabel="Horizonte de previsão (semanas)", ylabel="RMSE (casos por semana)")
    ax.set_xticks([1, 2, 3, 4]); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_ablation_clima_rmse.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    base = overall.set_index("especificacao")
    best = overall.iloc[0]
    gain = 100 * (base.loc["Sem clima", "RMSE_casos_semana"] - best.RMSE_casos_semana) / base.loc["Sem clima", "RMSE_casos_semana"]
    lines = [
        "# Ablation de variáveis climáticas", "",
        f"As três especificações usam a mesma família (GLM Binomial Negativa), as mesmas {len(origins)} origens expansivas, os mesmos alvos e o mesmo horizonte 1–4 semanas. Em cada origem, treino contém apenas linhas até a origem; as features de clima respeitam o horizonte. Não houve embaralhamento.", "",
        "## Especificações", "",
        "- **Sem clima:** casos defasados (horizonte e 52 semanas), calendário e exposição populacional.",
        "- **Só temperatura:** especificação anterior + temperatura, com lag mínimo 2 e lag pelo menos igual ao horizonte.",
        "- **Três variáveis climáticas:** especificação anterior + temperatura, precipitação e umidade, com lags preferidos 2, 3 e 4 semanas (limitados pelo horizonte).",
        "", "## Resultado e decisão", "",
        f"Menor RMSE: **{best.especificacao}**, RMSE={best.RMSE_casos_semana:.2f}, MAE={best.MAE_casos_semana:.2f}, MASE={best.MASE:.3f}. Diferença de RMSE da melhor especificação contra Sem clima: {gain:+.2f}% (positivo significa erro menor). Veja cobertura e largura dos intervalos antes de preferir um modelo.",
        "", "A especificação final do modelo de produção deve equilibrar RMSE, MAE, MASE e cobertura dos intervalos, não apenas correlações. Esta comparação avalia inclusão de clima no modelo NB selecionado; não substitui as comparações entre famílias de modelos. As métricas detalhadas por horizonte e capital estão nas tabelas CSV ao lado.",
        "", f"Origens: {origins[0]:%d/%m/%Y} a {origins[-1]:%d/%m/%Y}; uma origem por ponto espaçado no período recente. O conjunto de previsões foi salvo para auditoria.", "",
    ]
    (OUT / "interpretacao_ablation_clima.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-origins", type=int, default=24)
    args = parser.parse_args()
    pred, origins = run(args.n_origins)
    overall, by_h, _ = metrics(pred)
    save(pred, overall, by_h, origins)
    print("Comparação geral:")
    print(overall.to_string(index=False))
    print(f"Resultados salvos em: {OUT}")


if __name__ == "__main__":
    main()
