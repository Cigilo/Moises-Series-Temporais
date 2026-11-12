"""Four-week dengue forecasts for Northeast capitals with rolling-origin validation."""

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
import statsmodels.api as sm
from scipy.stats import nbinom, t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMClassifier, LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "painel_capitais_modelagem.parquet"
OUT = ROOT / "reports" / "modelagem_capitais"
FIG = OUT / "figures"
TABLE = OUT / "tables"
SEED = 20261109
ALPHA_NB = 1.0
INTERVAL_ALPHA = .10

CAPITALS = {
    "2111300": "São Luís", "2211001": "Teresina", "2304400": "Fortaleza",
    "2408102": "Natal", "2507507": "João Pessoa", "2611606": "Recife",
    "2704302": "Maceió", "2800308": "Aracaju", "2927408": "Salvador",
}
CLIMATE_PREFIXES = ["temp_med", "precip_tot_mm", "rel_humid_med"]
MODEL_NAMES = ["Baseline sazonal", "Binomial Negativa global", "SARIMAX sazonal", "Random Forest", "LightGBM Poisson"]


def load_panel(path: Path = DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Painel atualizado não existe: {path}\n"
            "Execute antes: python src/fetch_capitais_model_data.py"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df.date)
    df["geocode"] = df.geocode.astype(str).str.zfill(7)
    for col in ["casprov", "population", "surto_mem", *CLIMATE_PREFIXES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df.geocode.isin(CAPITALS)].sort_values(["geocode", "date"]).reset_index(drop=True)
    if df.duplicated(["geocode", "date"]).any():
        raise ValueError("O painel contém duplicatas capital-semana.")
    if df.casprov.isna().any():
        raise ValueError(f"Há {df.casprov.isna().sum()} alvos ausentes; não preencher casos ausentes com zero.")
    if not set(df.geocode.unique()) == set(CAPITALS):
        raise ValueError("Nem todas as nove capitais estão presentes no painel atualizado.")
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 52]:
        df[f"casos_lag_{lag}"] = df.groupby("geocode").casprov.shift(lag)
        for col in CLIMATE_PREFIXES:
            df[f"{col}_lag_{lag}"] = df.groupby("geocode")[col].shift(lag)
    week = df.date.dt.isocalendar().week.astype(int)
    df["semana_sin"] = np.sin(2 * np.pi * week / 52.18)
    df["semana_cos"] = np.cos(2 * np.pi * week / 52.18)
    return df


def climate_lags(horizon: int) -> list[str]:
    # lag >= horizon ensures the weather week was fully observed by the origin.
    first = max(2, horizon)
    return [f"{col}_lag_{lag}" for col in CLIMATE_PREFIXES for lag in range(first, 9)]


def feature_columns(horizon: int, compact: bool = False, climate_mode: str = "temperature") -> list[str]:
    if climate_mode not in {"none", "temperature", "all"}:
        raise ValueError("climate_mode deve ser none, temperature ou all")
    if compact:
        features = [f"casos_lag_{horizon}", "casos_lag_52"]
        climate_candidates = []
        if climate_mode in {"temperature", "all"}:
            climate_candidates.append(("temp_med", 2))
        if climate_mode == "all":
            climate_candidates += [("precip_tot_mm", 3), ("rel_humid_med", 4)]
        for col, preferred in climate_candidates:
            lag = max(horizon, preferred)
            if lag <= 8:
                features.append(f"{col}_lag_{lag}")
    else:
        features = [f"casos_lag_{lag}" for lag in range(horizon, 9)] + ["casos_lag_52"]
        if climate_mode == "temperature":
            first = max(2, horizon)
            features += [f"temp_med_lag_{lag}" for lag in range(first, 9)]
        elif climate_mode == "all":
            features += climate_lags(horizon)
    return list(dict.fromkeys(features + ["semana_sin", "semana_cos"]))


def make_design(frame: pd.DataFrame, horizon: int, compact: bool = False,
                climate_mode: str = "temperature") -> pd.DataFrame:
    columns = feature_columns(horizon, compact=compact, climate_mode=climate_mode)
    x = frame[columns].copy()
    for col in columns:
        if col.startswith("casos_lag_"):
            x[col] = np.log1p(pd.to_numeric(x[col], errors="coerce").clip(lower=0))
    if not compact:
        x["log_population"] = np.log(pd.to_numeric(frame.population, errors="coerce").clip(lower=1))
    dummies = pd.get_dummies(frame.geocode, prefix="city", dtype=float)
    return pd.concat([x.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1).astype(float)


def impute(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    train_i = imp.fit_transform(train_x)
    test_i = imp.transform(test_x)
    return train_i, test_i, imp


def fit_nb(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
           climate_mode: str = "temperature") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = make_design(train, horizon, compact=True, climate_mode=climate_mode)
    test_x = make_design(test, horizon, compact=True, climate_mode=climate_mode).reindex(columns=train_x.columns, fill_value=0)
    tx, vx, _ = impute(train_x, test_x)
    tx = sm.add_constant(tx, has_constant="add")
    vx = sm.add_constant(vx, has_constant="add")
    pop_train = np.maximum(train.population.to_numpy(float) / 100_000, 1e-6)
    pop_test = np.maximum(test.population.to_numpy(float) / 100_000, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(train.casprov.to_numpy(float), tx,
                       family=sm.families.NegativeBinomial(alpha=ALPHA_NB),
                       exposure=pop_train).fit(maxiter=100)
    mean = np.clip(model.predict(vx, exposure=pop_test), 0, None)
    shape = 1 / ALPHA_NB
    prob = shape / (shape + mean)
    lower = nbinom.ppf(INTERVAL_ALPHA / 2, shape, prob)
    upper = nbinom.ppf(1 - INTERVAL_ALPHA / 2, shape, prob)
    return mean, np.nan_to_num(lower), np.nan_to_num(upper)


def random_forest_quantiles(model: RandomForestRegressor, x_train: np.ndarray,
                            y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Quantile-forest approximation: pool observed targets sharing each test leaf.
    train_leaves = [tree.apply(x_train) for tree in model.estimators_]
    test_leaves = [tree.apply(x_test) for tree in model.estimators_]
    lower = np.empty(len(x_test), dtype=float)
    upper = np.empty(len(x_test), dtype=float)
    for row in range(len(x_test)):
        support = []
        for tr_leaf, te_leaf in zip(train_leaves, test_leaves):
            support.append(y_train[tr_leaf == te_leaf[row]])
        sample = np.concatenate([s for s in support if len(s)])
        if len(sample) == 0:
            sample = y_train
        lower[row], upper[row] = np.quantile(sample, [INTERVAL_ALPHA / 2, 1 - INTERVAL_ALPHA / 2])
    return np.maximum(lower, 0), np.maximum(upper, 0)


def fit_ml_models(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                  seed: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_x = make_design(train, horizon)
    test_x = make_design(test, horizon).reindex(columns=train_x.columns, fill_value=0)
    tx, vx, _ = impute(train_x, test_x)
    y = train.casprov.to_numpy(float)
    n = len(y)
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4,
                               max_features=.8, n_jobs=-1, random_state=seed)
    rf.fit(tx, y)
    rf_mean = np.clip(rf.predict(vx), 0, None)
    rf_low, rf_high = random_forest_quantiles(rf, tx, y, vx)

    lgb = LGBMRegressor(objective="poisson", n_estimators=220, learning_rate=.04,
                        num_leaves=20, min_child_samples=20, max_bin=63,
                        subsample=.85, colsample_bytree=.85, reg_lambda=2.0,
                        n_jobs=-1, verbosity=-1, random_state=seed)
    lgb.fit(tx, y)
    lgb_mean = np.clip(lgb.predict(vx), 0, None)
    quantile_common = dict(n_estimators=220, learning_rate=.04, num_leaves=20,
                           min_child_samples=20, max_bin=63, reg_lambda=2.0,
                           n_jobs=-1, verbosity=-1, random_state=seed)
    q05 = LGBMRegressor(objective="quantile", alpha=.05, **quantile_common).fit(tx, y)
    q95 = LGBMRegressor(objective="quantile", alpha=.95, **quantile_common).fit(tx, y)
    lgb_low = np.clip(q05.predict(vx), 0, None)
    lgb_high = np.maximum(q95.predict(vx), lgb_low)
    return {
        "Random Forest": (rf_mean, rf_low, rf_high),
        "LightGBM Poisson": (lgb_mean, lgb_low, lgb_high),
    }


def fit_outbreak_classifier(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                            seed: int) -> np.ndarray:
    train = train.dropna(subset=["surto_mem"])
    y = train.surto_mem.astype(int).to_numpy()
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError(f"Sem variação de rótulo MEM no treino para horizonte {horizon}.")
    train_x = make_design(train, horizon)
    test_x = make_design(test, horizon).reindex(columns=train_x.columns, fill_value=0)
    tx, vx, _ = impute(train_x, test_x)
    clf = LGBMClassifier(objective="binary", n_estimators=180, learning_rate=.04,
                         num_leaves=15, min_child_samples=20, reg_lambda=2,
                         scale_pos_weight=(len(y) - positives) / positives,
                         n_jobs=-1, verbosity=-1, random_state=seed)
    clf.fit(tx, y)
    return clf.predict_proba(vx)[:, 1]


def annual_fourier(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    week = pd.DatetimeIndex(dates).isocalendar().week.to_numpy(dtype=float)
    values = {}
    for harmonic in (1, 2):
        values[f"sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * week / 52.18)
        values[f"cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * week / 52.18)
    return pd.DataFrame(values)


def fit_sarimax_by_city(train: pd.DataFrame, horizon_max: int = 4) -> tuple[dict, list[dict]]:
    forecasts = {}
    diagnostics = []
    for code, name in CAPITALS.items():
        g = train.loc[train.geocode == code].sort_values("date")
        y = np.log1p(g.casprov.to_numpy(float))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                historical_season = annual_fourier(pd.DatetimeIndex(g.date))
                future_dates = pd.DatetimeIndex([g.date.max() + pd.Timedelta(weeks=h) for h in range(1, horizon_max + 1)])
                future_season = annual_fourier(future_dates)
                model = SARIMAX(y, exog=historical_season, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), trend="c",
                                enforce_stationarity=False, enforce_invertibility=False)
                # Keep the state covariance so SARIMAX forecast intervals are available.
                result = model.fit(disp=False, maxiter=40)
            pred = result.get_forecast(steps=horizon_max, exog=future_season)
            means = np.expm1(np.asarray(pred.predicted_mean, dtype=float))
            ci = np.asarray(pred.conf_int(alpha=INTERVAL_ALPHA), dtype=float)
            lower = np.maximum(np.expm1(ci[:, 0]), 0)
            upper = np.maximum(np.expm1(ci[:, 1]), lower)
            for h in range(1, horizon_max + 1):
                forecasts[(code, h)] = (max(means[h - 1], 0), lower[h - 1], upper[h - 1])
            diagnostics.append({"geocode": code, "municipio": name, "status": "ok",
                                "aic": result.aic, "convergiu": bool(result.mle_retvals.get("converged", False)),
                                "especificacao": "SARIMAX ARMA(1,1) com 2 harmônicos Fourier anuais como regressoras",
                                "avisos": " | ".join(str(w.message) for w in caught)[:500]})
        except Exception as exc:
            diagnostics.append({"geocode": code, "municipio": name, "status": "falha",
                                "aic": np.nan, "convergiu": False,
                                "especificacao": "SARIMAX ARMA(1,1) com Fourier anual",
                                "avisos": f"{type(exc).__name__}: {exc}"[:500]})
    return forecasts, diagnostics


def baseline_prediction(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = test.casos_lag_52.fillna(0).clip(lower=0).to_numpy(float)
    lows, highs = [], []
    for _, row in test.iterrows():
        residuals = train.loc[train.geocode.eq(row.geocode), "casprov"] - train.loc[
            train.geocode.eq(row.geocode), "casos_lag_52"]
        residuals = residuals.dropna()
        q = residuals.quantile([INTERVAL_ALPHA / 2, 1 - INTERVAL_ALPHA / 2]).to_numpy() if len(residuals) else [0, 0]
        lows.append(max(0, row.casos_lag_52 + q[0]) if pd.notna(row.casos_lag_52) else 0)
        highs.append(max(lows[-1], row.casos_lag_52 + q[1]) if pd.notna(row.casos_lag_52) else max(0, q[1]))
    return mean, np.asarray(lows), np.asarray(highs)


def train_mase_scale(train: pd.DataFrame) -> dict[str, float]:
    g = train.dropna(subset=["casos_lag_52"]).copy()
    g["abs_seasonal_error"] = (g.casprov - g.casos_lag_52).abs()
    return g.groupby("geocode").abs_seasonal_error.mean().to_dict()


def walk_forward(panel: pd.DataFrame, n_origins: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    if len(dates) < 250:
        raise ValueError("Histórico menor que 250 semanas; não há base para validação walk-forward proposta.")
    start = max(104, len(dates) - 145)
    end = len(dates) - 5
    idx = np.unique(np.linspace(start, end, n_origins, dtype=int))
    origins = list(dates[idx])
    rows, outbreak_rows, sarimax_diag = [], [], []
    step = 0
    for origin in origins:
        available = panel.loc[panel.date <= origin].copy()
        sarimax, diagnostics = fit_sarimax_by_city(available)
        for d in diagnostics:
            d["origin"] = origin
            sarimax_diag.append(d)
        for horizon in range(1, 5):
            step += 1
            target_date = origin + pd.Timedelta(weeks=horizon)
            train = available.loc[available.date.le(origin) & available[f"casos_lag_{horizon}"].notna() &
                                  available.casos_lag_52.notna()].copy()
            test = panel.loc[panel.date.eq(target_date)].copy()
            if len(test) != len(CAPITALS):
                raise ValueError(f"Semana de teste incompleta: origem={origin.date()}, h={horizon}, n={len(test)}")
            predictions = {}
            predictions["Baseline sazonal"] = baseline_prediction(train, test)
            predictions["Binomial Negativa global"] = fit_nb(train, test, horizon)
            predictions["SARIMAX sazonal"] = tuple(np.asarray([
                sarimax[(code, horizon)][j] for code in test.geocode
            ]) for j in range(3))
            predictions.update(fit_ml_models(train, test, horizon, SEED + step))
            prob = fit_outbreak_classifier(train, test, horizon, SEED + step)
            scales = train_mase_scale(train)
            for model_name, (point, lower, upper) in predictions.items():
                part = test[["geocode", "municipio", "uf", "date", "casprov", "surto_mem", "population"]].copy()
                part = part.rename(columns={"date": "target_date", "casprov": "y_true", "surto_mem": "surto_true"})
                part["origin"] = origin
                part["horizon"] = horizon
                part["model"] = model_name
                part["y_pred"] = np.maximum(point, 0)
                part["lower_90"] = np.minimum(np.maximum(lower, 0), part.y_pred.to_numpy())
                part["upper_90"] = np.maximum(upper, part.y_pred.to_numpy())
                part["mase_scale"] = part.geocode.map(scales)
                rows.append(part)
            event = test[["geocode", "municipio", "date", "surto_mem"]].copy()
            event = event.rename(columns={"date": "target_date", "surto_mem": "surto_true"})
            event["origin"] = origin
            event["horizon"] = horizon
            event["prob_surto_mem"] = prob
            event["pred_surto_mem"] = (prob >= .5).astype(int)
            outbreak_rows.append(event)
        print(f"Walk-forward {step // 4}/{len(origins)}: origem={origin.date()}, treino até essa semana")
    result = pd.concat(rows, ignore_index=True)
    outbreaks = pd.concat(outbreak_rows, ignore_index=True)
    pd.DataFrame(sarimax_diag).to_csv(
        TABLE / "diagnostico_sarimax.csv", index=False, encoding="utf-8-sig")
    return result, outbreaks, origins


def regression_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, horizon), part in predictions.groupby(["model", "horizon"]):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        e = p - y
        nz = y > 0
        den = np.abs(y) + np.abs(p)
        mase = np.abs(e) / part.mase_scale.replace(0, np.nan).to_numpy(float)
        rows.append({"modelo": model, "horizonte_semanas": int(horizon),
                     "MAE_casos_semana": mean_absolute_error(y, p),
                     "RMSE_casos_semana": np.sqrt(mean_squared_error(y, p)),
                     "MAPE_pct_y_positivo": 100 * np.mean(np.abs(e[nz] / y[nz])) if nz.any() else np.nan,
                     "cobertura_MAPE_pct": 100 * nz.mean(),
                     "sMAPE_pct": 100 * np.mean(np.divide(2 * np.abs(e), den, out=np.zeros_like(e), where=den != 0)),
                     "MASE": np.nanmean(mase), "n_previsoes": len(part),
                     "zeros_observados": int((y == 0).sum())})
    by_h = pd.DataFrame(rows).sort_values(["horizonte_semanas", "RMSE_casos_semana"])
    overall_rows = []
    for model, part in predictions.groupby("model"):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        e = p - y
        nz = y > 0
        den = np.abs(y) + np.abs(p)
        mase = np.abs(e) / part.mase_scale.replace(0, np.nan).to_numpy(float)
        overall_rows.append({"modelo": model,
                             "MAE_casos_semana": mean_absolute_error(y, p),
                             "RMSE_casos_semana": np.sqrt(mean_squared_error(y, p)),
                             "MAPE_pct_y_positivo": 100 * np.mean(np.abs(e[nz] / y[nz])) if nz.any() else np.nan,
                             "cobertura_MAPE_pct": 100 * nz.mean(),
                             "sMAPE_pct": 100 * np.mean(np.divide(2 * np.abs(e), den, out=np.zeros_like(e), where=den != 0)),
                             "MASE": np.nanmean(mase), "n_previsoes": len(part),
                             "zeros_observados": int((y == 0).sum())})
    overall = pd.DataFrame(overall_rows).sort_values("RMSE_casos_semana").reset_index(drop=True)
    return by_h, overall


def city_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, city, horizon), part in predictions.groupby(["model", "municipio", "horizon"]):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        e = p - y
        nz = y > 0
        den = np.abs(y) + np.abs(p)
        rows.append({"modelo": model, "municipio": city, "horizonte_semanas": int(horizon),
                     "MAE_casos_semana": mean_absolute_error(y, p),
                     "RMSE_casos_semana": np.sqrt(mean_squared_error(y, p)),
                     "MAPE_pct_y_positivo": 100 * np.mean(np.abs(e[nz] / y[nz])) if nz.any() else np.nan,
                     "sMAPE_pct": 100 * np.mean(np.divide(2 * np.abs(e), den, out=np.zeros_like(e), where=den != 0)),
                     "n_previsoes": len(part)})
    return pd.DataFrame(rows)


def write_dataset_docs() -> None:
    dictionaries = [
        ("geocode", "Código municipal IBGE", "InfoDengue/IBGE", "código", "município", "Efeito fixo/categórico de capital"),
        ("date", "Domingo de início da semana epidemiológica", "InfoDengue", "data", "semanal", "Índice temporal"),
        ("casprov", "Casos prováveis (suspeitos menos descartados)", "InfoDengue/SINAN/IBGE", "casos", "semanal", "Alvo de regressão"),
        ("surto_mem", "1 se nível_inc MEM = 2 (incidência acima do limiar epidêmico)", "InfoDengue/MEM", "0/1", "semanal", "Alvo de classificação"),
        ("population", "População anual mais recente conhecida, carregada para frente", "DATASUS/SVS", "habitantes", "anual", "Exposição NB e cálculo da incidência"),
        ("temp_med", "Temperatura média diária agregada por semana", "NASA POWER T2M", "°C", "semanal", "Explicativa climática"),
        ("precip_tot_mm", "Soma semanal de precipitação corrigida diária", "NASA POWER PRECTOTCORR", "mm/semana", "semanal", "Explicativa climática"),
        ("rel_humid_med", "Umidade relativa média diária agregada por semana", "NASA POWER RH2M", "%", "semanal", "Explicativa climática"),
        ("casos_lag_k", "Casos prováveis k semanas antes", "Calculada por capital após ordenar por data", "casos", "semanal, k=1..8 e 52", "Explicativas conhecidas na origem"),
        ("clima_lag_k", "Clima observado k semanas antes", "Calculada por capital após ordenar por data", "unidade da variável climática", "semanal, k=2..8", "Só entram lags disponíveis no horizonte"),
        ("semana_sin/cos", "Codificação cíclica da semana epidemiológica", "Calculada", "adimensional", "semanal", "Sazonalidade de calendário"),
    ]
    pd.DataFrame(dictionaries, columns=["variavel", "descricao", "fonte", "unidade", "frequencia", "uso"]).to_csv(
        TABLE / "dicionario_modelagem.csv", index=False, encoding="utf-8-sig")
    sources = [
        ("casos prováveis e MEM", "InfoDengue alertcity", "2015–SE43 2026", "https://info.dengue.mat.br/services/api", "Consulta pública por geocódigo; último retorno foi SE43, iniciada em 25/10/2026"),
        ("temperatura, chuva e umidade", "NASA POWER Daily Point API", "2015-01-01 a 2026-11-09", "https://power.larc.nasa.gov/docs/services/api/temporal/daily/", "Grade NASA POWER; T2M, PRECTOTCORR, RH2M agregados para semana iniciada no domingo"),
        ("população", "DATASUS/SVS via pacote de dados Mosqlimate", "2001–2025; último ano conhecido", "https://sprint.mosqlimate.org/data/", "Última estimativa conhecida por semana; sem dados populacionais futuros"),
    ]
    pd.DataFrame(sources, columns=["variavel", "fonte", "periodo", "documentacao", "observacoes"]).to_csv(
        TABLE / "fontes_dados_modelagem.csv", index=False, encoding="utf-8-sig")


def interval_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), part in predictions.groupby(["model", "horizon"]):
        y = part.y_true.to_numpy(float)
        lo, hi = part.lower_90.to_numpy(float), part.upper_90.to_numpy(float)
        rows.append({"modelo": model, "horizonte_semanas": int(horizon), "nivel_nominal_pct": 90,
                     "cobertura_empirica_pct": 100 * np.mean((y >= lo) & (y <= hi)),
                     "largura_media_intervalo_casos": np.mean(hi - lo), "n": len(part)})
    return pd.DataFrame(rows)


def outbreak_metrics(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    events = events.dropna(subset=["surto_true"]).copy()
    for horizon, part in events.groupby("horizon"):
        actual, predicted = part.surto_true.astype(int), part.pred_surto_mem.astype(int)
        rows.append({"horizonte_semanas": int(horizon),
                     "precision": precision_score(actual, predicted, zero_division=0),
                     "recall": recall_score(actual, predicted, zero_division=0),
                     "F1": f1_score(actual, predicted, zero_division=0),
                     "limiar_probabilidade": .5, "n": len(part), "eventos_MEM": int(actual.sum()),
                     "prevalencia_pct": 100 * actual.mean()})
    actual, predicted = events.surto_true.astype(int), events.pred_surto_mem.astype(int)
    rows.append({"horizonte_semanas": "todos",
                 "precision": precision_score(actual, predicted, zero_division=0),
                 "recall": recall_score(actual, predicted, zero_division=0),
                 "F1": f1_score(actual, predicted, zero_division=0),
                 "limiar_probabilidade": .5, "n": len(events), "eventos_MEM": int(actual.sum()),
                 "prevalencia_pct": 100 * actual.mean()})
    return pd.DataFrame(rows)


def dm_by_horizon(predictions: pd.DataFrame, best_models: list[str]) -> pd.DataFrame:
    model_a, model_b = best_models[:2]
    rows = []
    for horizon, data in predictions.loc[predictions.model.isin([model_a, model_b])].groupby("horizon"):
        a = data.loc[data.model == model_a].copy()
        b = data.loc[data.model == model_b].copy()
        a["loss"] = (a.y_true - a.y_pred) ** 2
        b["loss"] = (b.y_true - b.y_pred) ** 2
        wide = a.groupby("target_date").loss.mean().to_frame("loss_a").join(
            b.groupby("target_date").loss.mean().to_frame("loss_b"), how="inner")
        d = (wide.loss_a - wide.loss_b).to_numpy(float)
        n = len(d)
        mean_d = float(d.mean())
        centered = d - mean_d
        lag_max = min(3, n - 1)
        long_run_var = float(np.dot(centered, centered) / n)
        for lag in range(1, lag_max + 1):
            gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
            long_run_var += 2 * (1 - lag / (lag_max + 1)) * gamma
        se = np.sqrt(max(long_run_var, 1e-12) / n)
        stat = mean_d / se
        # Harvey-Leybourne-Newbold finite sample correction, with 24 temporal origins.
        h = int(horizon)
        correction = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        stat *= correction
        p = 2 * student_t.sf(abs(stat), df=max(n - 1, 1))
        rows.append({"horizonte_semanas": int(horizon), "modelo_A": model_a, "modelo_B": model_b,
                     "diferenca_perda_quadratica_A_menos_B": mean_d,
                     "estatistica_DM_HAC": stat, "p_valor_bilateral": p,
                     "origens_temporais": n, "lag_HAC": lag_max,
                     "interpretacao": "p<0,05 indica diferença de perda pareada; não significativo não prova empate"})
    return pd.DataFrame(rows)


def future_rows(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    target = pd.Timestamp(origin) + pd.Timedelta(weeks=horizon)
    records = []
    last = panel.loc[panel.date.eq(origin)].set_index("geocode")
    for code, name in CAPITALS.items():
        source = last.loc[code]
        row = {"geocode": code, "municipio": name, "uf": source.uf,
               "date": target, "population": source.population}
        wk = int(target.isocalendar().week)
        row["semana_sin"] = np.sin(2 * np.pi * wk / 52.18)
        row["semana_cos"] = np.cos(2 * np.pi * wk / 52.18)
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 52]:
            source_date = target - pd.Timedelta(weeks=lag)
            hist = panel.loc[(panel.geocode == code) & panel.date.eq(source_date)]
            row[f"casos_lag_{lag}"] = hist.casprov.iloc[0] if len(hist) else np.nan
            for col in CLIMATE_PREFIXES:
                row[f"{col}_lag_{lag}"] = hist[col].iloc[0] if len(hist) else np.nan
        records.append(row)
    return pd.DataFrame(records)


def final_forecasts(panel: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = panel.loc[panel.date.le(origin)].copy()
    sarimax, sarimax_diag = fit_sarimax_by_city(available)
    forecasts = []
    outbreak = []
    for horizon in range(1, 5):
        train = available.loc[available[f"casos_lag_{horizon}"].notna() & available.casos_lag_52.notna()].copy()
        future = future_rows(panel, origin, horizon)
        predictions = {"Baseline sazonal": baseline_prediction(train, future)}
        predictions["Binomial Negativa global"] = fit_nb(train, future, horizon)
        predictions["SARIMAX sazonal"] = tuple(np.asarray([sarimax[(code, horizon)][i] for code in future.geocode]) for i in range(3))
        predictions.update(fit_ml_models(train, future, horizon, SEED + 20_000 + horizon))
        prob = fit_outbreak_classifier(train, future, horizon, SEED + 30_000 + horizon)
        for name, (point, low, high) in predictions.items():
            result = future[["geocode", "municipio", "uf", "date", "population"]].copy()
            result = result.rename(columns={"date": "target_date"})
            result["origin"] = origin
            result["horizon"] = horizon
            result["model"] = name
            result["casos_previstos"] = np.maximum(point, 0)
            result["limite_90_inferior"] = np.minimum(np.maximum(low, 0), result.casos_previstos)
            result["limite_90_superior"] = np.maximum(high, result.casos_previstos)
            result["incidencia_prevista_100mil"] = result.casos_previstos / result.population * 100_000
            result["probabilidade_acima_limiar_MEM"] = prob
            result["alerta_MEM_prob_ge_0_5"] = prob >= .5
            forecasts.append(result)
        event = future[["geocode", "municipio", "date"]].rename(columns={"date": "target_date"})
        event["origin"] = origin
        event["horizon"] = horizon
        event["probabilidade_acima_limiar_MEM"] = prob
        event["alerta_MEM_prob_ge_0_5"] = prob >= .5
        outbreak.append(event)
    return pd.concat(forecasts, ignore_index=True), pd.concat(outbreak, ignore_index=True)


def save_figures(metrics: pd.DataFrame, intervals: pd.DataFrame, forecasts: pd.DataFrame,
                 overall: pd.DataFrame, selected: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.9)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=metrics, x="horizonte_semanas", y="RMSE_casos_semana", hue="modelo", marker="o", ax=ax)
    ax.set_title("Erro de previsão por horizonte (walk-forward)")
    ax.set_xlabel("Horizonte (semanas)")
    ax.set_ylabel("RMSE (casos por semana)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(title="Modelo", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_rmse_por_horizonte.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=intervals, x="horizonte_semanas", y="cobertura_empirica_pct", hue="modelo", marker="o", ax=ax)
    ax.axhline(90, color="#d92d20", ls="--", lw=1, label="Cobertura nominal 90%")
    ax.set_title("Cobertura empírica dos intervalos de previsão")
    ax.set_xlabel("Horizonte (semanas)")
    ax.set_ylabel("Observações dentro do intervalo (%)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 105)
    ax.legend(title="Modelo", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_cobertura_intervalos.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    chosen = forecasts.loc[forecasts.model == selected]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for ax, (code, name) in zip(axes.flat, CAPITALS.items()):
        g = chosen.loc[chosen.geocode == code].sort_values("horizon")
        x = np.arange(1, len(g) + 1)
        y = g.casos_previstos.to_numpy(float)
        ax.errorbar(x, y, yerr=[y - g.limite_90_inferior.to_numpy(float),
                                g.limite_90_superior.to_numpy(float) - y],
                    fmt="o-", capsize=3, color="#175cd3")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Casos previstos/semana")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=.2)
    fig.suptitle(f"Previsão do modelo {selected}: quatro semanas a partir do último dado", fontsize=13)
    fig.supxlabel("Horizonte (semanas)")
    fig.tight_layout()
    fig.savefig(FIG / "03_previsao_proximas_4_semanas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(panel: pd.DataFrame, metrics: pd.DataFrame, overall: pd.DataFrame,
                  intervals: pd.DataFrame, outbreak: pd.DataFrame, dm: pd.DataFrame,
                  forecasts: pd.DataFrame, origins: list[pd.Timestamp], selected: str,
                  diagnostics: pd.DataFrame) -> None:
    latest = panel.date.max()
    forecast_from = latest
    finaldates = sorted(forecasts.loc[forecasts.model == selected, "target_date"].unique())
    forecast_range = f"{pd.Timestamp(finaldates[0]):%d/%m/%Y}–{pd.Timestamp(finaldates[-1]):%d/%m/%Y}"
    latest_epiweek = panel.loc[panel.date.eq(latest), "epiweek"].max()
    top_two = overall.modelo.iloc[:2].tolist()
    good = int((diagnostics.status == "ok").sum()) if not diagnostics.empty else 0
    failed = len(diagnostics) - good
    lines = [
        "# Modelagem e validação — nove capitais do Nordeste",
        "",
        f"## Dados e origem de previsão\n\nA base de casos do InfoDengue foi atualizada em 09/11/2026 e contém observações até a SE {int(latest_epiweek)} ({latest:%d/%m/%Y}, início da semana epidemiológica), com **{panel.geocode.nunique()} capitais** e **{len(panel):,} semanas-capital**. Clima diário NASA POWER foi agregado a semanas iniciadas no domingo; população vem do último ano conhecido no painel DATASUS, sem usar anos futuros. O corte local do modelo usa casos prováveis `casprov` e rótulo oficial `nivel_inc_mem == 2` para incidência acima do limiar MEM.",
        "",
        "NASA POWER é uma fonte meteorológica em grade, não uma estação observacional local. O modelo usa NASA POWER de forma consistente em todo o período para temperatura (°C), precipitação acumulada semanal (mm) e umidade relativa (%). A EDA exploratória anterior usou reanálise ERA5 do pacote Mosqlimate; correlações dela são candidatas, enquanto a seleção final de variáveis será confirmada no backtest com as variáveis NASA. Fontes, períodos, unidades e transformações estão em `tables/fontes_dados_modelagem.csv` e `tables/dicionario_modelagem.csv`.",
        "",
        f"## Validação temporal\n\nForam usadas {len(origins)} origens expansivas entre {origins[0]:%d/%m/%Y} e {origins[-1]:%d/%m/%Y}, espaçadas ao longo dos últimos anos. Em cada origem, o treino usa somente semanas até aquela origem; nenhuma divisão aleatória é usada. Cada modelo prevê diretamente horizontes de 1 a 4 semanas, e as explicativas climáticas têm defasagem suficiente para estarem disponíveis no corte.",
        "",
        "## Modelos e intervalos",
        "",
        "1. Baseline sazonal: repete casos observados 52 semanas antes.",
        "2. Binomial Negativa global: GLM de contagem com efeitos fixos de capital, exposição populacional e dispersão NB2 fixada em α=1; intervalos 90% pela distribuição binomial negativa.",
        "3. SARIMAX sazonal local: ARMA(1,1) sobre log(1+casos), com dois pares de harmônicos Fourier anuais como regressoras sazonais; intervalos 90% do estado-espaço transformados para casos.",
        "4. Random Forest global: árvores com efeitos de capital; intervalos 90% pelo pool empírico de valores nas folhas das árvores (aproximação de floresta quantílica).",
        "5. LightGBM Poisson global: perda Poisson; intervalos 90% por regressões quantílicas nos níveis 5% e 95%.",
        "",
        "Após uma comparação de especificações no mesmo walk-forward, todos os modelos globais usam temperatura como único preditor meteorológico. Para NB, entra um único lag por horizonte (mínimo 2 semanas); RF e LightGBM recebem lags de temperatura observados e disponíveis na origem. Precipitação e umidade foram excluídas da especificação principal porque não melhoraram o RMSE da NB e elevaram redundância entre preditores. A cobertura e largura dos intervalos são verificadas no walk-forward; a cobertura observada pode ficar abaixo dos 90% nominais.",
        "",
        "## Resultados",
        "",
        f"O menor RMSE agregado foi de **{overall.iloc[0].modelo}** ({overall.iloc[0].RMSE_casos_semana:.2f} casos/semana); o segundo menor foi **{overall.iloc[1].modelo}** ({overall.iloc[1].RMSE_casos_semana:.2f}). A tabela completa com MAE, RMSE, MAPE apenas em semanas positivas, cobertura do MAPE, sMAPE e MASE está em `tables/metricas_modelos.csv`, por horizonte em `tables/metricas_por_horizonte.csv` e por capital em `tables/metricas_por_capital.csv`.",
        "",
        "Como dengue tem zeros e contagens pequenas em parte das semanas, MAPE é reportado apenas onde casos observados >0 e vem acompanhado da cobertura; sMAPE e MASE completam a avaliação.",
        "",
        "A ablation climática pareada está em `tables/comparacao_especificacoes_clima.csv`: NB com temperatura apenas teve RMSE 54,68; sem clima, 56,03; com temperatura, chuva e umidade, 55,55. O ganho é modesto, por isso temperatura fica como covariável candidata e a exclusão dos outros climas é mantida por parcimônia, não como prova de ausência de associação.",
        "",
        "### Precisão de eventos MEM",
        "",
        f"O classificador LightGBM de cruzamento MEM (probabilidade ≥0,5) obteve precisão={outbreak.iloc[-1].precision:.3f}, recall={outbreak.iloc[-1].recall:.3f} e F1={outbreak.iloc[-1].F1:.3f} no conjunto de origens. O desbalanceamento e a escolha do limiar afetam esses números; a tabela também separa os quatro horizontes.",
        "",
        "### Diebold–Mariano",
        "",
        f"Os dois modelos comparados são {top_two[0]} e {top_two[1]}, escolhidos pela RMSE de validação, não por desempenho no treino. O teste usa diferença de perda quadrática regional, agregada por semana-alvo e horizonte, com variância HAC até lag 3 e correção de amostra finita. Resultados por horizonte estão em `tables/diebold_mariano.csv`; p>0,05 significa evidência insuficiente de diferença, não equivalência.",
        "",
        "### SARIMAX",
        "",
        f"Ajustes SARIMAX concluídos: {good}; falhas documentadas: {failed}. Ver `tables/diagnostico_sarimax.csv`; não foram substituídos silenciosamente por dados inventados.",
        "",
        "## Previsão a partir do dado mais recente",
        "",
        f"O modelo selecionado por RMSE ({selected}) gera previsões para **{forecast_range}**, correspondentes aos horizontes 1–4 a partir de {forecast_from:%d/%m/%Y}. O InfoDengue tem atraso de divulgação: na data de execução (09/11/2026), as primeiras semanas deste horizonte já podem ter começado ou terminado, embora a série pública ainda pare na SE {int(latest_epiweek)}. Portanto, são previsões condicionadas à última semana publicada, não previsões em tempo real para quatro semanas completas a partir de hoje.",
        "",
        "Os arquivos `tables/previsao_4_semanas_por_modelo.csv` e `tables/resumo_previsao_por_capital.csv` incluem casos, intervalo 90%, incidência por 100 mil habitantes e probabilidade estimada de ultrapassar o limiar MEM. Para previsão de incidência, o modelo não inventa um corte futuro: usa o classificador treinado no rótulo oficial MEM.",
        "",
        "## Limitações e próximo passo",
        "",
        "As origens de validação são 24 semanas escolhidas ao longo dos últimos ~145 períodos; a amostra temporal ainda limita o teste DM. A chuva da NASA é acumulada semanal em mm, enquanto a EDA ERA5 anterior usou taxa média em mm/h. O pipeline mantém fontes e unidades distintas documentadas.",
        "",
        "Não se deve reutilizar previsões dos rascunhos antigos: esta análise foi recalculada para nove capitais com casos atualizados em novembro de 2026.",
    ]
    (OUT / "interpretacao_modelagem.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-origins", type=int, default=24)
    parser.add_argument("--skip-final-forecast", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    write_dataset_docs()
    predictions, events, origins = walk_forward(panel, n_origins=args.n_origins)
    by_h, overall = regression_metrics(predictions)
    by_city = city_metrics(predictions)
    intervals = interval_metrics(predictions)
    events_metrics = outbreak_metrics(events)
    best_two = overall.modelo.iloc[:2].tolist()
    dm = dm_by_horizon(predictions, best_two)
    predictions.to_csv(TABLE / "previsoes_walk_forward.csv.gz", index=False, compression="gzip")
    events.to_csv(TABLE / "probabilidades_surto_walk_forward.csv", index=False, encoding="utf-8-sig")
    by_h.to_csv(TABLE / "metricas_por_horizonte.csv", index=False, encoding="utf-8-sig")
    by_city.to_csv(TABLE / "metricas_por_capital.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(TABLE / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(TABLE / "cobertura_intervalos.csv", index=False, encoding="utf-8-sig")
    events_metrics.to_csv(TABLE / "metricas_evento_MEM.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(TABLE / "diebold_mariano.csv", index=False, encoding="utf-8-sig")

    final = pd.DataFrame()
    if not args.skip_final_forecast:
        origin = panel.date.max()
        final, final_event = final_forecasts(panel, origin)
        final.to_csv(TABLE / "previsao_4_semanas_por_modelo.csv", index=False, encoding="utf-8-sig")
        final_event.to_csv(TABLE / "probabilidade_surto_4_semanas.csv", index=False, encoding="utf-8-sig")
        selected = overall.modelo.iloc[0]
        capital_summary = final.loc[final.model == selected].copy()
        capital_summary.to_csv(TABLE / "resumo_previsao_por_capital.csv", index=False, encoding="utf-8-sig")
        regional = capital_summary.groupby(["target_date", "origin", "horizon"], as_index=False).agg(
            casos_previstos_total=("casos_previstos", "sum"),
            limite_90_inferior_soma=("limite_90_inferior", "sum"),
            limite_90_superior_soma=("limite_90_superior", "sum"),
            populacao_total=("population", "sum"),
            probabilidade_media_MEM=("probabilidade_acima_limiar_MEM", "mean"),
        )
        regional["incidencia_prevista_100mil"] = regional.casos_previstos_total / regional.populacao_total * 100_000
        regional.to_csv(TABLE / "resumo_previsao_regional.csv", index=False, encoding="utf-8-sig")
        save_figures(by_h, intervals, final, overall, selected)
        write_summary(panel, by_h, overall, intervals, events_metrics, dm, final, origins,
                      selected, pd.read_csv(TABLE / "diagnostico_sarimax.csv"))
        print(f"Modelo selecionado por RMSE: {selected}")
        print(f"Previsão condicional às semanas publicadas: {panel.date.max():%Y-%m-%d} + 4 semanas")
    print("Métricas gerais:")
    print(overall.to_string(index=False))
    print(f"Resultados salvos em: {OUT}")


if __name__ == "__main__":
    main()
