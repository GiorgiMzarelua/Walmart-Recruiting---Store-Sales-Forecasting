"""Feature engineering for the Walmart store-sales forecasting task.

Design decisions are grounded in EDA (notebooks/00_eda.ipynb):
  - Markdowns dropped: 64-74% missing, |corr with sales| <= 0.09.
  - Store Size kept: corr 0.807 with store-average sales; Type separates A/B/C.
  - 340 series have < 52 weeks (37 have a single row), so the same-week-last-year
    signal is unavailable for them -> hierarchical fallback (series -> dept ->
    store -> global) guarantees a defined value for every row.
  - Target is right-skewed (skew 3.26) with 1,285 negative rows (returns), so a
    plain log1p is unsafe. Use a signed log transform that is invertible on
    negatives, and clip predictions at 0 only at the very end.

LEAKAGE RULE: any feature that depends on the target is FIT ON TRAIN ONLY
(fit_seasonal_profiles) and then merely *applied* to valid/test. This covers
both the seasonal profiles and lag_52.

HORIZON RULE: the test set is 39 weeks ahead of the end of training, so any
feature that needs recent target values (lag_1, lag_2, lag_4, rolling stats)
is IMPOSSIBLE to compute at inference time. Those lags were removed. The only
surviving lag is lag_52 (same week last year), because for every valid/test
date the value 52 weeks earlier falls *inside* the training window. It is built
as a lookup from training history, never as a shift() on the frame being
predicted -- which is what makes it identical on train, valid and test (no
train/serve skew) and free of leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Holiday-flagged weeks: Super Bowl(6), Labor Day(36), Thanksgiving(47), Christmas(52)
HOLIDAY_WEEKS = (6, 36, 47, 52)
CATEGORICAL = ["Store", "Dept", "Type"]
MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]

# Only lag_52 survives the 39-week horizon. Short lags were removed on purpose.
LAG_COLUMNS = ["lag_52"]

FEATURE_COLUMNS = [
    "Store", "Dept", "Type", "Size", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "year", "month", "week", "dayofyear",
    "weeks_to_holiday", "weeks_since_holiday",
    "sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3", "sin_4", "cos_4",
    "seasonal_week_avg",
]


# --------------------------------------------------------------------------- #
# Target transform (safe on negative sales)
# --------------------------------------------------------------------------- #
def signed_log1p(y):
    """log1p that works for negatives: sign(y) * log(1 + |y|). Invertible."""
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_signed_log1p(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


# --------------------------------------------------------------------------- #
# Stateless calendar / seasonal features (safe to apply to any split)
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["Date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["week"] = d.isocalendar().week.astype(int)
    df["dayofyear"] = d.dayofyear
    return df


def add_holiday_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Weeks to nearest upcoming / most-recent holiday week (circular, 52-week).
    Captures the pre-holiday build-up the raw IsHoliday flag misses (e.g. wk 51)."""
    df = df.copy()
    wk = df["Date"].dt.isocalendar().week.astype(int).to_numpy()
    hol = np.asarray(HOLIDAY_WEEKS)
    diff = wk[:, None] - hol[None, :]
    df["weeks_to_holiday"] = np.mod(-diff, 52).min(axis=1)
    df["weeks_since_holiday"] = np.mod(diff, 52).min(axis=1)
    return df


def add_fourier(df: pd.DataFrame, n_terms: int = 4) -> pd.DataFrame:
    """Fourier terms for the 52-week annual cycle."""
    df = df.copy()
    t = df["Date"].dt.dayofyear.to_numpy() / 365.25
    for k in range(1, n_terms + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t)
    return df


# --------------------------------------------------------------------------- #
# Target-dependent lookups -> FIT ON TRAIN ONLY
# --------------------------------------------------------------------------- #
def fit_seasonal_profiles(train: pd.DataFrame) -> dict:
    """Learn every target-derived lookup from TRAIN only.

    Seasonal profiles: average sales per (unique_id, week) / (Dept, week) /
    (Store, week) plus a global mean -> hierarchical fallback.

    lag_52 tables: each training row's sales, keyed by the date it PREDICTS
    (i.e. its own date + 52 weeks). At predict time we look a row's own Date up
    in this table, which returns the value from exactly one year earlier.
    """
    tr = add_calendar_features(train)

    t = train[["unique_id", "Dept", "Date", "Weekly_Sales"]].copy()
    t["target_date"] = t["Date"] + pd.Timedelta(weeks=52)

    return {
        "series_week": tr.groupby(["unique_id", "week"])["Weekly_Sales"].mean(),
        "dept_week":   tr.groupby(["Dept", "week"])["Weekly_Sales"].mean(),
        "store_week":  tr.groupby(["Store", "week"])["Weekly_Sales"].mean(),
        "global_mean": float(tr["Weekly_Sales"].mean()),
        "lag52_series": t.groupby(["unique_id", "target_date"])["Weekly_Sales"].mean(),
        "lag52_dept":   t.groupby(["Dept", "target_date"])["Weekly_Sales"].mean(),
    }


def _map_index(df: pd.DataFrame, keys: list[str], lookup: pd.Series) -> np.ndarray:
    return df.set_index(keys).index.map(lookup).to_numpy()


def apply_seasonal_profiles(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Add `seasonal_week_avg` with the series->dept->store->global cascade so
    every row (including the 340 short series) gets a defined value."""
    df = add_calendar_features(df)
    series = _map_index(df, ["unique_id", "week"], profiles["series_week"])
    dept = _map_index(df, ["Dept", "week"], profiles["dept_week"])
    store = _map_index(df, ["Store", "week"], profiles["store_week"])
    out = np.where(pd.isna(series), np.where(pd.isna(dept), store, dept), series)
    out = np.where(pd.isna(out), profiles["global_mean"], out)
    df["seasonal_week_avg"] = out.astype(float)
    return df


def apply_lag52(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """Same-week-last-year, looked up from TRAIN history.

    Never reads df's own Weekly_Sales, so it is identical on train / valid /
    test: no leakage and no train-serve skew. Falls back series -> dept ->
    seasonal_week_avg, so it is never NaN.

    MUST be called after apply_seasonal_profiles (it uses it as the fallback).
    """
    df = df.copy()
    s = _map_index(df, ["unique_id", "Date"], profiles["lag52_series"])
    d = _map_index(df, ["Dept", "Date"], profiles["lag52_dept"])
    out = np.where(pd.isna(s), d, s)
    df["lag_52"] = (
        pd.Series(out, index=df.index).astype(float).fillna(df["seasonal_week_avg"])
    )
    return df


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
def apply_features_with_profiles(
    df: pd.DataFrame,
    profiles: dict,
    *,
    n_fourier: int = 4,
    drop_markdowns: bool = True,
    use_lags: bool = False,
) -> pd.DataFrame:
    """Raw merged frame -> model-ready features. Row order is preserved."""
    df = add_calendar_features(df)
    df = add_holiday_distance(df)
    df = add_fourier(df, n_terms=n_fourier)
    df = apply_seasonal_profiles(df, profiles)

    if use_lags:
        df = apply_lag52(df, profiles)

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    if drop_markdowns:
        df = df.drop(columns=[c for c in MARKDOWNS if c in df.columns])
    return df


def build_features(train: pd.DataFrame, other: pd.DataFrame, *,
                   n_fourier: int = 4, drop_markdowns: bool = True,
                   use_lags: bool = False):
    """Fit profiles on `train`, apply the feature set to both. `other` = valid/test."""
    profiles = fit_seasonal_profiles(train)
    kw = dict(n_fourier=n_fourier, drop_markdowns=drop_markdowns, use_lags=use_lags)
    return (apply_features_with_profiles(train, profiles, **kw),
            apply_features_with_profiles(other, profiles, **kw))


def make_sample_weight(df: pd.DataFrame, holiday_weight: float = 5.0) -> np.ndarray:
    """Holiday rows weigh 5x -- matches the official WMAE."""
    return np.where(df["IsHoliday"].astype(bool), holiday_weight, 1.0)


def _feature_cols(use_lags: bool) -> list[str]:
    return list(FEATURE_COLUMNS) + (LAG_COLUMNS if use_lags else [])


def _finish_predict(preds_log: np.ndarray, df_fe: pd.DataFrame,
                    original_index) -> np.ndarray:
    """Invert the target transform, clip at 0, realign to the caller's row order."""
    preds = np.clip(inverse_signed_log1p(preds_log), 0, None)
    return pd.Series(preds, index=df_fe.index).reindex(original_index).to_numpy()


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
LGBM_DEFAULT_PARAMS = dict(
    objective="regression_l1",
    metric="l1",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def fit_lightgbm(train: pd.DataFrame, *, lgbm_params: dict | None = None,
                 n_fourier: int = 4, drop_markdowns: bool = True,
                 holiday_weight: float = 5.0, use_lags: bool = False) -> dict:
    from lightgbm import LGBMRegressor

    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier,
        drop_markdowns=drop_markdowns, use_lags=use_lags,
    )
    feature_cols = _feature_cols(use_lags)

    params = dict(LGBM_DEFAULT_PARAMS)
    if lgbm_params:
        params.update(lgbm_params)

    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)

    booster = LGBMRegressor(**params)
    booster.fit(train_fe[feature_cols], y, sample_weight=w)

    return {"profiles": profiles, "booster": booster,
            "feature_columns": feature_cols, "n_fourier": n_fourier,
            "drop_markdowns": drop_markdowns, "use_lags": use_lags}


def predict_lightgbm(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    original_index = raw_df.index
    df_fe = apply_features_with_profiles(
        raw_df, bundle["profiles"], n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"], use_lags=bundle.get("use_lags", False),
    )
    preds_log = bundle["booster"].predict(df_fe[bundle["feature_columns"]])
    return _finish_predict(preds_log, df_fe, original_index)


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
XGB_DEFAULT_PARAMS = dict(
    objective="reg:absoluteerror",
    eval_metric="mae",
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def fit_xgboost(train: pd.DataFrame, *, xgb_params: dict | None = None,
                n_fourier: int = 4, drop_markdowns: bool = True,
                holiday_weight: float = 5.0, use_lags: bool = False) -> dict:
    from xgboost import XGBRegressor

    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier,
        drop_markdowns=drop_markdowns, use_lags=use_lags,
    )
    feature_cols = _feature_cols(use_lags)

    params = dict(XGB_DEFAULT_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)

    booster = XGBRegressor(**params)
    booster.fit(train_fe[feature_cols], y, sample_weight=w)

    return {"profiles": profiles, "booster": booster,
            "feature_columns": feature_cols, "n_fourier": n_fourier,
            "drop_markdowns": drop_markdowns, "use_lags": use_lags}


def predict_xgboost(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    original_index = raw_df.index
    df_fe = apply_features_with_profiles(
        raw_df, bundle["profiles"], n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"], use_lags=bundle.get("use_lags", False),
    )
    preds_log = bundle["booster"].predict(df_fe[bundle["feature_columns"]])
    return _finish_predict(preds_log, df_fe, original_index)


# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #
CATBOOST_DEFAULT_PARAMS = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    one_hot_max_size=255,   # avoid target-statistic (CTR) encoding on high-card IDs
    random_seed=42,
    verbose=0,
    thread_count=-1,
    task_type="GPU",
)


def fit_catboost(train: pd.DataFrame, *, cat_params: dict | None = None,
                 n_fourier: int = 4, drop_markdowns: bool = True,
                 holiday_weight: float = 5.0, use_lags: bool = False) -> dict:
    from catboost import CatBoostRegressor, Pool

    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier,
        drop_markdowns=drop_markdowns, use_lags=use_lags,
    )
    feature_cols = _feature_cols(use_lags)

    params = dict(CATBOOST_DEFAULT_PARAMS)
    if cat_params:
        params.update(cat_params)

    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)

    cat_features = [c for c in CATEGORICAL if c in feature_cols]
    X = train_fe[feature_cols].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)

    booster = CatBoostRegressor(**params)
    booster.fit(Pool(data=X, label=y, weight=w, cat_features=cat_features))

    return {"profiles": profiles, "booster": booster,
            "feature_columns": feature_cols, "cat_features": cat_features,
            "n_fourier": n_fourier, "drop_markdowns": drop_markdowns,
            "use_lags": use_lags}


def predict_catboost(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    from catboost import Pool

    original_index = raw_df.index
    df_fe = apply_features_with_profiles(
        raw_df, bundle["profiles"], n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"], use_lags=bundle.get("use_lags", False),
    )
    X = df_fe[bundle["feature_columns"]].copy()
    for c in bundle.get("cat_features", []):
        X[c] = X[c].astype(str)

    preds_log = bundle["booster"].predict(Pool(data=X, cat_features=bundle["cat_features"]))
    return _finish_predict(preds_log, df_fe, original_index)