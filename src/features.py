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

Leakage rule: seasonal-profile features depend on the target, so they are FIT ON
TRAIN ONLY (fit_seasonal_profiles) and then applied to valid/test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Holiday-flagged weeks: Super Bowl(6), Labor Day(36), Thanksgiving(47), Christmas(52)
HOLIDAY_WEEKS = (6, 36, 47, 52)
CATEGORICAL = ["Store", "Dept", "Type"]
MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]


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
# Lag & Rolling features
# --------------------------------------------------------------------------- #
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical lags + rolling stats per series. Restores the original row
    order before returning, so predictions stay aligned with the caller's frame."""
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))          # positional marker
    df = df.sort_values(["unique_id", "Date"])      # lags need chronological order per series

    if "Weekly_Sales" in df.columns:
        g = df.groupby("unique_id")["Weekly_Sales"]
        df["lag_1"]  = g.shift(1)
        df["lag_2"]  = g.shift(2)
        df["lag_4"]  = g.shift(4)
        df["lag_52"] = g.shift(52)
        df["rolling_mean_4"] = g.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        df["rolling_std_4"]  = g.transform(lambda x: x.shift(1).rolling(4, min_periods=1).std()).fillna(0)
    else:
        for col in LAG_COLUMNS:
            df[col] = np.nan

    return df.sort_values("_orig_order").drop(columns="_orig_order")   # <-- restore order

# --------------------------------------------------------------------------- #
# Seasonal profiles (target-dependent -> fit on TRAIN only)
# --------------------------------------------------------------------------- #
def fit_seasonal_profiles(train: pd.DataFrame) -> dict:
    """Average sales per (unique_id, week), (Dept, week), (Store, week), plus
    a global mean. These form the hierarchical fallback used at predict time."""
    tr = add_calendar_features(train)
    return {
        "series_week": tr.groupby(["unique_id", "week"])["Weekly_Sales"].mean(),
        "dept_week": tr.groupby(["Dept", "week"])["Weekly_Sales"].mean(),
        "store_week": tr.groupby(["Store", "week"])["Weekly_Sales"].mean(),
        "global_mean": float(tr["Weekly_Sales"].mean()),
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


# --------------------------------------------------------------------------- #
# Top-level builder for tree models
# --------------------------------------------------------------------------- #
def build_features(train: pd.DataFrame, other: pd.DataFrame,
                   *, n_fourier: int = 4, drop_markdowns: bool = True):
    """Fit seasonal profiles on `train`, apply the full feature set to both.
    Returns (train_fe, other_fe). `other` is valid or test.
    Keeps Store/Dept/Type as pandas 'category' so LightGBM handles them natively.
    """
    profiles = fit_seasonal_profiles(train)

    def _pipe(df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar_features(df)
        df = add_holiday_distance(df)
        df = add_fourier(df, n_terms=n_fourier)
        df = apply_seasonal_profiles(df, profiles)
        df["IsHoliday"] = df["IsHoliday"].astype(int)
        for c in CATEGORICAL:
            df[c] = df[c].astype("category")
        if drop_markdowns:
            df = df.drop(columns=[c for c in MARKDOWNS if c in df.columns])
        return df

    return _pipe(train), _pipe(other)

LAG_COLUMNS = ["lag_1", "lag_2", "lag_4", "lag_52", "rolling_mean_4", "rolling_std_4"]

FEATURE_COLUMNS = [
    "Store", "Dept", "Type", "Size", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "year", "month", "week", "dayofyear",
    "weeks_to_holiday", "weeks_since_holiday",
    "sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3", "sin_4", "cos_4",
    "seasonal_week_avg",
]

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
)
 
 
def apply_features_with_profiles(df: pd.DataFrame, profiles: dict,
                                  *, n_fourier: int = 4, drop_markdowns: bool = True, use_lags: bool = False,) -> pd.DataFrame:
    if use_lags:
      df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_holiday_distance(df)
    df = add_fourier(df, n_terms=n_fourier)
    df = apply_seasonal_profiles(df, profiles)

    if use_lags:
        for col in ["lag_1", "lag_2", "lag_4"]:
          df[col] = df[col].fillna(df["lag_52"]).fillna(df["seasonal_week_avg"])
        df["rolling_mean_4"] = df["rolling_mean_4"].fillna(df["seasonal_week_avg"])
        df["rolling_std_4"] = df["rolling_std_4"].fillna(0)
                                    
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    if drop_markdowns:
        df = df.drop(columns=[c for c in MARKDOWNS if c in df.columns])
    return df
 
 
def make_sample_weight(df: pd.DataFrame, holiday_weight: float = 5.0) -> np.ndarray:
    return np.where(df["IsHoliday"].astype(bool), holiday_weight, 1.0)
 
 
def fit_lightgbm(train: pd.DataFrame, *, lgbm_params: dict | None = None,
                 n_fourier: int = 4, drop_markdowns: bool = True,
                 holiday_weight: float = 5.0, use_lags: bool = False,) -> dict:
    from lightgbm import LGBMRegressor
 
    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier, drop_markdowns=drop_markdowns, use_lags=use_lags
    )

    feature_cols = list(FEATURE_COLUMNS) + (LAG_COLUMNS if use_lags else [])
                   
    params = dict(LGBM_DEFAULT_PARAMS)
    if lgbm_params:
        params.update(lgbm_params)
 
    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)
 
    booster = LGBMRegressor(**params)
    booster.fit(train_fe[feature_cols], y, sample_weight=w)
 
    return {
        "profiles": profiles,
        "booster": booster,
        "feature_columns": list(feature_cols),
        "n_fourier": n_fourier,
        "drop_markdowns": drop_markdowns,
        "use_lags": use_lags,
    }
 
 
def predict_lightgbm(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    original_index = raw_df.index

    df_fe = apply_features_with_profiles(
        raw_df,
        bundle["profiles"],
        n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"],
        use_lags=bundle.get("use_lags", False),
    )

    preds_log = bundle["booster"].predict(df_fe[bundle["feature_columns"]])
    preds = inverse_signed_log1p(preds_log)
    preds = np.clip(preds, 0, None)

    return pd.Series(preds, index=df_fe.index).reindex(original_index).to_numpy()


# --------------------------------------------------------------------------- #
# XGBoost Model Handlers
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
)


def fit_xgboost(
    train: pd.DataFrame,
    *,
    xgb_params: dict | None = None,
    n_fourier: int = 4,
    drop_markdowns: bool = True,
    holiday_weight: float = 5.0,
    use_lags: bool = False,
) -> dict:
    from xgboost import XGBRegressor

    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier, drop_markdowns=drop_markdowns, use_lags=use_lags
    )

    feature_cols = list(FEATURE_COLUMNS) + (LAG_COLUMNS if use_lags else [])

    params = dict(XGB_DEFAULT_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)

    booster = XGBRegressor(**params)
    booster.fit(train_fe[feature_cols], y, sample_weight=w)

    return {
        "profiles": profiles,
        "booster": booster,
        "feature_columns": feature_cols,
        "n_fourier": n_fourier,
        "drop_markdowns": drop_markdowns,
        "use_lags": use_lags,
    }


def predict_xgboost(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    df_fe = apply_features_with_profiles(
        raw_df,
        bundle["profiles"],
        n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"],
        use_lags=bundle.get("use_lags", False),
    )
    preds_log = bundle["booster"].predict(df_fe[bundle["feature_columns"]])
    preds = inverse_signed_log1p(preds_log)
    return np.clip(preds, 0, None)

# --------------------------------------------------------------------------- #
# CatBoost Model Handlers
# --------------------------------------------------------------------------- #
CATBOOST_DEFAULT_PARAMS = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)


def fit_catboost(
    train: pd.DataFrame,
    *,
    cat_params: dict | None = None,
    n_fourier: int = 4,
    drop_markdowns: bool = True,
    holiday_weight: float = 5.0,
    use_lags: bool = False,
) -> dict:
    from catboost import CatBoostRegressor, Pool

    profiles = fit_seasonal_profiles(train)
    train_fe = apply_features_with_profiles(
        train, profiles, n_fourier=n_fourier, drop_markdowns=drop_markdowns, use_lags=use_lags
    )

    feature_cols = list(FEATURE_COLUMNS) + (LAG_COLUMNS if use_lags else [])

    params = dict(CATBOOST_DEFAULT_PARAMS)
    if cat_params:
        params.update(cat_params)

    y = signed_log1p(train_fe["Weekly_Sales"])
    w = make_sample_weight(train_fe, holiday_weight=holiday_weight)

    # Convert categorical columns to string/category format for CatBoost
    cat_features = [c for c in CATEGORICAL if c in feature_cols]
    train_fe_cb = train_fe[feature_cols].copy()
    for col in cat_features:
        train_fe_cb[col] = train_fe_cb[col].astype(str)

    train_pool = Pool(
        data=train_fe_cb,
        label=y,
        weight=w,
        cat_features=cat_features,
    )

    booster = CatBoostRegressor(**params)
    booster.fit(train_pool)

    return {
        "profiles": profiles,
        "booster": booster,
        "feature_columns": feature_cols,
        "cat_features": cat_features,
        "n_fourier": n_fourier,
        "drop_markdowns": drop_markdowns,
        "use_lags": use_lags,
    }


def predict_catboost(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    from catboost import Pool

    original_index = raw_df.index

    df_fe = apply_features_with_profiles(
        raw_df,
        bundle["profiles"],
        n_fourier=bundle["n_fourier"],
        drop_markdowns=bundle["drop_markdowns"],
        use_lags=bundle.get("use_lags", False),
    )

    eval_df = df_fe[bundle["feature_columns"]].copy()
    cat_features = bundle.get("cat_features", [])
    for col in cat_features:
        eval_df[col] = eval_df[col].astype(str)

    eval_pool = Pool(data=eval_df, cat_features=cat_features)

    preds_log = bundle["booster"].predict(eval_pool)
    preds = inverse_signed_log1p(preds_log)
    preds = np.clip(preds, 0, None)

    preds_series = pd.Series(preds, index=df_fe.index)
    return preds_series.reindex(original_index).to_numpy()
