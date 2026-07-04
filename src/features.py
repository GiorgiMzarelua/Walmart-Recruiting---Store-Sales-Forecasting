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


FEATURE_COLUMNS = [
    "Store", "Dept", "Type", "Size", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "year", "month", "week", "dayofyear",
    "weeks_to_holiday", "weeks_since_holiday",
    "sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3", "sin_4", "cos_4",
    "seasonal_week_avg",
]