"""Time-based validation that mirrors the real forecasting task.

The single most important idea here: the test set runs Nov 2012 -> Jul 2013
(~39 weeks) and CONTAINS Thanksgiving, Christmas, and the Super Bowl. But train
ends 2012-11-01, just BEFORE the 2012 holiday season, so the most recent weeks
of train have no Thanksgiving/Christmas. To validate on a window with the same
length AND the same holiday composition as the test, hold out one year earlier
(Nov 2011 -> ~Aug 2012). That is what `seasonal_holdout_split` does, and it is
the metric you should trust most. Never use a random split; it leaks the future.
"""
import numpy as np
import pandas as pd


def seasonal_holdout_split(df, horizon_weeks: int = 39, date_col: str = "Date"):
    """Primary validation. Holds out a window one year back so it aligns with the
    test period's season and holidays. Returns (train_df, valid_df)."""
    max_date = df[date_col].max()
    valid_start = max_date - pd.Timedelta(weeks=52)
    valid_end = valid_start + pd.Timedelta(weeks=horizon_weeks)
    train = df[df[date_col] < valid_start]
    valid = df[(df[date_col] >= valid_start) & (df[date_col] < valid_end)]
    return train.copy(), valid.copy()


def holdout_split(df, horizon_weeks: int = 39, date_col: str = "Date"):
    """Alternative: simply hold out the most recent `horizon_weeks` (max recency,
    but note it will NOT contain Thanksgiving/Christmas). Returns (train, valid)."""
    dates = np.sort(pd.unique(df[date_col]))
    cutoff = dates[-horizon_weeks]
    train = df[df[date_col] < cutoff]
    valid = df[df[date_col] >= cutoff]
    return train.copy(), valid.copy()


class RollingOriginSplitter:
    """Expanding-window CV for panel data, keyed on Date (not row index).

    Assumes weekly data, so `horizon_weeks`/`step_weeks` count distinct dates.
    Yields (train_idx, valid_idx) as pandas Index objects, oldest fold first.
    Use these for stability across time; use seasonal_holdout_split as the
    headline number. With ~143 weekly dates, keep horizon modest for >1 fold.
    """

    def __init__(self, horizon_weeks: int = 13, n_splits: int = 3,
                 step_weeks: int | None = None, gap_weeks: int = 0):
        self.horizon_weeks = horizon_weeks
        self.n_splits = n_splits
        self.step_weeks = step_weeks or horizon_weeks
        self.gap_weeks = gap_weeks

    def split(self, df, date_col: str = "Date"):
        dates = np.sort(pd.unique(df[date_col]))
        n, H, S, G = len(dates), self.horizon_weeks, self.step_weeks, self.gap_weeks
        folds = []
        for i in range(self.n_splits):
            v_end = n - i * S
            v_start = v_end - H
            t_end = v_start - G
            if t_end <= 0 or v_start < 0:
                break
            train_dates, valid_dates = dates[:t_end], dates[v_start:v_end]
            tr = df.index[df[date_col].isin(train_dates)]
            va = df.index[df[date_col].isin(valid_dates)]
            folds.append((tr, va))
        return list(reversed(folds))