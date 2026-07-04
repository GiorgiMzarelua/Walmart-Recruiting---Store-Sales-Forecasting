"""Load, extract, and merge the raw Walmart competition files into one frame.

Every notebook starts here. This module locks the decisions that MUST be
identical across all models: the merge, the date parsing, and the sort order.
"""
from pathlib import Path
import zipfile
import pandas as pd

DATA_DIR = "data"
RAW_FILES = ["train.csv", "test.csv", "features.csv", "stores.csv"]


def prepare_data_dir(data_dir: str = DATA_DIR) -> None:
    """Handle the competition's double-zip: extract the outer archive, then any
    inner *.csv.zip. Safe to call repeatedly; does nothing if CSVs already exist."""
    data_dir = Path(data_dir)
    outer = data_dir / "walmart-recruiting-store-sales-forecasting.zip"
    if outer.exists():
        with zipfile.ZipFile(outer) as z:
            z.extractall(data_dir)
    for zp in data_dir.glob("*.zip"):
        if zp.name == outer.name:
            continue
        with zipfile.ZipFile(zp) as z:
            z.extractall(data_dir)


def load_raw(data_dir: str = DATA_DIR):
    """Return (train, test, features, stores) as raw DataFrames with Date parsed."""
    data_dir = Path(data_dir)
    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        prepare_data_dir(data_dir)
    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in '{data_dir}'. Download the competition data "
            f"into that folder first (kaggle competitions download -c "
            f"walmart-recruiting-store-sales-forecasting -p {data_dir})."
        )
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["Date"])
    feats = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    return train, test, feats, stores


def _merge_one(df: pd.DataFrame, feats: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    # features.csv also carries IsHoliday; drop it so we keep the train/test copy
    feats = feats.drop(columns=["IsHoliday"], errors="ignore")
    df = df.merge(stores, on="Store", how="left")
    df = df.merge(feats, on=["Store", "Date"], how="left")
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    sort_cols = ["Store", "Dept", "Date"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def load_data(data_dir: str = DATA_DIR):
    """Main entry point. Return (train, test) merged and ready to use.

    train has Weekly_Sales; test does not. Both carry:
    Store, Dept, Date, IsHoliday, Type, Size, Temperature, Fuel_Price,
    MarkDown1-5, CPI, Unemployment, unique_id.
    """
    train, test, feats, stores = load_raw(data_dir)
    return _merge_one(train, feats, stores), _merge_one(test, feats, stores)