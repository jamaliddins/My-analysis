"""Load and clean the Amazon Sale Report export.

The raw CSV is not committed (it is ~50 MB and redistributable only from its
Kaggle source). Download it and place it in the repository root as
`Amazon Sale Report.csv` — see the README.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Amazon Sale Report.csv"
SAMPLE_CSV = ROOT / "amazon_sales_clean_sample.csv"

STRING_COLUMNS = [
    "Category", "Size", "Status", "Fulfilment", "Style", "Courier Status",
    "currency",
]

# Anything above this is a data-entry error rather than a real bulk order.
MAX_PLAUSIBLE_QTY = 100

SIZE_ORDER = {
    "Xs": 1, "S": 2, "M": 3, "L": 4, "Xl": 5, "Xxl": 6, "3Xl": 7,
    "4Xl": 8, "5Xl": 9, "6Xl": 10, "Free": 11, "Unspecified": 12,
}


class MissingDataError(FileNotFoundError):
    """Raised when the raw export is not where the script expects it."""


def load_raw(path: pathlib.Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw export, failing with instructions if it is absent."""
    if not path.exists():
        raise MissingDataError(
            f"Could not find {path.name} in {path.parent}.\n"
            "This file is not committed. Download 'Amazon Sale Report.csv' from\n"
            "  https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data\n"
            f"and save it as: {path}"
        )
    return pd.read_csv(path, low_memory=False)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the sale date into the parts the dashboard slices on."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["WeekNumber"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    # Note the call: `dt.day_name` without parentheses stores the bound method
    # itself, which silently fills the column with method objects.
    df["DayOfWeek"] = df["Date"].dt.day_name()
    return df


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalise casing on the categorical columns."""
    df = df.copy()
    for column in STRING_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().str.title()
    return df


def handle_missing(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows that cannot be analysed; label the rest."""
    df = df.copy()
    before = len(df)

    # Amount and Qty carry every downstream statistic, so a row missing either
    # cannot contribute to any test.
    df = df.dropna(subset=["Amount", "Qty"])

    df["Courier Status"] = df["Courier Status"].fillna("Unknown")
    df["Size"] = df["Size"].fillna("Unspecified")
    df["Style"] = df["Style"].fillna("Unspecified")

    if verbose:
        print(f"  dropped {before - len(df):,} rows missing Amount or Qty")
    return df


def filter_invalid(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only rows with a positive quantity and amount."""
    df = df.copy()
    before = len(df)

    df = df[(df["Qty"] > 0) & (df["Amount"] > 0)]
    df = df[df["Qty"] <= MAX_PLAUSIBLE_QTY]

    if verbose:
        print(f"  dropped {before - len(df):,} rows with non-positive or "
              f"implausible Qty/Amount")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build the flags and buckets the hypothesis tests group by."""
    df = df.copy()

    # B2B arrives as bool, string or 0/1 depending on how the file was exported.
    b2b = df["B2B"].astype("string").str.strip().str.lower()
    df["B2B"] = b2b.map({
        "true": True, "1": True, "1.0": True, "yes": True,
        "false": False, "0": False, "0.0": False, "no": False,
    }).fillna(False).astype(bool)
    df["B2BLabel"] = df["B2B"].map({True: "B2B", False: "B2C"})

    df["RevenuePerUnit"] = df["Amount"] / df["Qty"]

    df["OrderSizeBucket"] = pd.cut(
        df["Qty"],
        bins=[0, 1, 5, 20, MAX_PLAUSIBLE_QTY],
        labels=["Single (1)", "Small (2-5)", "Medium (6-20)", "Bulk (20+)"],
    )

    status = df["Status"].astype("string").str.lower()
    df["IsCancelled"] = status.str.contains("cancel", na=False).astype(int)
    df["IsDelivered"] = (
        df["Courier Status"].astype("string").str.lower() == "delivered"
    ).astype(int)

    df["SizeRank"] = df["Size"].map(SIZE_ORDER).fillna(99).astype(int)
    return df


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    """Full pipeline: raw CSV -> analysis-ready frame."""
    if verbose:
        print("Loading raw export...")
    df = load_raw()
    if verbose:
        print(f"  {len(df):,} raw rows")

    df = add_date_features(df)
    df = standardise_strings(df)
    df = handle_missing(df, verbose)
    df = filter_invalid(df, verbose)
    df = add_derived_columns(df)

    if verbose:
        print(f"  {len(df):,} rows ready for analysis")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = build_dataset()
    data.head(50).to_csv(SAMPLE_CSV, index=False)
    print(f"\nwrote a 50-row sample to {SAMPLE_CSV.name}")
