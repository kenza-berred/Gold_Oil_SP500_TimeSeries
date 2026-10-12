
from pathlib import Path

import pandas as pd
import yfinance as yf

# Define project folders relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Select three financial assets.
TICKERS = {
    "Gold": "GC=F",
    "Oil": "CL=F",
    "SP500": "^GSPC",
}

START_DATE = "2021-10-01"
END_DATE = "2026-10-02"  # Yahoo Finance end date is exclusive.

# Download daily market data.
downloaded = yf.download(
    list(TICKERS.values()),
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=False,
    group_by="column",
    progress=False,
)

if downloaded.empty:
    raise RuntimeError(
        "No data downloaded. Check your internet connection and try again."
    )

print("Downloaded column structure:")
print(downloaded.columns)

# Extract the Close and Adjusted Close prices.
close_data = downloaded["Close"].copy()
adjusted_close_data = downloaded["Adj Close"].copy()

# Rename tickers to readable asset names.
rename_map = {ticker: name for name, ticker in TICKERS.items()}
close_data = close_data.rename(columns=rename_map)
adjusted_close_data = adjusted_close_data.rename(columns=rename_map)

# Sort dates and remove duplicate timestamps.
close_data = close_data.sort_index()
close_data = close_data.loc[~close_data.index.duplicated(keep="first")]

# Save original downloaded data without dropping missing observations.
close_data.to_csv(RAW_DIR / "daily_close_prices.csv")
adjusted_close_data.to_csv(RAW_DIR / "daily_adjusted_close_prices.csv")

# Inspect the data before any cleaning.
print("\nDataset shape:", close_data.shape)
print("\nFirst five rows:")
print(close_data.head())

print("\nLast five rows:")
print(close_data.tail())

print("\nMissing values per asset:")
print(close_data.isna().sum())

print("\nDate range:")
print("Start:", close_data.index.min())
print("End:", close_data.index.max())

print("\nDuplicate dates:", close_data.index.duplicated().sum())

print("\nData types:")
print(close_data.dtypes)

print("\nFiles saved in:", RAW_DIR)