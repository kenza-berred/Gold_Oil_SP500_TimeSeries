
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load the raw closing prices.
prices = pd.read_csv(
    RAW_DIR / "daily_close_prices.csv",
    index_col=0,
    parse_dates=True,
)

# Ensure the data is ordered chronologically.
prices = prices.sort_index()

# Convert columns to numeric and represent invalid values as missing.
prices = prices.apply(pd.to_numeric, errors="coerce")

# Inspect missing values before deciding how to handle them.
print("Original shape:", prices.shape)
print("\nMissing values:")
print(prices.isna().sum())

print("\nMissing rows:")
print(prices[prices.isna().any(axis=1)].head(10))

# Check duplicate dates and invalid/non-positive prices.
print("\nDuplicate dates:", prices.index.duplicated().sum())
print("\nNon-positive prices:")
print((prices <= 0).sum())

# Remove duplicate timestamps, keeping the first occurrence.
prices = prices.loc[~prices.index.duplicated(keep="first")]

# Keep only dates where all three assets have valid closing prices.
# This creates a synchronized dataset for multivariate analysis.
prices_clean = prices.dropna().copy()
prices_clean = prices_clean[(prices_clean > 0).all(axis=1)]

print("\nCleaned shape:", prices_clean.shape)
print("\nCleaned date range:")
print(prices_clean.index.min(), "to", prices_clean.index.max())

# Calculate daily logarithmic returns.
# Log returns are useful for analyzing relative price changes.
log_returns = np.log(prices_clean / prices_clean.shift(1))
log_returns = log_returns.dropna()

# Calculate a 20-observation rolling volatility for each asset.
rolling_volatility = log_returns.rolling(window=20).std()

# Normalize prices to 100 at the first synchronized observation.
normalized_prices = prices_clean.div(prices_clean.iloc[0]).mul(100)

# Save processed datasets.
prices_clean.to_csv(PROCESSED_DIR / "prices_clean.csv")
log_returns.to_csv(PROCESSED_DIR / "log_returns.csv")
rolling_volatility.to_csv(PROCESSED_DIR / "rolling_volatility_20d.csv")
normalized_prices.to_csv(PROCESSED_DIR / "normalized_prices.csv")

print("\nLog-return observations:", len(log_returns))
print("\nLog-return summary:")
print(log_returns.describe())

print("\nSaved processed datasets to:", PROCESSED_DIR)