
from pathlib import Path
import math

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Load the synchronized, cleaned closing prices.
prices = pd.read_csv(
    PROCESSED_DIR / "prices_clean.csv",
    index_col=0,
    parse_dates=True,
).sort_index()

# Use the first 90% of observations for training.
split_position = math.floor(len(prices) * 0.90)

train_prices = prices.iloc[:split_position].copy()
test_prices = prices.iloc[split_position:].copy()

# Calculate returns from prices before separating the periods.
# This preserves the return between the final training price
# and the first test price as the first out-of-sample return.
returns = (prices / prices.shift(1)).apply(
    lambda column: __import__("numpy").log(column)
).dropna()

train_returns = returns.loc[returns.index <= train_prices.index[-1]]
test_returns = returns.loc[returns.index > train_prices.index[-1]]

# Save each dataset for reproducible downstream analysis.
train_prices.to_csv(PROCESSED_DIR / "train_prices.csv")
test_prices.to_csv(PROCESSED_DIR / "test_prices.csv")
train_returns.to_csv(PROCESSED_DIR / "train_returns.csv")
test_returns.to_csv(PROCESSED_DIR / "test_returns.csv")

print("Total price observations:", len(prices))
print("Training price observations:", len(train_prices))
print("Testing price observations:", len(test_prices))

print("\nTraining period:")
print(train_prices.index.min(), "to", train_prices.index.max())

print("\nTesting period:")
print(test_prices.index.min(), "to", test_prices.index.max())

print("\nTraining returns:", len(train_returns))
print("Testing returns:", len(test_returns))

# Verify that the split is strictly chronological.
assert train_prices.index.max() < test_prices.index.min()
assert len(train_prices) + len(test_prices) == len(prices)

print("\nChronological split validation passed.")