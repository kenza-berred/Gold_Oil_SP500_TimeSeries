
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "python" / "diebold_mariano"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "ARIMA": (
        "results/python/arima/arima_forecasts_test.csv",
        "Date", "Asset", "Actual Return", "ARIMA Forecast",
    ),
    "SARIMA": (
        "results/python/sarima/sarima_forecasts.csv",
        "Date", "Asset", "Actual_Return", "SARIMA_Forecast",
    ),
    "LSTM": (
        "results/python/deep_learning_multistep/lstm_test_forecasts.csv",
        "Date", "Asset", "Actual", "Predicted",
    ),
    "GRU": (
        "results/python/deep_learning_multistep/gru_test_forecasts.csv",
        "Date", "Asset", "Actual", "Predicted",
    ),
}

def load_forecasts(name, config):
    file, date_col, asset_col, actual_col, forecast_col = config
    df = pd.read_csv(ROOT / file)

    if "Horizon_Step" in df.columns:
        df = df[df["Horizon_Step"] == 1].copy()

    df = df.rename(columns={
        date_col: "Date",
        asset_col: "Asset",
        actual_col: "Actual",
        forecast_col: "Forecast",
    })
    df["Date"] = pd.to_datetime(df["Date"])

    if df.duplicated(["Date", "Asset"]).any():
        raise ValueError(f"Duplicate Date/Asset rows in {name}")

    return df[["Date", "Asset", "Actual", "Forecast"]].rename(
        columns={"Forecast": name, "Actual": f"Actual_{name}"}
    )

frames = {
    name: load_forecasts(name, config)
    for name, config in FILES.items()
}

# Keep only dates/assets present in every model.
merged = None
for name, df in frames.items():
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=["Date", "Asset"], how="inner")

actual_cols = [f"Actual_{name}" for name in FILES]
if not all(
    np.allclose(merged[actual_cols[0]], merged[col], equal_nan=True)
    for col in actual_cols[1:]
):
    raise ValueError("Actual returns differ between model files.")

merged["Baseline"] = 0.0
merged.to_csv(OUT / "aligned_forecasts.csv", index=False)

# DM test on squared-error loss.
# These are 25 non-overlapping block-start dates per asset.
# Since the compared forecast errors are not overlapping in time,
# use lag-zero variance of the loss differential.
models = list(FILES.keys()) + ["Baseline"]
results = []

for asset in sorted(merged["Asset"].unique()):
    subset = merged[merged["Asset"] == asset].sort_values("Date")

    for model_a, model_b in itertools.combinations(models, 2):
        pred_a = (
            subset[model_a].to_numpy()
            if model_a == "Baseline"
            else subset[model_a].to_numpy()
        )
        pred_b = (
            subset[model_b].to_numpy()
            if model_b == "Baseline"
            else subset[model_b].to_numpy()
        )
        actual = subset["Actual_ARIMA"].to_numpy()

        loss_diff = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
        n = len(loss_diff)
        mean_diff = float(np.mean(loss_diff))
        sd_diff = float(np.std(loss_diff, ddof=1))

        if n < 3 or not np.isfinite(sd_diff) or sd_diff == 0:
            statistic, p_value = np.nan, np.nan
        else:
            raw_stat = mean_diff / (sd_diff / np.sqrt(n))

            # Harvey-Leybourne-Newbold small-sample correction for h=1.
            h = 1
            correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
            statistic = raw_stat * correction
            p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)

        results.append({
            "Asset": asset,
            "Model_A": model_a,
            "Model_B": model_b,
            "N": n,
            "Mean_Squared_Loss_Difference_A_minus_B": mean_diff,
            "DM_Statistic": statistic,
            "P_Value": p_value,
            "Interpretation": (
                "A has higher squared loss" if mean_diff > 0
                else "A has lower squared loss" if mean_diff < 0
                else "Equal squared loss"
            ),
        })

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "dm_test_results.csv", index=False)

print("Aligned observations:", len(merged))
print("Observations per asset:", merged.groupby("Asset").size().to_dict())
print("\nDM test results:")
print(result_df.to_string(index=False))
print("\nSaved to:", OUT)
