
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_returns.csv"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test_returns.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "python" / "garch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN_PATH, index_col=0, parse_dates=True)
test = pd.read_csv(TEST_PATH, index_col=0, parse_dates=True)

train = train.apply(pd.to_numeric, errors="coerce")
test = test.apply(pd.to_numeric, errors="coerce")

if train.isna().any().any() or test.isna().any().any():
    raise ValueError("Missing values found in train or test returns.")

if not train.index.is_monotonic_increasing:
    raise ValueError("Training dates are not chronological.")

if not test.index.is_monotonic_increasing:
    raise ValueError("Testing dates are not chronological.")

if list(train.columns) != list(test.columns):
    raise ValueError("Train and test assets do not match.")

if train.index.max() >= test.index.min():
    raise ValueError("Train/test date ranges overlap.")

forecast_rows = []
metric_rows = []

for asset in train.columns:
    # Fit the model on training data only.
    train_pct = train[asset] * 100
    test_pct = test[asset] * 100

    model = arch_model(
        train_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    fitted = model.fit(disp="off", show_warning=False)

    if fitted.convergence_flag != 0:
        raise RuntimeError(f"GARCH did not converge for {asset}.")

    # Use test returns only as subsequent observations, not for refitting parameters.
    combined_pct = pd.concat([train_pct, test_pct])

    forecasts = fitted.forecast(
        horizon=1,
        start=len(train_pct) - 1,
        align="origin",
        method="analytic",
        reindex=False,
    )

    # The forecast generated at the final training date is the first test-day forecast.
    # To update volatility each day without refitting parameters, use fixed parameters
    # with the complete return history through each forecast origin.
    params = fitted.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    mu = params["mu"]

    last_train_residual = train_pct.iloc[-1] - mu
    last_train_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)

    # First forecast: variance for the first test day,
    # conditional on information available through the last training day.
    variance_pct = (
    omega
    + alpha * last_train_residual**2
    + beta * last_train_variance
)
        # Forecast first test day, then update the variance with each observed test return.
    for date, observed_return_pct in test_pct.items():
            forecast_rows.append({
                "Date": date,
                "Asset": asset,
                "Actual_Return": observed_return_pct / 100,
                "Actual_Squared_Return": (observed_return_pct / 100) ** 2,
                "Forecast_Variance": variance_pct / (100 ** 2),
                "Forecast_Volatility": np.sqrt(variance_pct) / 100,
            })

            residual = observed_return_pct - mu
            variance_pct = omega + alpha * residual**2 + beta * variance_pct

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path = OUTPUT_DIR / "garch11_student_t_test_forecasts.csv"
    forecast_df.to_csv(forecast_path, index=False)

# Compare forecasts against the squared daily return proxy.
for asset, group in forecast_df.groupby("Asset"):
    actual_var_proxy = group["Actual_Squared_Return"].to_numpy()
    forecast_var = group["Forecast_Variance"].to_numpy()

    mse = np.mean((actual_var_proxy - forecast_var) ** 2)
    mae = np.mean(np.abs(actual_var_proxy - forecast_var))

    # Gaussian QLIKE-style loss, used as a comparative volatility score.
    eps = 1e-12
    qlike = np.mean(
        np.log(np.maximum(forecast_var, eps))
        + actual_var_proxy / np.maximum(forecast_var, eps)
    )

    # Simple constant-variance baseline estimated from training returns only.
    baseline_var = float(train[asset].var(ddof=1))
    baseline_forecast = np.full(len(group), baseline_var)

    baseline_mse = np.mean((actual_var_proxy - baseline_forecast) ** 2)
    baseline_mae = np.mean(np.abs(actual_var_proxy - baseline_forecast))
    baseline_qlike = np.mean(
        np.log(np.maximum(baseline_forecast, eps))
        + actual_var_proxy / np.maximum(baseline_forecast, eps)
    )

    metric_rows.append({
        "Asset": asset,
        "Test_Observations": len(group),
        "GARCH_MSE": mse,
        "Baseline_MSE": baseline_mse,
        "GARCH_MAE": mae,
        "Baseline_MAE": baseline_mae,
        "GARCH_QLIKE": qlike,
        "Baseline_QLIKE": baseline_qlike,
        "GARCH_Better_QLIKE": qlike < baseline_qlike,
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_path = OUTPUT_DIR / "garch11_student_t_forecast_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("Forecasts saved to:", forecast_path)
print("Metrics saved to:", metrics_path)
print("\nForecast date range:")
print(forecast_df.groupby("Asset")["Date"].agg(["min", "max", "count"]))
print("\nVolatility forecast evaluation:")
print(metrics_df.to_string(index=False))
