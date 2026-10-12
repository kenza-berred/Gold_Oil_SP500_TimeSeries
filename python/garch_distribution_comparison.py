
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
    raise ValueError("Missing values found in train or test data.")

if list(train.columns) != list(test.columns):
    raise ValueError("Train/test asset columns do not match.")

if train.index.max() >= test.index.min():
    raise ValueError("Training and testing periods overlap.")

forecast_rows = []
parameter_rows = []

for asset in train.columns:
    train_pct = train[asset] * 100
    test_pct = test[asset] * 100

    for distribution in ["t", "normal"]:
        model = arch_model(
            train_pct,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist=distribution,
            rescale=False,
        )

        fitted = model.fit(disp="off", show_warning=False)

        if fitted.convergence_flag != 0:
            raise RuntimeError(
                f"{asset} GARCH model did not converge for {distribution}."
            )

        params = fitted.params
        omega = params["omega"]
        alpha = params["alpha[1]"]
        beta = params["beta[1]"]
        mu = params["mu"]

        last_residual = train_pct.iloc[-1] - mu
        last_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)

        # One-step-ahead variance forecast for the first test observation.
        variance = omega + alpha * last_residual**2 + beta * last_variance

        parameter_rows.append({
            "Asset": asset,
            "Distribution": distribution,
            "Converged": True,
            "LogLikelihood": fitted.loglikelihood,
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "mu": mu,
            "omega": omega,
            "alpha1": alpha,
            "beta1": beta,
            "alpha_plus_beta": alpha + beta,
            "nu": params.get("nu", np.nan),
        })

        for date, observed_pct in test_pct.items():
            actual_return = observed_pct / 100
            forecast_variance = variance / (100**2)

            forecast_rows.append({
                "Date": date,
                "Asset": asset,
                "Distribution": distribution,
                "Actual_Return": actual_return,
                "Actual_Squared_Return": actual_return**2,
                "Forecast_Variance": forecast_variance,
                "Forecast_Volatility": np.sqrt(forecast_variance),
            })

            # Update conditional variance using the observed return,
            # while keeping training-estimated parameters fixed.
            residual = observed_pct - mu
            variance = omega + alpha * residual**2 + beta * variance

forecasts = pd.DataFrame(forecast_rows)
parameters = pd.DataFrame(parameter_rows)

forecast_path = OUTPUT_DIR / "garch_distribution_test_forecasts.csv"
parameter_path = OUTPUT_DIR / "garch_distribution_parameters.csv"
metric_path = OUTPUT_DIR / "garch_distribution_comparison_metrics.csv"

forecasts.to_csv(forecast_path, index=False)
parameters.to_csv(parameter_path, index=False)

metric_rows = []
eps = 1e-12

for (asset, distribution), group in forecasts.groupby(
    ["Asset", "Distribution"]
):
    actual = group["Actual_Squared_Return"].to_numpy()
    predicted = group["Forecast_Variance"].to_numpy()
    predicted = np.maximum(predicted, eps)

    metric_rows.append({
        "Asset": asset,
        "Distribution": distribution,
        "Test_Observations": len(group),
        "MSE": np.mean((actual - predicted) ** 2),
        "MAE": np.mean(np.abs(actual - predicted)),
        "QLIKE": np.mean(np.log(predicted) + actual / predicted),
    })

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(metric_path, index=False)

print("\n=== Parameter comparison ===")
print(parameters.to_string(index=False))

print("\n=== Out-of-sample volatility metrics ===")
print(metrics.to_string(index=False))

print("\nSaved files:")
print(parameter_path)
print(forecast_path)
print(metric_path)
