
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "python" / "var"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_returns.csv"
TEST_FILE = DATA_DIR / "test_returns.csv"

MAX_LAGS = 10
ASSETS = ["Oil", "Gold", "SP500"]


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------
def load_returns(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    missing_columns = set(ASSETS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {path.name}: {missing_columns}")

    df = df[ASSETS].astype(float)

    if df.index.has_duplicates:
        raise ValueError(f"Duplicate dates in {path.name}")

    if df.isna().any().any():
        raise ValueError(f"Missing values in {path.name}")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Non-finite values in {path.name}")

    return df


train = load_returns(TRAIN_FILE)
test = load_returns(TEST_FILE)

if train.index.max() >= test.index.min():
    raise ValueError("Training and test periods overlap or are incorrectly ordered.")

if list(train.columns) != list(test.columns):
    raise ValueError("Train and test columns do not match.")

print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Training period:", train.index.min(), "to", train.index.max())
print("Testing period:", test.index.min(), "to", test.index.max())


# --------------------------------------------------
# 3. Select the lag order using training data only
# --------------------------------------------------
var_model = VAR(train)

maxlags = min(
    MAX_LAGS,
    max(1, (len(train) - len(ASSETS) - 1) // (len(ASSETS) + 1))
)

lag_selection = var_model.select_order(maxlags=maxlags, trend="c")

# statsmodels select_order reports lags as integers where 0 means
# "no lags improve the criterion" (i.e. a VAR(0) white-noise process).
# We record the raw criterion output in its own column so it is never
# confused with the lag that was actually used for model estimation.
raw_bic_lag = int(lag_selection.bic)

# A VAR(0) contains only intercepts and carries no cross-variable
# dynamics. We therefore apply a minimum-lag floor of 1 to obtain
# a proper dynamic VAR, and document this decision explicitly.
if raw_bic_lag < 1:
    selected_lag = 1
    lag_override_note = (
        "Raw BIC selected lag 0 (no-lag model). "
        "Minimum-lag floor applied: VAR(1) used for estimation."
    )
else:
    selected_lag = raw_bic_lag
    lag_override_note = "Raw BIC selection used directly."

selection_table = pd.DataFrame({
    "criterion": ["AIC", "BIC", "HQIC", "FPE"],
    "raw_criterion_lag": [
        int(lag_selection.aic),
        int(lag_selection.bic),
        int(lag_selection.hqic),
        int(lag_selection.fpe),
    ],
    "lag_used_for_estimation": selected_lag,
    "note": [
        "Informational only; BIC used as primary criterion.",
        lag_override_note,
        "Informational only; BIC used as primary criterion.",
        "Informational only; BIC used as primary criterion.",
    ],
})

selection_table.to_csv(
    OUTPUT_DIR / "var_lag_selection.csv", index=False
)

print("\nLag-order selection:")
print(lag_selection.summary())

print(f"\nSelected VAR lag order: {selected_lag}")

# --------------------------------------------------
# 4. Fit VAR on the training returns
# --------------------------------------------------
fitted_model = var_model.fit(selected_lag, trend="c")

with open(OUTPUT_DIR / "var_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(fitted_model.summary()))
    
print("\nVAR model fitted successfully.")
print(fitted_model.summary())


# --------------------------------------------------
# 5. Residual diagnostics
# --------------------------------------------------
residuals = pd.DataFrame(
    fitted_model.resid,
    index=train.index[selected_lag:],
    columns=ASSETS,
)

residuals.to_csv(OUTPUT_DIR / "var_residuals.csv")

# Residual whiteness test
diagnostic_rows = []

whiteness_lag = max(12, selected_lag + 1)
if whiteness_lag < len(residuals):
    whiteness = fitted_model.test_whiteness(
        nlags=whiteness_lag,
        adjusted=True,
    )

    diagnostic_rows.append({
        "test": "Portmanteau residual whiteness",
        "statistic": float(whiteness.test_statistic),
        "p_value": float(whiteness.pvalue),
        "lags": whiteness_lag,
    })

# Multivariate residual normality test
normality = fitted_model.test_normality(signif=0.05)

diagnostic_rows.append({
    "test": "Multivariate residual normality",
    "statistic": float(normality.test_statistic),
    "p_value": float(normality.pvalue),
    "lags": np.nan,
})

pd.DataFrame(diagnostic_rows).to_csv(
    OUTPUT_DIR / "var_diagnostics.csv", index=False
)

print("\nResidual diagnostics:")
print(pd.DataFrame(diagnostic_rows).to_string(index=False))


# --------------------------------------------------
# 6. Granger causality tests
# --------------------------------------------------
causality_rows = []

for caused_asset in ASSETS:
    for causing_asset in ASSETS:
        if caused_asset == causing_asset:
            continue

        result = fitted_model.test_causality(
            caused=caused_asset,
            causing=[causing_asset],
            kind="f",
        )

        causality_rows.append({
            "causing": causing_asset,
            "caused": caused_asset,
            "test_statistic": float(result.test_statistic),
            "p_value": float(result.pvalue),
            "significant_at_5pct": bool(result.pvalue < 0.05),
        })

causality_df = pd.DataFrame(causality_rows)
causality_df.to_csv(
    OUTPUT_DIR / "var_granger_causality.csv", index=False
)

print("\nGranger causality results:")
print(causality_df.to_string(index=False))


# --------------------------------------------------
# 7. Forecast the entire test horizon
#    Fixed-origin multi-step forecast
# --------------------------------------------------
forecast_values = fitted_model.forecast(
    y=train.to_numpy()[-selected_lag:],
    steps=len(test),
)

forecast_df = pd.DataFrame(
    forecast_values,
    index=test.index,
    columns=ASSETS,
)

forecast_df.index.name = "Date"
forecast_df.to_csv(OUTPUT_DIR / "var_forecasts_test.csv")

# Compare VAR with the zero-return baseline
metric_rows = []

for asset in ASSETS:
    actual = test[asset].to_numpy()
    predicted = forecast_df[asset].to_numpy()
    baseline = np.zeros(len(test))

    var_mae = mean_absolute_error(actual, predicted)
    var_rmse = np.sqrt(mean_squared_error(actual, predicted))

    baseline_mae = mean_absolute_error(actual, baseline)
    baseline_rmse = np.sqrt(mean_squared_error(actual, baseline))

    metric_rows.append({
        "asset": asset,
        "selected_lag": selected_lag,
        "VAR_MAE": var_mae,
        "VAR_RMSE": var_rmse,
        "zero_baseline_MAE": baseline_mae,
        "zero_baseline_RMSE": baseline_rmse,
        "MAE_improvement_pct": (
            (baseline_mae - var_mae) / baseline_mae * 100
            if baseline_mae != 0 else np.nan
        ),
        "RMSE_improvement_pct": (
            (baseline_rmse - var_rmse) / baseline_rmse * 100
            if baseline_rmse != 0 else np.nan
        ),
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(
    OUTPUT_DIR / "var_forecast_metrics.csv", index=False
)

print("\nOut-of-sample forecast metrics:")
print(metrics_df.round(6).to_string(index=False))


# --------------------------------------------------
# 8. Forecast plots
# --------------------------------------------------
for asset in ASSETS:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        test.index,
        test[asset],
        label="Actual returns",
        linewidth=1,
    )
    ax.plot(
        forecast_df.index,
        forecast_df[asset],
        label="VAR forecast",
        linewidth=1,
    )

    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title(f"VAR({selected_lag}) Forecast vs Actual — {asset}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"var_forecast_{asset.lower()}.png",
        dpi=150,
    )
    plt.close(fig)


# --------------------------------------------------
# 9. Residual plots
# --------------------------------------------------
fig, axes = plt.subplots(
    len(ASSETS), 1, figsize=(12, 8), sharex=True
)

for ax, asset in zip(axes, ASSETS):
    ax.plot(residuals.index, residuals[asset], linewidth=0.8)
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title(f"VAR residuals — {asset}")
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "var_residuals.png", dpi=150)
plt.close(fig)

print(f"\nAll VAR outputs saved to: {OUTPUT_DIR}")