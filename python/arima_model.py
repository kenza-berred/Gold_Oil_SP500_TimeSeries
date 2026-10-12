
from pathlib import Path
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "python" / "arima"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_returns.csv"
TEST_FILE = DATA_DIR / "test_returns.csv"

CANDIDATE_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 0),
    (0, 0, 2),
    (2, 0, 1),
    (1, 0, 2),
    (2, 0, 2),
]

DIAGNOSTIC_LAGS = [10, 20]
ARCH_LAGS = 10
ACF_LAGS = 30


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

train_returns = pd.read_csv(
    TRAIN_FILE,
    index_col=0,
    parse_dates=True,
)

test_returns = pd.read_csv(
    TEST_FILE,
    index_col=0,
    parse_dates=True,
)

train_returns = train_returns.apply(pd.to_numeric, errors="raise")
test_returns = test_returns.apply(pd.to_numeric, errors="raise")

train_returns = train_returns.sort_index()
test_returns = test_returns.sort_index()

if train_returns.empty or test_returns.empty:
    raise ValueError("Training or testing dataset is empty.")

if train_returns.isna().any().any():
    raise ValueError("Training returns contain missing values.")

if test_returns.isna().any().any():
    raise ValueError("Testing returns contain missing values.")

if not np.isfinite(train_returns.to_numpy()).all():
    raise ValueError("Training returns contain infinite values.")

if not np.isfinite(test_returns.to_numpy()).all():
    raise ValueError("Testing returns contain infinite values.")

if not train_returns.index.is_unique:
    raise ValueError("Duplicate dates detected in training data.")

if not test_returns.index.is_unique:
    raise ValueError("Duplicate dates detected in testing data.")

if not train_returns.index.max() < test_returns.index.min():
    raise ValueError("Training and testing periods overlap or are out of order.")

if list(train_returns.columns) != list(test_returns.columns):
    raise ValueError("Training and testing columns do not match.")

print("=" * 70)
print("ARIMA MODELING — FINANCIAL TIME SERIES")
print("=" * 70)

print("\nTraining shape:", train_returns.shape)
print("Testing shape:", test_returns.shape)
print("Assets:", train_returns.columns.tolist())

print(
    "Training period:",
    train_returns.index.min(),
    "to",
    train_returns.index.max(),
)

print(
    "Testing period:",
    test_returns.index.min(),
    "to",
    test_returns.index.max(),
)


# ==================================================
# 3. FIT CANDIDATE ARIMA MODELS
# ==================================================

selection_results = []
fitted_models = {}

print("\n" + "=" * 70)
print("STEP 1 — ARIMA MODEL SELECTION")
print("=" * 70)

for asset in train_returns.columns:

    # Use positional indexing to avoid unsupported date-frequency issues.
    # Original dates remain available in train_returns and test_returns.
    values = train_returns[asset].to_numpy(dtype=float)

    series = pd.Series(
        values,
        index=pd.RangeIndex(start=0, stop=len(values)),
        name=asset,
    )

    print(f"\nEvaluating candidates for {asset}...")

    for order in CANDIDATE_ORDERS:

        try:
            model = ARIMA(
                series,
                order=order,
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                fitted = model.fit()

            converged = bool(
                fitted.mle_retvals.get("converged", True)
            )

            key = (asset, order)
            fitted_models[key] = fitted

            selection_results.append({
                "Asset": asset,
                "Order": str(order),
                "AIC": float(fitted.aic),
                "BIC": float(fitted.bic),
                "Converged": converged,
            })

            print(
                f"  ARIMA{order}: "
                f"AIC={fitted.aic:.3f}, "
                f"BIC={fitted.bic:.3f}, "
                f"Converged={converged}"
            )

        except Exception as exc:
            print(
                f"  ARIMA{order} failed for {asset}: {exc}"
            )


# ==================================================
# 4. COMPARE MODELS AND SELECT CANDIDATES
# ==================================================

selection_df = pd.DataFrame(selection_results)

if selection_df.empty:
    raise RuntimeError("No ARIMA candidate models fitted successfully.")

selection_df = selection_df.sort_values(
    ["Asset", "AIC"]
).reset_index(drop=True)

selection_df.to_csv(
    RESULTS_DIR / "arima_model_selection.csv",
    index=False,
)

print("\nFull model selection results:")
print(selection_df.round(4).to_string(index=False))

# Do not select a model whose optimizer did not converge.
converged_models = selection_df[
    selection_df["Converged"]
].copy()

assets_without_models = (
    set(train_returns.columns)
    - set(converged_models["Asset"])
)

if assets_without_models:
    raise RuntimeError(
        "No converged ARIMA model for: "
        + ", ".join(sorted(assets_without_models))
    )

best_candidates = (
    converged_models.loc[
        converged_models.groupby("Asset")["AIC"].idxmin()
    ]
    .sort_values("Asset")
    .reset_index(drop=True)
)

print("\nSelected candidates by lowest AIC among converged models:")
print(best_candidates.round(4).to_string(index=False))

best_candidates.to_csv(
    RESULTS_DIR / "arima_selected_models.csv",
    index=False,
)


# ==================================================
# 5. RESIDUAL AUTOCORRELATION DIAGNOSTICS
# ==================================================

print("\n" + "=" * 70)
print("STEP 2 — LJUNG-BOX RESIDUAL DIAGNOSTICS")
print("=" * 70)

ljung_box_rows = []

for _, row in best_candidates.iterrows():

    asset = row["Asset"]
    order = ast.literal_eval(row["Order"])

    fitted = fitted_models[(asset, order)]

    residuals = np.asarray(fitted.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) <= max(DIAGNOSTIC_LAGS):
        raise ValueError(
            f"Not enough residual observations for {asset}."
        )

    lb = acorr_ljungbox(
        residuals,
        lags=DIAGNOSTIC_LAGS,
        return_df=True,
    )

    for lag in DIAGNOSTIC_LAGS:

        p_value = float(lb.loc[lag, "lb_pvalue"])

        ljung_box_rows.append({
            "Asset": asset,
            "Order": str(order),
            "Lag": lag,
            "Q Statistic": float(lb.loc[lag, "lb_stat"]),
            "P-value": p_value,
            "Conclusion": (
                "Residual autocorrelation detected"
                if p_value < 0.05
                else "No significant residual autocorrelation detected"
            ),
        })

    # Residual ACF plot
    fig, ax = plt.subplots(figsize=(9, 4))

    plot_acf(
        residuals,
        lags=ACF_LAGS,
        ax=ax,
    )

    ax.set_title(f"{asset} — ARIMA{order} Residual ACF")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    fig.tight_layout()

    filename = asset.lower().replace(" ", "_")

    fig.savefig(
        RESULTS_DIR / f"{filename}_residual_acf.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


ljung_box_df = pd.DataFrame(ljung_box_rows)

ljung_box_df.to_csv(
    RESULTS_DIR / "arima_ljung_box_diagnostics.csv",
    index=False,
)

print(ljung_box_df.round(5).to_string(index=False))


# ==================================================
# 6. ARCH-LM AND JARQUE-BERA DIAGNOSTICS
# ==================================================

print("\n" + "=" * 70)
print("STEP 3 — ARCH-LM AND JARQUE-BERA TESTS")
print("=" * 70)

residual_diagnostic_rows = []

for _, row in best_candidates.iterrows():

    asset = row["Asset"]
    order = ast.literal_eval(row["Order"])

    fitted = fitted_models[(asset, order)]

    residuals = np.asarray(fitted.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    # ARCH-LM: H0 = no ARCH effects
    arch_lm, arch_p, arch_f, arch_f_p = het_arch(
        residuals,
        nlags=ARCH_LAGS,
    )

    # Jarque-Bera: H0 = normally distributed residuals
    jb_stat, jb_p, skewness, kurtosis = jarque_bera(
        residuals
    )

    residual_diagnostic_rows.append({
        "Asset": asset,
        "ARIMA Order": str(order),
        "ARCH-LM Statistic": float(arch_lm),
        "ARCH-LM P-value": float(arch_p),
        "ARCH-LM F Statistic": float(arch_f),
        "ARCH-LM F P-value": float(arch_f_p),
        "JB Statistic": float(jb_stat),
        "JB P-value": float(jb_p),
        "Residual Skewness": float(skewness),
        "Residual Kurtosis": float(kurtosis),
    })


residual_diagnostics_df = pd.DataFrame(
    residual_diagnostic_rows
)

residual_diagnostics_df.to_csv(
    RESULTS_DIR / "arima_residual_diagnostics.csv",
    index=False,
)

print(residual_diagnostics_df.round(5).to_string(index=False))


# ==================================================
# 7. OUT-OF-SAMPLE FORECASTING AND EVALUATION
# ==================================================

print("\n" + "=" * 70)
print("STEP 4 — TEST-SET FORECASTING AND EVALUATION")
print("=" * 70)

forecast_rows = []
metric_rows = []

for _, row in best_candidates.iterrows():

    asset = row["Asset"]
    order = ast.literal_eval(row["Order"])

    # This model was fitted on training observations only.
    fitted = fitted_models[(asset, order)]

    actual = test_returns[asset].copy()
    n_test = len(actual)

    # Forecast all test observations from the end of training.
    # The forecast is positional; no regular date frequency is needed.
    forecast_result = fitted.get_forecast(
        steps=n_test
    )

    predicted_values = np.asarray(
        forecast_result.predicted_mean,
        dtype=float,
    ).reshape(-1)

    confidence_intervals = np.asarray(
        forecast_result.conf_int(),
        dtype=float,
    )

    if len(predicted_values) != n_test:
        raise RuntimeError(
            f"Forecast length mismatch for {asset}."
        )

    # Attach the original test dates to the forecast values.
    predicted = pd.Series(
        predicted_values,
        index=actual.index,
        name="ARIMA Forecast",
    )

    lower = pd.Series(
        confidence_intervals[:, 0],
        index=actual.index,
        name="Lower 95% CI",
    )

    upper = pd.Series(
        confidence_intervals[:, 1],
        index=actual.index,
        name="Upper 95% CI",
    )

    # Baseline: predict zero log return for every test observation.
    baseline = pd.Series(
        np.zeros(n_test),
        index=actual.index,
        name="Zero-return Baseline",
    )

    # MAE and RMSE for ARIMA and the baseline.
    arima_mae = float(
        np.mean(np.abs(actual.to_numpy() - predicted.to_numpy()))
    )

    arima_rmse = float(
        np.sqrt(
            np.mean(
                (actual.to_numpy() - predicted.to_numpy()) ** 2
            )
        )
    )

    baseline_mae = float(
        np.mean(np.abs(actual.to_numpy() - baseline.to_numpy()))
    )

    baseline_rmse = float(
        np.sqrt(
            np.mean(
                (actual.to_numpy() - baseline.to_numpy()) ** 2
            )
        )
    )

    mae_improvement = (
        100.0 * (baseline_mae - arima_mae) / baseline_mae
        if baseline_mae > 0
        else np.nan
    )

    rmse_improvement = (
        100.0 * (baseline_rmse - arima_rmse) / baseline_rmse
        if baseline_rmse > 0
        else np.nan
    )

    metric_rows.append({
        "Asset": asset,
        "ARIMA Order": str(order),
        "ARIMA MAE": arima_mae,
        "ARIMA RMSE": arima_rmse,
        "Baseline MAE": baseline_mae,
        "Baseline RMSE": baseline_rmse,
        "MAE Improvement (%)": mae_improvement,
        "RMSE Improvement (%)": rmse_improvement,
        "Test Observations": n_test,
    })

    # Save actual returns, forecasts and prediction intervals.
    for date in actual.index:

        forecast_rows.append({
            "Date": date,
            "Asset": asset,
            "Actual Return": float(actual.loc[date]),
            "ARIMA Forecast": float(predicted.loc[date]),
            "Lower 95% CI": float(lower.loc[date]),
            "Upper 95% CI": float(upper.loc[date]),
            "Zero-return Baseline": 0.0,
        })

    # Plot actual returns against the multi-step ARIMA forecast.
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        actual.index,
        actual.to_numpy(),
        label="Actual returns",
        linewidth=1,
    )

    ax.plot(
        predicted.index,
        predicted.to_numpy(),
        label=f"ARIMA{order} forecast",
        linewidth=1.5,
    )

    ax.fill_between(
        actual.index,
        lower.to_numpy(),
        upper.to_numpy(),
        alpha=0.2,
        label="95% forecast interval",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_title(f"{asset} — Actual vs ARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend()

    fig.tight_layout()

    filename = asset.lower().replace(" ", "_")

    fig.savefig(
        RESULTS_DIR / f"{filename}_arima_forecast.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{asset}: ARIMA{order} | "
        f"MAE={arima_mae:.6f} | "
        f"RMSE={arima_rmse:.6f} | "
        f"Baseline MAE={baseline_mae:.6f} | "
        f"Baseline RMSE={baseline_rmse:.6f}"
    )


# ==================================================
# 8. SAVE FORECASTS AND EVALUATION METRICS
# ==================================================

forecasts_df = pd.DataFrame(forecast_rows)

forecasts_df.to_csv(
    RESULTS_DIR / "arima_forecasts_test.csv",
    index=False,
)

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    RESULTS_DIR / "arima_forecast_metrics.csv",
    index=False,
)

print("\n" + "=" * 70)
print("FINAL OUT-OF-SAMPLE EVALUATION")
print("=" * 70)

print(metrics_df.round(6).to_string(index=False))

print("\nAll results saved to:")
print(RESULTS_DIR)

print("\nGenerated files include:")
print("- arima_model_selection.csv")
print("- arima_selected_models.csv")
print("- arima_ljung_box_diagnostics.csv")
print("- arima_residual_diagnostics.csv")
print("- arima_forecasts_test.csv")
print("- arima_forecast_metrics.csv")
print("- Residual ACF plots for each asset")
print("- Forecast plots for each asset")

print("\nARIMA pipeline completed.")