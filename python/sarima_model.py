# ============================================================
# SARIMA MODEL — GOLD, OIL, S&P 500
# Bloc A: Fondamentaux
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


# ------------------------------------------------------------
# 1. Project paths and configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "results" / "python" / "sarima"

TRAIN_PATH = DATA_DIR / "train_returns.csv"
TEST_PATH = DATA_DIR / "test_returns.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["Oil", "Gold", "SP500"]

# Non-seasonal ARIMA orders selected previously using training data.
BASE_ORDERS = {
    "Gold": (0, 0, 1),
    "Oil": (2, 0, 2),
    "SP500": (0, 0, 0),
}

# Seasonal period: five trading days.
# D=0 because we model returns, not non-stationary price levels.
SEASONAL_CANDIDATES = [
    (0, 0, 0, 5),
    (1, 0, 0, 5),
    (0, 0, 1, 5),
    (1, 0, 1, 5),
]


# ------------------------------------------------------------
# 2. Load and validate datasets
# ------------------------------------------------------------

def load_returns(path: Path) -> pd.DataFrame:
    """Load return data with the date column as the index."""

    df = pd.read_csv(path)

    date_candidates = [
        col for col in df.columns
        if str(col).strip().lower()
        in {"date", "datetime", "timestamp", "unnamed: 0"}
    ]

    if date_candidates:
        date_col = date_candidates[0]
    else:
        # The first column may contain the saved datetime index.
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df = df.set_index(date_col)
    df.index.name = "Date"

    missing_assets = [asset for asset in ASSETS if asset not in df.columns]

    if missing_assets:
        raise ValueError(
            f"{path.name} is missing asset columns: {missing_assets}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[ASSETS].apply(pd.to_numeric, errors="raise")
    df = df.sort_index()

    if not df.index.is_unique:
        raise ValueError(f"Duplicate dates found in {path.name}.")

    if df.isna().any().any():
        raise ValueError(f"Missing values found in {path.name}.")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Non-finite values found in {path.name}.")

    return df


train_returns = load_returns(TRAIN_PATH)
test_returns = load_returns(TEST_PATH)

if train_returns.index.max() >= test_returns.index.min():
    raise ValueError(
        "Training and test periods overlap or are incorrectly ordered."
    )

print("=" * 65)
print("SARIMA — DATA VALIDATION")
print("=" * 65)

print(f"Training observations: {len(train_returns)}")
print(
    f"Training period: {train_returns.index.min().date()} "
    f"to {train_returns.index.max().date()}"
)

print(f"Testing observations: {len(test_returns)}")
print(
    f"Testing period: {test_returns.index.min().date()} "
    f"to {test_returns.index.max().date()}"
)

print(f"Assets: {', '.join(ASSETS)}")


# ------------------------------------------------------------
# 3. Fit SARIMA candidates using training data only
# ------------------------------------------------------------

candidate_results = []
selected_models = {}

for asset in ASSETS:

    print("\n" + "=" * 65)
    print(f"ASSET: {asset}")
    print("=" * 65)

    # Use a regular integer index for modeling because
    # trading dates do not have a fixed calendar frequency.
    y_train = train_returns[asset].astype(float).reset_index(drop=True)

    base_order = BASE_ORDERS[asset]
    successful_fits = []

    for seasonal_order in SEASONAL_CANDIDATES:

        model_label = (
            f"ARIMA{base_order} x "
            f"Seasonal{seasonal_order}"
        )

        print(f"Fitting: {model_label}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = SARIMAX(
                    y_train,
                    order=base_order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )

                 
            fit_method = (
                "powell"
                if asset == "SP500"
                and seasonal_order == (0, 0, 0, 5)
                else "lbfgs"
            )

            max_iterations = 1000 if fit_method == "powell" else 200

            fitted = model.fit(
                method=fit_method,
                maxiter=max_iterations,
                disp=False,
            )

  

            retvals = getattr(fitted, "mle_retvals", {}) or {}
            converged = bool(retvals.get("converged", True))

            aic = float(fitted.aic)
            bic = float(fitted.bic)

            candidate_results.append({
                "Asset": asset,
                "Order": str(base_order),
                "Seasonal_Order": str(seasonal_order),
                "Seasonal_Period": seasonal_order[3],
                "AIC": aic,
                "BIC": bic,
                "Converged": converged,
                "Status": "OK" if converged else "Did not converge",
            })

            if converged and np.isfinite(aic):
                successful_fits.append({
                    "seasonal_order": seasonal_order,
                    "fitted": fitted,
                    "aic": aic,
                    "bic": bic,
                })

        except Exception as exc:
            candidate_results.append({
                "Asset": asset,
                "Order": str(base_order),
                "Seasonal_Order": str(seasonal_order),
                "Seasonal_Period": seasonal_order[3],
                "AIC": np.nan,
                "BIC": np.nan,
                "Converged": False,
                "Status": f"Failed: {str(exc)[:180]}",
            })

            print(f"  Candidate failed: {exc}")

    if not successful_fits:
        raise RuntimeError(
            f"No SARIMA candidate converged for {asset}. "
            "Inspect the candidate results and model warnings."
        )

    # Select the lowest-AIC converged model.
    best = min(successful_fits, key=lambda item: item["aic"])

    selected_models[asset] = best

    print(f"\nSelected seasonal order: {best['seasonal_order']}")
    print(f"Selected AIC: {best['aic']:.4f}")
    print(f"Selected BIC: {best['bic']:.4f}")


# Save all candidate comparisons.
candidate_df = pd.DataFrame(candidate_results)

candidate_df.to_csv(
    OUTPUT_DIR / "sarima_candidate_comparison.csv",
    index=False,
)


# ------------------------------------------------------------
# 4. Residual diagnostics and out-of-sample forecasts
# ------------------------------------------------------------

diagnostic_results = []
forecast_results = []
metric_results = []

for asset in ASSETS:

    best = selected_models[asset]
    fitted = best["fitted"]
    seasonal_order = best["seasonal_order"]
    base_order = BASE_ORDERS[asset]

    print("\n" + "=" * 65)
    print(f"SELECTED MODEL DIAGNOSTICS: {asset}")
    print("=" * 65)

    # Residual diagnostics use training data only.
    residuals = np.asarray(fitted.resid, dtype=float)

    burn = int(getattr(fitted.model, "loglikelihood_burn", 0) or 0)
    residuals = residuals[burn:]
    residuals = residuals[np.isfinite(residuals)]

    # Degrees of freedom used for the Ljung-Box adjustment.
    model_df = (
        base_order[0]
        + base_order[2]
        + seasonal_order[0]
        + seasonal_order[2]
    )

    lb = acorr_ljungbox(
        residuals,
        lags=[10, 20],
        model_df=model_df,
        return_df=True,
    )

    arch_lm, arch_pvalue, arch_f, arch_f_pvalue = het_arch(
        residuals,
        nlags=10,
    )

    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(residuals)

    diagnostic_row = {
        "Asset": asset,
        "Order": str(base_order),
        "Seasonal_Order": str(seasonal_order),
        "AIC": float(fitted.aic),
        "BIC": float(fitted.bic),
        "Ljung_Box_pvalue_lag10": float(lb.loc[10, "lb_pvalue"]),
        "Ljung_Box_pvalue_lag20": float(lb.loc[20, "lb_pvalue"]),
        "ARCH_LM_statistic": float(arch_lm),
        "ARCH_LM_pvalue": float(arch_pvalue),
        "Jarque_Bera_statistic": float(jb_stat),
        "Jarque_Bera_pvalue": float(jb_pvalue),
        "Residual_skewness": float(skewness),
        "Residual_kurtosis": float(kurtosis),
    }

    diagnostic_results.append(diagnostic_row)

    print(f"Ljung-Box p-value (lag 10): {lb.loc[10, 'lb_pvalue']:.6g}")
    print(f"Ljung-Box p-value (lag 20): {lb.loc[20, 'lb_pvalue']:.6g}")
    print(f"ARCH-LM p-value: {arch_pvalue:.6g}")
    print(f"Jarque-Bera p-value: {jb_pvalue:.6g}")

    # Forecast the full test horizon from the end of training.
    # Do not refit on the test set.
    forecasts = np.asarray(
        fitted.get_forecast(steps=len(test_returns)).predicted_mean,
        dtype=float,
    )

    actual = test_returns[asset].to_numpy(dtype=float)
    zero_baseline = np.zeros(len(actual))

    sarima_mae = mean_absolute_error(actual, forecasts)
    sarima_rmse = np.sqrt(mean_squared_error(actual, forecasts))

    baseline_mae = mean_absolute_error(actual, zero_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(actual, zero_baseline))

    mae_improvement = (
        100 * (baseline_mae - sarima_mae) / baseline_mae
        if baseline_mae != 0 else np.nan
    )

    rmse_improvement = (
        100 * (baseline_rmse - sarima_rmse) / baseline_rmse
        if baseline_rmse != 0 else np.nan
    )

    metric_results.append({
        "Asset": asset,
        "Order": str(base_order),
        "Seasonal_Order": str(seasonal_order),
        "SARIMA_MAE": sarima_mae,
        "SARIMA_RMSE": sarima_rmse,
        "Zero_Baseline_MAE": baseline_mae,
        "Zero_Baseline_RMSE": baseline_rmse,
        "MAE_Improvement_pct": mae_improvement,
        "RMSE_Improvement_pct": rmse_improvement,
    })

    for date, observed, prediction in zip(
        test_returns.index, actual, forecasts
    ):
        forecast_results.append({
            "Date": date,
            "Asset": asset,
            "Actual_Return": observed,
            "SARIMA_Forecast": prediction,
            "Zero_Baseline": 0.0,
        })


# ------------------------------------------------------------
# 5. Save final outputs
# ------------------------------------------------------------

diagnostics_df = pd.DataFrame(diagnostic_results)
metrics_df = pd.DataFrame(metric_results)
forecasts_df = pd.DataFrame(forecast_results)

diagnostics_df.to_csv(
    OUTPUT_DIR / "sarima_diagnostics.csv",
    index=False,
)

metrics_df.to_csv(
    OUTPUT_DIR / "sarima_metrics.csv",
    index=False,
)

forecasts_df.to_csv(
    OUTPUT_DIR / "sarima_forecasts.csv",
    index=False,
)

print("\n" + "=" * 65)
print("SARIMA CANDIDATE COMPARISON")
print("=" * 65)
print(candidate_df.to_string(index=False))

print("\n" + "=" * 65)
print("SARIMA OUT-OF-SAMPLE METRICS")
print("=" * 65)
print(metrics_df.round(6).to_string(index=False))

print("\n" + "=" * 65)
print("SARIMA RESIDUAL DIAGNOSTICS")
print("=" * 65)
print(diagnostics_df.round(6).to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)