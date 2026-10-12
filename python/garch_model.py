
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_returns.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "python" / "garch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
returns = returns.apply(pd.to_numeric, errors="coerce")

if returns.isna().any().any():
    raise ValueError("Missing values found in training returns.")

if not returns.index.is_monotonic_increasing:
    raise ValueError("Training dates are not chronological.")

if len(returns) != 1128:
    raise ValueError(f"Expected 1128 training returns, got {len(returns)}.")

parameter_rows = []
diagnostic_rows = []

for asset in returns.columns:
    # Scaling improves numerical stability for small daily returns.
    series = returns[asset] * 100

    model = arch_model(
        series,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    result = model.fit(disp="off", show_warning=False)

    print("\n" + "=" * 65)
    print(f"{asset} — GARCH(1,1), Student-t errors")
    print("=" * 65)
    print(result.summary())

    params = result.params
    parameter_rows.append({
        "Asset": asset,
        "Model": "GARCH(1,1)",
        "Distribution": "Student-t",
        "Converged": result.convergence_flag == 0,
        "LogLikelihood": result.loglikelihood,
        "AIC": result.aic,
        "BIC": result.bic,
        "mu": params.get("mu", np.nan),
        "omega": params.get("omega", np.nan),
        "alpha1": params.get("alpha[1]", np.nan),
        "beta1": params.get("beta[1]", np.nan),
        "nu": params.get("nu", np.nan),
        "alpha_plus_beta": (
            params.get("alpha[1]", np.nan)
            + params.get("beta[1]", np.nan)
        ),
    })

    standardized_resid = pd.Series(
        result.std_resid,
        index=series.index
    ).dropna()

    squared_resid = standardized_resid ** 2

    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    lb_returns = acorr_ljungbox(
        standardized_resid,
        lags=[10],
        return_df=True
    )

    lb_squared = acorr_ljungbox(
        squared_resid,
        lags=[10],
        return_df=True
    )

    arch_lm, arch_p, arch_f, arch_f_p = het_arch(
        standardized_resid,
        nlags=10
    )

    diagnostic_rows.append({
        "Asset": asset,
        "Converged": result.convergence_flag == 0,
        "LjungBox_StdResid_Lag10_Stat": lb_returns["lb_stat"].iloc[0],
        "LjungBox_StdResid_Lag10_PValue": lb_returns["lb_pvalue"].iloc[0],
        "LjungBox_SquaredResid_Lag10_Stat": lb_squared["lb_stat"].iloc[0],
        "LjungBox_SquaredResid_Lag10_PValue": lb_squared["lb_pvalue"].iloc[0],
        "ARCH_LM_StdResid_Statistic": arch_lm,
        "ARCH_LM_StdResid_PValue": arch_p,
    })

parameters_df = pd.DataFrame(parameter_rows)
diagnostics_df = pd.DataFrame(diagnostic_rows)

parameters_path = OUTPUT_DIR / "garch11_student_t_parameters.csv"
diagnostics_path = OUTPUT_DIR / "garch11_student_t_diagnostics.csv"

parameters_df.to_csv(parameters_path, index=False)
diagnostics_df.to_csv(diagnostics_path, index=False)

print("\nSaved parameters:", parameters_path)
print("Saved diagnostics:", diagnostics_path)
print("\nParameter summary:")
print(parameters_df.to_string(index=False))
print("\nDiagnostic summary:")
print(diagnostics_df.to_string(index=False))
