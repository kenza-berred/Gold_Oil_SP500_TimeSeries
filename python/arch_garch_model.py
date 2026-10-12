
from pathlib import Path

import pandas as pd
from statsmodels.stats.diagnostic import het_arch

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_returns.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "python" / "garch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training returns only
returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
returns = returns.apply(pd.to_numeric, errors="coerce")

print("Training returns shape:", returns.shape)
print("Date range:", returns.index.min(), "to", returns.index.max())
print("\nMissing values:\n", returns.isna().sum())

if returns.isna().any().any():
    raise ValueError("Missing values detected. Investigate before running ARCH-LM.")

if not returns.index.is_monotonic_increasing:
    raise ValueError("Dates are not in chronological order.")

# ARCH-LM test on original training returns
results = []

for asset in returns.columns:
    series = returns[asset].dropna()

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
        series,
        nlags=10
    )

    results.append({
        "Asset": asset,
        "Observations": len(series),
        "Lags": 10,
        "ARCH_LM_Statistic": lm_stat,
        "ARCH_LM_PValue": lm_pvalue,
        "ARCH_F_Statistic": f_stat,
        "ARCH_F_PValue": f_pvalue,
    })

    print(f"\n{asset}")
    print(f"ARCH-LM statistic: {lm_stat:.4f}")
    print(f"ARCH-LM p-value:   {lm_pvalue:.6g}")
    print(f"F-test p-value:    {f_pvalue:.6g}")

results_df = pd.DataFrame(results)
output_path = OUTPUT_DIR / "arch_lm_training_returns.csv"
results_df.to_csv(output_path, index=False)

print("\nSaved results to:", output_path)
