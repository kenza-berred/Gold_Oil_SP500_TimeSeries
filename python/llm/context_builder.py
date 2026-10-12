import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "llm" / "financial_project_context.json"

FILES = {
    "comparative_evaluation": "comparative_evaluation.csv",
    "arima_metrics": "python/arima/arima_forecast_metrics.csv",
    "arima_residual_diagnostics": "python/arima/arima_residual_diagnostics.csv",
    "arima_ljung_box": "python/arima/arima_ljung_box_diagnostics.csv",
    "sarima_metrics": "python/sarima/sarima_metrics.csv",
    "var_metrics": "python/var/var_forecast_metrics.csv",
    "var_diagnostics": "python/var/var_diagnostics.csv",
    "granger_causality": "python/var/var_granger_causality.csv",
    "arch_lm": "python/garch/arch_lm_training_returns.csv",
    "garch_forecast_metrics": "python/garch/garch11_student_t_forecast_metrics.csv",
    "deep_learning_metrics": "python/deep_learning_multistep/final_test_metrics.csv",
    "dm_holm_tests": "python/diebold_mariano/dm_test_results_holm.csv",
    "vecm_decision": "python/vecm/vecm_decision.csv",
    "stationarity_tests": "python/vecm/stationarity_tests.csv",
}

def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

context = {
    "project": {
        "title": "Multivariate Financial Time-Series Modelling and Forecasting",
        "assets": ["Gold", "Oil", "SP500"],
        "target": "Daily log-returns for mean forecasts; conditional variance for GARCH",
        "train_observations": 1128,
        "test_observations": 126,
        "baseline": "Zero-return forecast for return prediction",
    },
    "bloc_a_conclusions": None,
    "results": {},
    "interpretation_constraints": [
        "Use only evidence supported by the supplied results.",
        "Small percentage improvements must not be described as major gains.",
        "Distinguish return forecasts from volatility forecasts.",
        "Do not claim Granger predictability proves real-world causality.",
        "VAR residual serial correlation was reported with p=0.028.",
        "Python and R selected different seasonal orders for Oil.",
        "Johansen trace and maximum-eigenvalue tests disagreed; no definitive cointegration rank was established.",
        "Diebold-Mariano Holm-adjusted results were reported as non-significant, but their sample alignment and N=25 require verification.",
        "GARCH and ARIMA/LSTM/GRU metrics target different quantities and must not be ranked directly against each other.",
        "Identify uncertainty and limitations instead of inventing explanations.",
    ],
}

conclusions_file = RESULTS / "BLOC_A_CONCLUSIONS.md"
if conclusions_file.exists():
    context["bloc_a_conclusions"] = conclusions_file.read_text(
        encoding="utf-8", errors="replace"
    )

for name, relative_path in FILES.items():
    path = RESULTS / relative_path
    if not path.exists():
        context["results"][name] = {
            "status": "missing",
            "path": relative_path,
        }
        continue

    try:
        df = pd.read_csv(path)
        context["results"][name] = {
            "status": "loaded",
            "path": relative_path,
            "columns": list(df.columns),
            "row_count": len(df),
            "data": [
                {key: clean_value(value) for key, value in row.items()}
                for row in df.to_dict(orient="records")
            ],
        }
    except Exception as exc:
        context["results"][name] = {
            "status": "read_error",
            "path": relative_path,
            "error": str(exc),
        }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(context, indent=2, ensure_ascii=False, allow_nan=False),
    encoding="utf-8",
)

print(f"Context file created: {OUTPUT}")
print(f"Result files processed: {len(FILES)}")
print(f"Loaded successfully: {sum(v['status'] == 'loaded' for v in context['results'].values())}")
print(f"Missing files: {sum(v['status'] == 'missing' for v in context['results'].values())}")
