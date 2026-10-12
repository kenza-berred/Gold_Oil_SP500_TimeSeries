# Financial Time-Series Econometrics & Generative AI

Final course submission — **Data Science & AI Engineering (TEK-UP)**, course on financial time series and generative AI.

The project downloads five years of daily prices for three assets, cleans and splits them under a strict chronological 90/10 protocol, fits classical, volatility, multivariate and deep-learning models **in both Python and R**, evaluates every return forecast against a common zero-return baseline, tests pairwise superiority with Diebold–Mariano, and runs an empirical study of a gated LLM interpreter (Bloc C).

**Deliverable read first:** [`results/RAPPORT_FINAL_SERIES_TEMPORELLES.pdf`](results/RAPPORT_FINAL_SERIES_TEMPORELLES.pdf)

---

## 1. Research scope

| Dimension | Value | Verified in |
|---|---|---|
| Assets | Gold (`GC=F`), Crude Oil WTI (`CL=F`), S&P 500 (`^GSPC`) | `src/download_data.py` L13–17 |
| Period | 2021-10-01 → 2026-10-01 (5 years, 1 day) | `data/raw/daily_close_prices.csv` |
| Raw observations | **1 258** rows | `data/raw/daily_close_prices.csv` |
| Cleaned synchronized panel | **1 255** prices | `data/processed/prices_clean.csv` |
| Log returns | **1 254** (first difference, one observation consumed) | `data/processed/log_returns.csv` |
| Train prices / returns | **1 129** / **1 128** — 2021-10-01 → 2026-04-01 | `data/processed/train_*.csv` |
| Test prices / returns | **126** / **126** — 2026-04-02 → 2026-10-01 | `data/processed/test_*.csv` |
| Modelling target | Daily **log returns** (`I(0)`); prices are `I(1)` | `src/prepare_data.py`, `notebooks/01_EDA.ipynb` |
| Return baseline | Zero-return forecast | every evaluation script |
| Volatility baseline | Training unconditional variance | `python/garch_forecast_evaluation.py` |
| LLM under test | `openai/gpt-oss-120b` via Groq Cloud | `results/llm/experiments/experiments_manifest.json` |

**Note on counting.** 1 255 is the *price* count and 1 254 the *return* count. Any statement in the repository that calls the panel "1 254 daily observations" is mixing the two; see §8.

---

## 2. Modelling suite

| Family | Specification | Language | Artifacts |
|---|---|---|---|
| ARIMA | 9 candidate orders, `trend="c"`, AIC among *converged* only | Python **and** R | `results/python/arima/`, `results/R/arima/` |
| SARIMA | 4 seasonal candidates, weekly `s = 5` | Python **and** R | `results/python/sarima/`, `results/R/sarima/` |
| VAR | Lag selection (`maxlags=10`, BIC primary), Granger causality | Python **and** R | `results/python/var/`, `results/R/var/` |
| Cointegration / VECM | ADF + KPSS, Johansen trace & max-eigenvalue, rank sensitivity grid | Python **and** R | code only — see §8 |
| ARCH / GARCH | ARCH-LM, GARCH(1,1) Student-t vs Normal, QLIKE evaluation | Python (`arch`) | `results/python/garch/` |
| Deep learning | LSTM, GRU — 1-step and 5-day direct multi-step, walk-forward | Python (TensorFlow) | `results/python/deep_learning*/` |
| Statistical tests | Ljung–Box, Jarque–Bera, Diebold–Mariano + Holm | Python | `results/python/diebold_mariano/` |
| Generative AI | Grounded prompting, claim-by-claim hallucination audit, human-vs-LLM rubric | Python + Groq | `results/llm/` |

Not implemented, despite appearing in some narrative text: MLP networks, asymmetric GARCH variants (EGARCH/GJR), parametric VaR. Treat them as absent.

---

## 3. Project layout

```
.
├── src/                    # Stage 1 — download → clean → split (asserted chronological)
├── data/
│   ├── raw/                # daily_close_prices.csv (1 258 rows, frozen snapshot)
│   └── processed/          # clean panel, returns, normalised prices, rolling vol, train/test
├── notebooks/
│   └── 01_EDA.ipynb        # Stage 2 — executed EDA (16 code cells, outputs stored)
├── python/                 # Stage 3 — one script per model family
│   └── llm/                # Stage 5 — Bloc C: context builder + 3 Groq experiments
├── R/                      # Stage 4 — ARIMA, SARIMA, VAR, VECM mirrors
├── results/
│   ├── python/<family>/    # per-model CSVs, PNGs, JSON summaries
│   ├── R/<family>/         # same for the R suite (arima, sarima, var)
│   ├── llm/                # prompts, responses, context JSON, audits
│   ├── figures/            # 23 report-ready figures
│   ├── comparative_evaluation.csv   # consolidated return-forecast table (18 rows)
│   ├── BLOC_A_CONCLUSIONS.md
│   └── RAPPORT_FINAL_SERIES_TEMPORELLES.pdf
├── requirements.txt        # Python dependencies
└── README.md               # this file
```

There is **no orchestrator**. `src/` writes `data/processed/*.csv`; each `python/` and `R/` script reads those CSVs independently and writes to `results/<language>/<family>/`. Cross-stage coordination is purely filesystem-driven, and every script resolves its own root from `__file__`.

---

## 4. Environment

**Python 3.12.** Install from `requirements.txt`:

```powershell
python -m venv dl_env
.\dl_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`statsmodels`, `arch` and `yfinance` were installed ad hoc when the econometric stages ran, so their exact build is not recorded — `requirements.txt` gives a floor for each and flags the gap. Record the installed version once you reinstall, otherwise the ARIMA/SARIMA/VAR/VECM/GARCH stages are not byte-reproducible.

**R.** Required for the four scripts under `R/`:

```r
install.packages(c("forecast", "tseries", "FinTS", "lmtest", "vars", "urca"))
```

Actual usage per script: `arima_model.R` → forecast, tseries, FinTS, lmtest · `sarima_model.R` → forecast, FinTS, tseries · `var_model.R` → vars, tseries · `vecm_model.R` → urca, vars.

**LLM key (optional).** Read from the environment, never hardcoded:

```powershell
$env:GROQ_API_KEY = "your-key-here"
```

Without it `python/llm/run_experiments.py` writes placeholder responses with status `PENDING_API_KEY` instead of failing. The experiments shipped in `results/llm/` were executed with a real key on 2026-10-12 (see §6).

---

## 5. How to reproduce

### Stage 1 — data pipeline

```powershell
python src\download_data.py    # rewrites data/raw/ — SKIP to keep the shipped snapshot frozen
python src\prepare_data.py     # -> data/processed/prices_clean.csv, log_returns.csv, ...
python src\split_data.py       # -> train/test CSVs + hard integrity asserts
```

`src/download_data.py` also writes `data/raw/daily_adjusted_close_prices.csv`, which is **not** part of this submission and is not consumed downstream. Prefer re-running Stages 2–5 on the shipped CSVs over re-downloading: Yahoo Finance revisions would silently change every reported number.

### Stage 2 — exploratory analysis

Open `notebooks/01_EDA.ipynb` and run all cells. The notebook is **fully executed as delivered** (code cells carry execution counts 1 → 16 with stored outputs, including the ADF/KPSS table). It renders figures inline; it does **not** write them to disk — see §8.

### Stage 3 — Python modelling

Each script is standalone:

```powershell
python python\arima_model.py
python python\sarima_model.py
python python\var_model.py
python python\vecm_model.py
python python\arch_garch_model.py
python python\garch_model.py
python python\garch_forecast_evaluation.py
python python\garch_distribution_comparison.py
python python\deep_learning_models.py
python python\deep_learning_multistep.py
python python\diebold_mariano_test.py
```

### Stage 4 — R modelling (cross-language validation)

```powershell
Rscript R\arima_model.R
Rscript R\sarima_model.R
Rscript R\var_model.R
Rscript R\vecm_model.R
```

**Read this before running:** all four scripts open with an absolute path outside this folder —
`project_dir <- "C:/Users/kenza/Desktop/Financial_Time_Series_Restart"` (`R/arima_model.R` L15, `R/sarima_model.R` L14, `R/var_model.R` L13, `R/vecm_model.R` L11). As shipped they read the *other* project's data and overwrite *its* results. Edit that single line to the submission root before running Stage 4.

### Stage 5 — LLM experiments (Bloc C)

```powershell
python python\llm\context_builder.py    # -> results/llm/financial_project_context.json
python python\llm\run_experiments.py    # 3 experiments, or --experiment <1|2|3>
```

`context_builder.py` aggregates 14 result files. Eleven resolve inside this folder; `python/vecm/vecm_decision.csv` and `python/vecm/stationarity_tests.csv` do not (their outputs were not shipped), so a fresh run marks those two `"status": "missing"`. The shipped `results/llm/financial_project_context.json` predates this and contains all fourteen as `loaded`.

### Stage 6 — report assembly

**Not included.** The Markdown → Word → PDF pipeline lives outside this submission. `results/RAPPORT_FINAL_SERIES_TEMPORELLES.pdf` is the final rendered deliverable.

---

## 6. Results index

| Finding | Where to check it |
|---|---|
| Selected ARIMA orders — Gold `(0,0,1)`, Oil `(2,0,2)`, SP500 `(0,0,0)` (white noise with drift) | `results/python/arima/arima_selected_models.csv`, `results/R/arima/arima_metrics.csv` |
| Ljung–Box / residual diagnostics, ARCH-LM, Jarque–Bera on ARIMA residuals | `results/python/arima/arima_ljung_box_diagnostics.csv`, `arima_residual_diagnostics.csv` |
| SARIMA candidates and AIC ranking, `s = 5` | `results/python/sarima/sarima_candidate_comparison.csv`, `results/R/sarima/sarima_candidate_comparison.csv` |
| VAR lag decision, residual whiteness (p = 0.0280), non-normality, Granger matrix | `results/python/var/var_lag_selection.csv`, `var_diagnostics.csv`, `var_granger_causality.csv`; R mirrors in `results/R/var/` |
| ARCH effects — LM statistics 120.40 (Oil), 124.04 (Gold), 148.86 (SP500) | `results/python/garch/arch_lm_training_returns.csv` |
| GARCH(1,1)-t parameters, α+β = 0.9688 / 0.9831 / 0.9864, ν = 8.71 / 5.08 / 6.79 | `results/python/garch/garch11_student_t_parameters.csv` |
| Out-of-sample volatility: MSE / MAE / QLIKE vs constant-variance benchmark | `results/python/garch/garch11_student_t_forecast_metrics.csv`, `garch_distribution_comparison_metrics.csv` |
| Walk-forward folds and final LSTM/GRU metrics | `results/python/deep_learning/walk_forward_validation.csv`, `results/python/deep_learning_multistep/final_test_metrics.csv` |
| Consolidated return-forecast comparison, 18 configurations | `results/comparative_evaluation.csv` |
| 30 Diebold–Mariano pairwise tests with Holm-adjusted p-values | `results/python/diebold_mariano/dm_test_results_holm.csv` |
| Prompts, raw responses, execution manifest | `results/llm/experiments/` |
| Claim-by-claim hallucination audit (16 assertions) | `results/llm/hallucination_audit.md` |
| Human vs LLM 6-dimension rubric | `results/llm/human_vs_llm_comparison.md` |
| Bloc C narrative and validation | `results/llm/CHAPITRE_BLOC_C.md`, `FINAL_BLOC_C_VALIDATION.md`, `BLOC_C_REVIEW.md` |

---

## 7. Headline findings

1. **Prices are non-stationary, returns are stationary.** ADF fails to reject a unit root on levels (p = 0.339 / 0.999 / 0.935) while KPSS rejects stationarity (2.776 / 4.325 / 4.766 against a 0.463 critical value); on log returns ADF rejects decisively (p = 0.000). This is what justifies `d = 0` on the return scale — `notebooks/01_EDA.ipynb` cell 27, `results/figures/eda_09_acf_pacf.png`.
2. **No return model meaningfully beats zero.** Across all 18 ARIMA/SARIMA/VAR configurations in `results/comparative_evaluation.csv`, MAE and RMSE gains over the zero-return baseline stay within **−0.51 % to +0.76 %** — several configurations are marginally *worse* than the baseline. This is consistent with weak-form market efficiency, not a modelling failure.
3. **Weekly seasonality does not survive.** The no-seasonal variant is best or near-best for Gold and SP500; for Oil the seasonal term buys ΔAIC ≈ 0.07 and Python (`(1,0,0)₅`) and R (`(0,0,1)₅`) disagree on which seasonal structure to use — documented in the report rather than smoothed over.
4. **Volatility is predictable even when returns are not.** ARCH effects are overwhelming (LM > 120, p < 10⁻²⁰), persistence is high (α+β ≈ 0.97–0.99) with fat tails (ν ≈ 5–9), and GARCH(1,1)-t beats the constant-variance benchmark on **QLIKE for all three assets**. Note the asymmetry: on the *squared*-error scale GARCH is worse than the benchmark for Gold (MSE 1.058e-7 vs 1.000e-8) — QLIKE, not MSE, is the loss that detects the gain (Patton 2011).
5. **Cointegration evidence is inconclusive, so no VECM was estimated.** Trace and maximum-eigenvalue statistics disagree; the Python gate `should_estimate_vecm = ALLOW_VECM_ESTIMATION and tests_agree and valid_cointegration_rank` therefore refused the estimation, and the report explains why level-scale VECM forecasts cannot enter a return-scale ranking.
6. **No forecast is statistically superior.** Of 30 Diebold–Mariano comparisons (N = 25 aligned blocks), one reaches nominal p < 0.05 (ARIMA vs GRU on Oil, p = 0.0483); after Holm correction **every adjusted p-value is 1.000 and nothing is significant**.
7. **The LLM interpreted real results faithfully, with caveats.** All three experiments completed against the live API. Of 16 audited assertions: 12 SUPPORTED, 3 OVERSIMPLIFIED, 0 UNSUPPORTED, and 1 CONTRADICTED caused by a prompt-truncation artefact (`comp[:12]`), not by an invented number. Under the 6-dimension rubric in `results/llm/human_vs_llm_comparison.md` the human reference scores **28.5 / 30 (95.0 %)** and the LLM **28.0 / 30 (93.3 %)** — a 0.5-point gap concentrated on formal statistical rigour, while the LLM scores strictly higher on executive clarity (5/5 vs 4/5).

---

## 8. Methodological rules enforced by code

| Rule | Enforcement |
|---|---|
| No random splits, no look-ahead | `src/split_data.py`: `math.floor(len(prices)*0.90)` then `assert train.index.max() < test.index.min()` and `assert len(train)+len(test) == len(prices)`; returns are differenced **before** splitting so the boundary return is preserved |
| Data integrity before fitting | `src/prepare_data.py`: duplicated-date removal, `dropna()`, strict positivity check; downstream scripts re-assert monotonicity, uniqueness, NaN and finiteness |
| No leakage in deep learning | `shuffle=False` in every `model.fit`; `StandardScaler` refitted per fold on that fold's training slice only; early stopping uses an inner split of the training fold, never the validation targets; multi-step history is built from already-realised returns only |
| No test-set refitting | `python/arima_model.py` / `sarima_model.py` project `get_forecast(steps=n_test)` from the train-fitted model ("Do not refit on the test set"); GARCH variance recursion carries fixed train parameters through the test window |
| Non-converged models are discarded, not ranked | `Converged` column recorded in `results/python/arima/arima_model_selection.csv` and `results/python/sarima/sarima_candidate_comparison.csv`; selection filters on it |
| Degenerate lag orders handled explicitly | `python/var_model.py`: when BIC returns lag 0, `lag_used_for_estimation` is floored at 1 and the raw criterion is kept in a separate column |
| Multiple-testing honesty | Diebold–Mariano with Harvey–Leybourne–Newbold small-sample correction, then Holm (1979) across all 30 comparisons |
| MAPE excluded on purpose | Financial returns cross zero; `python/deep_learning_models.py` states "MAPE is intentionally omitted" and the report §B.3.2 justifies the exclusion (Hyndman & Koehler) |
| Interpretation guardrails fed to the LLM | `python/llm/context_builder.py` L46–57 injects 10 hard constraints (e.g. Granger ≠ physical causality, do not rank GARCH against return forecasters, Johansen tests disagreed) |
| Secrets only via environment | `run_experiments.py` reads `os.environ["GROQ_API_KEY"]`, honours `Retry-After` on HTTP 429, retries once, degrades to `PENDING_API_KEY` |
| Experiment state is append-only | `results/llm/experiments/experiments_manifest.json` merges new runs instead of replacing them |

---

## 9. Known issues and disclosure

Listed so a reviewer does not have to discover them.

**Package completeness**

1. **No VECM / Johansen artifacts are shipped.** `results/python/vecm/` and `results/R/vecm/` are absent, while `python/vecm_model.py`, `R/vecm_model.R` and finding §7.5 all depend on them. The cointegration ranks quoted in the report cannot currently be re-checked inside this folder. (The estimate files from an earlier forced-rank run were deliberately excluded because they contradict the final "VECM not estimated" decision — the *decision* and *stationarity* outputs are what is missing.)
2. **`results/python/var/` is stale relative to `python/var_model.py`.** The VAR artifacts are dated 2026-10-09 15:22, while `var_model.py` was last edited 2026-10-11 14:46 and now writes `raw_criterion_lag` / `lag_used_for_estimation` columns that the shipped `var_lag_selection.csv` does not contain. Re-run Stage 3 to regenerate.
3. **`requirements.txt` cannot fully pin `statsmodels`, `arch` and `yfinance`** — see §4.

**Presentation**

4. **The report body is 56 pages**, against the requested 15–25 excluding appendices and code, and its central comparative table (p. 29–30) renders with values split across lines.
5. **`notebooks/01_EDA.ipynb` contains no `savefig` call.** The ten `eda_*.png` files in `results/figures/` were exported by a manual step that is not versioned anywhere, so re-running the notebook will not regenerate them.
6. **SARIMA produces no figure at all** and ARIMA has no residual time-series plot — only residual ACF panels exist.

**Internal consistency**

7. **Human-vs-LLM totals disagree across artifacts.** In `results/llm/human_vs_llm_comparison.md` the six scored dimensions sum to **28.5 / 30 (95.0 %)** for the human reference and **28.0 / 30 (93.3 %)** for the LLM (arithmetic verified: 5 + 5 + 4 + 4.5 + 5 + 5 and 4 + 4.5 + 5 + 4.5 + 5 + 5); narrative documents outside this folder state **27.0 / 30 (90 %)**. The scored table is authoritative.
8. **Audited-assertion count disagrees.** `results/llm/hallucination_audit.md` audits **16** assertions (12 SUPPORTED / 3 OVERSIMPLIFIED / 0 UNSUPPORTED / 1 CONTRADICTED); the report and other summary documents claim "22 claims audited, 0 inventions". Both cannot be true.
9. **Observation counts are quoted inconsistently** — 1 254 vs 1 255 for the price panel (§1).
10. **`results/llm/*.md` link to absolute paths under `c:/Users/kenza/Desktop/Financial_Time_Series_Restart/...`**, which resolve on no other machine. Relative links were intended.
11. **Cointegration ranks cannot be re-checked in this folder.** `results/llm/reference_interpretation.md` states trace `r = 3` and maximum-eigenvalue `r = 1` (with the R `urca` trace statistic 34.80 marginally failing to reject `r = 0` against a 34.91 critical value), while an earlier summary document circulated outside this submission quotes "trace r = 1, max-eigenvalue r = 0". Neither reading is verifiable here until item 1 is fixed.

---

## 10. Academic context

Data Science & AI Engineering programme (TEK-UP), final deliverable for a course on financial time-series econometrics and generative AI. The negative results are the point: nothing beats a zero-return forecast, no cointegration rank is defensible enough to estimate a VECM, and no pairwise forecast comparison survives multiple-testing correction. Each is reported as obtained, with the diagnostic evidence attached.
