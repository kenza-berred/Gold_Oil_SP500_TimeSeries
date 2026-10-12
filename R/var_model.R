
# ============================================================
# VAR MODEL â€” OIL, GOLD, S&P 500
# Same chronological split used in Python
# ============================================================

rm(list = ls())

library(vars)
library(tseries)

# 1. Paths
project_dir <- "C:/Users/kenza/Desktop/Financial_Time_Series_Submission"

train_path <- file.path(
  project_dir, "data", "processed", "train_returns.csv"
)
test_path <- file.path(
  project_dir, "data", "processed", "test_returns.csv"
)

output_dir <- file.path(project_dir, "results", "R", "var")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Load and validate data
train_df <- read.csv(train_path, check.names = FALSE)
test_df  <- read.csv(test_path, check.names = FALSE)

train_df[[1]] <- as.Date(train_df[[1]])
test_df[[1]] <- as.Date(test_df[[1]])

assets <- c("Oil", "Gold", "SP500")

stopifnot(all(assets %in% names(train_df)))
stopifnot(all(assets %in% names(test_df)))

train <- as.data.frame(train_df[, assets])
test  <- as.data.frame(test_df[, assets])

stopifnot(!anyNA(train), !anyNA(test))

cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")
cat("Training period:",
    as.character(min(train_df[[1]])), "to",
    as.character(max(train_df[[1]])), "\n")
cat("Testing period:",
    as.character(min(test_df[[1]])), "to",
    as.character(max(test_df[[1]])), "\n")

# 3. Lag-order selection
# Use the same maximum lag as the Python analysis.
lag_selection <- VARselect(
  train,
  lag.max = 10,
  type = "const"
)

cat("\nLag selection criteria:\n")
print(lag_selection$selection)

capture.output(
  print(lag_selection),
  file = file.path(output_dir, "var_lag_selection.txt")
)

selection_df <- data.frame(
  Criterion = names(lag_selection$selection),
  Selected_Lag = as.integer(lag_selection$selection)
)

write.csv(
  selection_df,
  file.path(output_dir, "var_lag_selection.csv"),
  row.names = FALSE
)

# 4. Estimate VAR
# Use BIC as the primary criterion.
# VARselect returns lag 1..lag.max; a result of 0 is possible.
# If BIC selects 0, do not silently force lag 1.
selected_lag <- as.integer(lag_selection$selection["SC(n)"])

if (is.na(selected_lag) || selected_lag < 1) {
  cat(
    "\nBIC selects lag 0. A standard VAR(0) has no lagged ",
    "cross-variable terms. We fit VAR(1) as a clearly labelled ",
    "exploratory dynamic model for diagnostics and comparison.\n",
    sep = ""
  )
  selected_lag <- 1
  lag_decision <- "Exploratory VAR(1); BIC selected lag 0"
} else {
  lag_decision <- paste("VAR selected by BIC:", selected_lag)
}

cat("\nModel lag:", selected_lag, "\n")
cat("Lag decision:", lag_decision, "\n")

var_fit <- VAR(
  train,
  p = selected_lag,
  type = "const"
)

capture.output(
  summary(var_fit),
  file = file.path(output_dir, "var_model_summary.txt")
)

cat("\nVAR model summary:\n")
print(summary(var_fit))

# 5. Residual serial-correlation test
serial_test <- serial.test(
  var_fit,
  lags.pt = 12,
  type = "PT.asymptotic"
)

cat("\nPortmanteau serial-correlation test:\n")
print(serial_test)

# Save serial-correlation test result to a reproducible CSV.
# The vars serial.test object stores the Chi-squared statistic and
# p-value inside the $serial element.
serial_df <- data.frame(
  Test        = "Portmanteau (PT.asymptotic)",
  Lags        = 12L,
  Statistic   = unname(serial_test$serial$statistic),
  DF          = unname(serial_test$serial$parameter),
  P_value     = serial_test$serial$p.value,
  H0          = "No residual serial correlation",
  Significant_at_5pct = serial_test$serial$p.value < 0.05
)

write.csv(
  serial_df,
  file.path(output_dir, "var_serial_test.csv"),
  row.names = FALSE
)

cat("\nSerial-correlation result saved to var_serial_test.csv\n")

# 6. Residual normality test
normality_test <- normality.test(
  var_fit,
  multivariate.only = TRUE
)

cat("\nResidual normality test:\n")
print(normality_test)

# Save multivariate normality test result.
normality_df <- data.frame(
  Test        = "Multivariate normality (JB)",
  Statistic   = unname(normality_test$jb.mul$JB$statistic),
  DF          = unname(normality_test$jb.mul$JB$parameter),
  P_value     = normality_test$jb.mul$JB$p.value,
  H0          = "Multivariate normality of residuals",
  Significant_at_5pct = normality_test$jb.mul$JB$p.value < 0.05
)

write.csv(
  normality_df,
  file.path(output_dir, "var_normality_test.csv"),
  row.names = FALSE
)

cat("\nNormality result saved to var_normality_test.csv\n")

# 7. Granger/block exogeneity tests
granger_results <- list()
counter <- 1

for (causing in assets) {
  for (caused in assets) {

    if (causing != caused) {

      test_result <- causality(
        var_fit,
        cause = causing
      )

      # The test above is joint block exogeneity for all
      # equations, not a pair-specific test.
      # Store its result once per cause.
      if (!(causing %in% names(granger_results))) {
        granger_results[[causing]] <- data.frame(
          Cause = causing,
          Test = "Block exogeneity / Granger",
          Statistic = unname(test_result$Granger$statistic),
          P_value = test_result$Granger$p.value
        )
      }
    }
  }
}

granger_df <- do.call(rbind, granger_results)

write.csv(
  granger_df,
  file.path(output_dir, "var_granger_block_tests.csv"),
  row.names = FALSE
)

cat("\nGranger block-exogeneity tests:\n")
print(granger_df)

# Pairwise directional tests using restricted equation models
pairwise_results <- list()
idx <- 1

for (caused in assets) {
  for (causing in setdiff(assets, caused)) {

    # H0: lagged values of 'causing' do not help predict 'caused'
    formula_full <- as.formula(
      paste(caused, "~", paste(assets, collapse = " + "))
    )

    # Construct restricted model excluding the causing variable
    other_vars <- setdiff(assets, causing)
    formula_restricted <- as.formula(
      paste(caused, "~", paste(other_vars, collapse = " + "))
    )

    # Use VAR equation's own lag structure by building
    # the lagged regressors explicitly.
    y <- train[[caused]]
    X_full <- data.frame(y = y)

    for (v in assets) {
      for (lag_i in seq_len(selected_lag)) {
        X_full[[paste0(v, "_L", lag_i)]] <-
          dplyr::lag(train[[v]], lag_i)
      }
    }

    X_full$const <- 1
    X_full <- na.omit(X_full)

    rhs_full <- setdiff(names(X_full), "y")
    fit_full <- lm(
      as.formula(paste("y ~", paste(rhs_full, collapse = " + "), "- 1")),
      data = X_full
    )

    rhs_restricted <- rhs_full[
      !grepl(paste0("^", causing, "_L"), rhs_full)
    ]

    fit_restricted <- lm(
      as.formula(
        paste("y ~", paste(rhs_restricted, collapse = " + "), "- 1")
      ),
      data = X_full
    )

    test_compare <- anova(fit_restricted, fit_full)

    pairwise_results[[idx]] <- data.frame(
      Cause = causing,
      Effect = caused,
      F_statistic = test_compare$F[2],
      P_value = test_compare$`Pr(>F)`[2]
    )

    idx <- idx + 1
  }
}

pairwise_df <- do.call(rbind, pairwise_results)

write.csv(
  pairwise_df,
  file.path(output_dir, "var_pairwise_granger.csv"),
  row.names = FALSE
)

cat("\nPairwise Granger tests:\n")
print(pairwise_df)

# 8. Forecast the fixed test horizon
var_forecast <- predict(
  var_fit,
  n.ahead = nrow(test),
  ci = 0.95
)

forecast_matrix <- sapply(
  assets,
  function(asset) as.numeric(var_forecast$fcst[[asset]][, "fcst"])
)

colnames(forecast_matrix) <- assets

# 9. Evaluate against zero-return baseline
metric_results <- list()
forecast_results <- list()

for (asset in assets) {

  actual <- test[[asset]]
  predicted <- forecast_matrix[, asset]
  baseline <- rep(0, length(actual))

  mae_model <- mean(abs(actual - predicted))
  rmse_model <- sqrt(mean((actual - predicted)^2))

  mae_baseline <- mean(abs(actual - baseline))
  rmse_baseline <- sqrt(mean((actual - baseline)^2))

  metric_results[[asset]] <- data.frame(
    Asset = asset,
    Model = paste0("VAR(", selected_lag, ")"),
    MAE = mae_model,
    RMSE = rmse_model,
    Baseline_MAE = mae_baseline,
    Baseline_RMSE = rmse_baseline,
    MAE_Improvement_Pct =
      100 * (mae_baseline - mae_model) / mae_baseline,
    RMSE_Improvement_Pct =
      100 * (rmse_baseline - rmse_model) / rmse_baseline
  )

  forecast_results[[asset]] <- data.frame(
    Date = test_df[[1]],
    Asset = asset,
    Actual = actual,
    Forecast = predicted,
    Baseline = baseline
  )
}

metrics_df <- do.call(rbind, metric_results)
forecasts_df <- do.call(rbind, forecast_results)

write.csv(
  metrics_df,
  file.path(output_dir, "var_metrics.csv"),
  row.names = FALSE
)

write.csv(
  forecasts_df,
  file.path(output_dir, "var_forecasts.csv"),
  row.names = FALSE
)

# 10. Save residuals
residual_df <- as.data.frame(resid(var_fit))
write.csv(
  residual_df,
  file.path(output_dir, "var_residuals.csv"),
  row.names = FALSE
)

# 11. Save decision and diagnostics summary
decision_df <- data.frame(
  Criterion = "BIC / SC(n)",
  Selected_Lag = as.integer(lag_selection$selection["SC(n)"]),
  Estimated_Lag = selected_lag,
  Decision = lag_decision
)

write.csv(
  decision_df,
  file.path(output_dir, "var_model_decision.csv"),
  row.names = FALSE
)

cat("\nForecast metrics:\n")
print(metrics_df)

cat("\nVAR analysis completed.\n")
cat("Results saved to:", output_dir, "\n")
