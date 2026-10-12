
# ============================================================
# ARIMA MODEL â€” GOLD, OIL, S&P 500
# Same chronological train/test split as Python
# ============================================================

rm(list = ls())

library(forecast)
library(tseries)
library(FinTS)
library(lmtest)

# 1. Project paths
project_dir <- "C:/Users/kenza/Desktop/Financial_Time_Series_Submission"

train_path <- file.path(
  project_dir, "data", "processed", "train_returns.csv"
)

test_path <- file.path(
  project_dir, "data", "processed", "test_returns.csv"
)

output_dir <- file.path(project_dir, "results", "R", "arima")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Load data
train_df <- read.csv(train_path, check.names = FALSE)
test_df  <- read.csv(test_path, check.names = FALSE)

# Convert the first column to dates
train_df[[1]] <- as.Date(train_df[[1]])
test_df[[1]]  <- as.Date(test_df[[1]])

assets <- c("Oil", "Gold", "SP500")

stopifnot(all(assets %in% names(train_df)))
stopifnot(all(assets %in% names(test_df)))

# Validate chronological periods and dimensions
cat("Training rows:", nrow(train_df), "\n")
cat("Testing rows:", nrow(test_df), "\n")
cat("Training period:",
    as.character(min(train_df[[1]], na.rm = TRUE)), "to",
    as.character(max(train_df[[1]], na.rm = TRUE)), "\n")
cat("Testing period:",
    as.character(min(test_df[[1]], na.rm = TRUE)), "to",
    as.character(max(test_df[[1]], na.rm = TRUE)), "\n")

# 3. Candidate ARIMA models
candidates <- list(
  c(0, 0, 0),
  c(1, 0, 0),
  c(0, 0, 1),
  c(1, 0, 1),
  c(2, 0, 0),
  c(0, 0, 2),
  c(2, 0, 1),
  c(1, 0, 2),
  c(2, 0, 2)
)

metrics_list <- list()
model_list <- list()
diagnostics_list <- list()
forecast_list <- list()

# 4. Fit one ARIMA model per asset
for (asset in assets) {

  cat("\n====================================\n")
  cat("Asset:", asset, "\n")
  cat("====================================\n")

  train <- as.numeric(train_df[[asset]])
  test  <- as.numeric(test_df[[asset]])

  if (anyNA(train) || anyNA(test)) {
    stop(paste("Missing values found for", asset))
  }

  # Same model-selection approach as Python:
  # compare the predefined candidates using AIC.
  fitted_candidates <- list()
  aic_values <- rep(Inf, length(candidates))

  for (i in seq_along(candidates)) {

    order_i <- candidates[[i]]

    fit_i <- tryCatch(
      suppressWarnings(
        stats::arima(
          train,
          order = order_i,
          include.mean = TRUE,
          method = "ML"
        )
      ),
      error = function(e) NULL
    )

    if (!is.null(fit_i) &&
        is.finite(AIC(fit_i)) &&
        isTRUE(fit_i$code == 0)) {

      fitted_candidates[[i]] <- fit_i
      aic_values[i] <- AIC(fit_i)
    }
  }

  if (all(!is.finite(aic_values))) {
    stop(paste("No candidate model converged for", asset))
  }

  best_index <- which.min(aic_values)
  best_fit <- fitted_candidates[[best_index]]
  best_order <- candidates[[best_index]]

  model_list[[asset]] <- best_fit

  cat("Selected ARIMA(",
      paste(best_order, collapse = ","),
      ")\n", sep = "")
  cat("AIC:", AIC(best_fit), "\n")
  cat("BIC:", BIC(best_fit), "\n")

  # 5. Residual diagnostics
  residuals_i <- as.numeric(residuals(best_fit))
  residuals_i <- residuals_i[is.finite(residuals_i)]

  lb10 <- Box.test(
    residuals_i, lag = 10,
    type = "Ljung-Box",
    fitdf = sum(best_order[c(1, 3)])
  )

  lb20 <- Box.test(
    residuals_i, lag = 20,
    type = "Ljung-Box",
    fitdf = sum(best_order[c(1, 3)])
  )

  arch_test <- FinTS::ArchTest(
    residuals_i, lags = 10, demean = TRUE
  )

  jb_test <- tseries::jarque.bera.test(residuals_i)

  diagnostics_list[[asset]] <- data.frame(
    Asset = asset,
    Ljung_Box_10_p = lb10$p.value,
    Ljung_Box_20_p = lb20$p.value,
    ARCH_LM_p = arch_test$p.value,
    Jarque_Bera_p = jb_test$p.value
  )

  # 6. Fixed-origin forecast over the entire test horizon
  forecast_i <- forecast::forecast(
    best_fit,
    h = length(test)
  )

  predicted <- as.numeric(forecast_i$mean)

  # Zero-return baseline
  baseline <- rep(0, length(test))

  mae_model <- mean(abs(test - predicted))
  rmse_model <- sqrt(mean((test - predicted)^2))

  mae_base <- mean(abs(test - baseline))
  rmse_base <- sqrt(mean((test - baseline)^2))

  metrics_list[[asset]] <- data.frame(
    Asset = asset,
    Model = paste0(
      "ARIMA(",
      paste(best_order, collapse = ","),
      ")"
    ),
    AIC = AIC(best_fit),
    BIC = BIC(best_fit),
    MAE = mae_model,
    RMSE = rmse_model,
    Baseline_MAE = mae_base,
    Baseline_RMSE = rmse_base,
    MAE_Improvement_Pct =
      100 * (mae_base - mae_model) / mae_base,
    RMSE_Improvement_Pct =
      100 * (rmse_base - rmse_model) / rmse_base
  )

  forecast_list[[asset]] <- data.frame(
    Date = test_df[[1]],
    Asset = asset,
    Actual = test,
    Forecast = predicted,
    Baseline = baseline
  )

  cat("Ljung-Box p-value (lag 10):", lb10$p.value, "\n")
  cat("ARCH-LM p-value:", arch_test$p.value, "\n")
  cat("Test MAE:", mae_model, "\n")
  cat("Test RMSE:", rmse_model, "\n")
  cat("Baseline MAE:", mae_base, "\n")
  cat("Baseline RMSE:", rmse_base, "\n")
}

# 7. Save outputs
metrics_df <- do.call(rbind, metrics_list)
diagnostics_df <- do.call(rbind, diagnostics_list)
forecasts_df <- do.call(rbind, forecast_list)

write.csv(
  metrics_df,
  file.path(output_dir, "arima_metrics.csv"),
  row.names = FALSE
)

write.csv(
  diagnostics_df,
  file.path(output_dir, "arima_diagnostics.csv"),
  row.names = FALSE
)

write.csv(
  forecasts_df,
  file.path(output_dir, "arima_forecasts.csv"),
  row.names = FALSE
)

# Save selected model summaries
sink(file.path(output_dir, "arima_model_summaries.txt"))

for (asset in assets) {
  cat("\n\n==========", asset, "==========\n")
  print(model_list[[asset]])
}

sink()

cat("\nARIMA analysis completed.\n")
cat("Results saved to:", output_dir, "\n")
