
# ============================================================
# SARIMA MODEL - OIL, GOLD, S&P 500
# Same chronological train/test split as Python
# ============================================================

rm(list = ls())

library(forecast)
library(FinTS)
library(tseries)

# 1. Project paths
project_dir <- "C:/Users/kenza/Desktop/Financial_Time_Series_Submission"

train_path <- file.path(
  project_dir, "data", "processed", "train_returns.csv"
)

test_path <- file.path(
  project_dir, "data", "processed", "test_returns.csv"
)

output_dir <- file.path(
  project_dir, "results", "R", "sarima"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# 2. Load data
train_df <- read.csv(train_path, check.names = FALSE)
test_df  <- read.csv(test_path, check.names = FALSE)

train_df[[1]] <- as.Date(train_df[[1]])
test_df[[1]]  <- as.Date(test_df[[1]])

assets <- c("Oil", "Gold", "SP500")

stopifnot(all(assets %in% names(train_df)))
stopifnot(all(assets %in% names(test_df)))

train_dates <- train_df[[1]]
test_dates  <- test_df[[1]]

stopifnot(!anyNA(train_dates), !anyNA(test_dates))
stopifnot(all(diff(train_dates) > 0))
stopifnot(all(diff(test_dates) > 0))
stopifnot(max(train_dates) < min(test_dates))

train_data <- as.data.frame(train_df[, assets])
test_data  <- as.data.frame(test_df[, assets])

stopifnot(!anyNA(train_data), !anyNA(test_data))

cat("============================================================\n")
cat("SARIMA - DATA VALIDATION\n")
cat("============================================================\n")

cat("Training observations:", nrow(train_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")

cat(
  "Training period:",
  as.character(min(train_dates)), "to",
  as.character(max(train_dates)), "\n"
)

cat(
  "Testing period:",
  as.character(min(test_dates)), "to",
  as.character(max(test_dates)), "\n"
)

# 3. Non-seasonal orders inherited from Python ARIMA
base_orders <- list(
  Oil   = c(2, 0, 2),
  Gold  = c(0, 0, 1),
  SP500 = c(0, 0, 0)
)

# Seasonal order format: P, D, Q, period
seasonal_candidates <- list(
  c(0, 0, 0, 5),
  c(1, 0, 0, 5),
  c(0, 0, 1, 5),
  c(1, 0, 1, 5)
)

candidate_results <- list()
diagnostics_results <- list()
metrics_results <- list()
forecasts_results <- list()
selected_models <- list()

candidate_counter <- 1
diagnostic_counter <- 1
metrics_counter <- 1
forecast_counter <- 1

# 4. Fit candidates and select by AIC
for (asset in assets) {

  cat("\n============================================================\n")
  cat("ASSET:", asset, "\n")
  cat("============================================================\n")

  y_train <- as.numeric(train_data[[asset]])
  y_test  <- as.numeric(test_data[[asset]])

  base_order <- base_orders[[asset]]
  fitted_candidates <- list()
  aic_values <- rep(Inf, length(seasonal_candidates))

  for (i in seq_along(seasonal_candidates)) {

    so <- seasonal_candidates[[i]]

    candidate_label <- paste0(
      "ARIMA(",
      paste(base_order, collapse = ","),
      ") x Seasonal(",
      paste(so, collapse = ","),
      ")"
    )

    cat("Fitting:", candidate_label, "\n")

    fit_i <- tryCatch(
      suppressWarnings(
        forecast::Arima(
          y_train,
          order = base_order,
          seasonal = list(
            order = so[1:3],
            period = so[4]
          ),
          include.mean = TRUE,
          method = "ML"
        )
      ),
      error = function(e) {
        cat("Fit error:", conditionMessage(e), "\n")
        NULL
      }
    )

    if (is.null(fit_i)) {

      candidate_results[[candidate_counter]] <- data.frame(
        Asset = asset,
        Order = paste(base_order, collapse = ","),
        Seasonal_Order = paste(so, collapse = ","),
        Seasonal_Period = so[4],
        AIC = NA_real_,
        BIC = NA_real_,
        Converged = FALSE,
        Status = "Fit error"
      )

      candidate_counter <- candidate_counter + 1
      next
    }

    # stats::arima generally stores optimizer status in $code.
    # If no code is exposed by the returned object, record that
    # convergence could not be independently verified.
    fit_code <- fit_i$code

    convergence_known <- !is.null(fit_code)
    converged <- convergence_known && isTRUE(as.integer(fit_code) == 0)

    candidate_aic <- as.numeric(AIC(fit_i))
    candidate_bic <- as.numeric(BIC(fit_i))

    status <- if (!convergence_known) {
      "Convergence status unavailable"
    } else if (converged) {
      "OK"
    } else {
      paste("Optimizer code", fit_code)
    }

    candidate_results[[candidate_counter]] <- data.frame(
      Asset = asset,
      Order = paste(base_order, collapse = ","),
      Seasonal_Order = paste(so, collapse = ","),
      Seasonal_Period = so[4],
      AIC = candidate_aic,
      BIC = candidate_bic,
      Converged = converged,
      Status = status
    )

    candidate_counter <- candidate_counter + 1

    # Only verified converged fits are eligible for selection.
    if (converged && is.finite(candidate_aic)) {
      fitted_candidates[[i]] <- fit_i
      aic_values[i] <- candidate_aic
    }
  }

  if (all(!is.finite(aic_values))) {
    stop(
      paste(
        "No candidate with verified convergence for", asset,
        "- inspect candidate comparison output."
      )
    )
  }

  best_index <- which.min(aic_values)
  best_fit <- fitted_candidates[[as.character(best_index)]]

  # Fallback lookup if list indexing by character is unavailable.
  if (is.null(best_fit)) {
    best_fit <- fitted_candidates[[best_index]]
  }

  best_seasonal <- seasonal_candidates[[best_index]]

  selected_models[[asset]] <- best_fit

  cat(
    "Selected seasonal order:",
    paste(best_seasonal, collapse = ","),
    "\n"
  )
  cat("Selected AIC:", AIC(best_fit), "\n")
  cat("Selected BIC:", BIC(best_fit), "\n")

  # 5. Residual diagnostics
  residuals_i <- as.numeric(residuals(best_fit))
  residuals_i <- residuals_i[is.finite(residuals_i)]

  fit_df <- sum(base_order[c(1, 3)]) +
    sum(best_seasonal[c(1, 3)])

  lb10 <- Box.test(
    residuals_i,
    lag = 10,
    type = "Ljung-Box",
    fitdf = fit_df
  )

  lb20 <- Box.test(
    residuals_i,
    lag = 20,
    type = "Ljung-Box",
    fitdf = fit_df
  )

  arch_test <- FinTS::ArchTest(
    residuals_i,
    lags = 10,
    demean = TRUE
  )

  jb_test <- tseries::jarque.bera.test(residuals_i)

  diagnostics_results[[diagnostic_counter]] <- data.frame(
    Asset = asset,
    Order = paste(base_order, collapse = ","),
    Seasonal_Order = paste(best_seasonal, collapse = ","),
    AIC = AIC(best_fit),
    BIC = BIC(best_fit),
    Ljung_Box_pvalue_lag10 = lb10$p.value,
    Ljung_Box_pvalue_lag20 = lb20$p.value,
    ARCH_LM_statistic = unname(arch_test$statistic),
    ARCH_LM_pvalue = arch_test$p.value,
    Jarque_Bera_statistic = unname(jb_test$statistic),
    Jarque_Bera_pvalue = jb_test$p.value
  )

  diagnostic_counter <- diagnostic_counter + 1

  # 6. Fixed-origin forecast over the complete test horizon
  forecast_i <- forecast::forecast(
    best_fit,
    h = length(y_test)
  )

  predicted <- as.numeric(forecast_i$mean)
  baseline <- rep(0, length(y_test))

  mae_model <- mean(abs(y_test - predicted))
  rmse_model <- sqrt(mean((y_test - predicted)^2))

  mae_base <- mean(abs(y_test - baseline))
  rmse_base <- sqrt(mean((y_test - baseline)^2))

  metrics_results[[metrics_counter]] <- data.frame(
    Asset = asset,
    Order = paste(base_order, collapse = ","),
    Seasonal_Order = paste(best_seasonal, collapse = ","),
    AIC = AIC(best_fit),
    BIC = BIC(best_fit),
    SARIMA_MAE = mae_model,
    SARIMA_RMSE = rmse_model,
    Zero_Baseline_MAE = mae_base,
    Zero_Baseline_RMSE = rmse_base,
    MAE_Improvement_pct =
      100 * (mae_base - mae_model) / mae_base,
    RMSE_Improvement_pct =
      100 * (rmse_base - rmse_model) / rmse_base
  )

  metrics_counter <- metrics_counter + 1

  forecasts_results[[forecast_counter]] <- data.frame(
    Date = test_dates,
    Asset = asset,
    Actual = y_test,
    Forecast = predicted,
    Baseline = baseline
  )

  forecast_counter <- forecast_counter + 1

  cat("Ljung-Box p-value (lag 10):", lb10$p.value, "\n")
  cat("Ljung-Box p-value (lag 20):", lb20$p.value, "\n")
  cat("ARCH-LM p-value:", arch_test$p.value, "\n")
  cat("Jarque-Bera p-value:", jb_test$p.value, "\n")
  cat("Test MAE:", mae_model, "\n")
  cat("Test RMSE:", rmse_model, "\n")
  cat("Baseline MAE:", mae_base, "\n")
  cat("Baseline RMSE:", rmse_base, "\n")
}

# 7. Save results
candidate_df <- do.call(rbind, candidate_results)
diagnostics_df <- do.call(rbind, diagnostics_results)
metrics_df <- do.call(rbind, metrics_results)
forecasts_df <- do.call(rbind, forecasts_results)

write.csv(
  candidate_df,
  file.path(output_dir, "sarima_candidate_comparison.csv"),
  row.names = FALSE
)

write.csv(
  diagnostics_df,
  file.path(output_dir, "sarima_diagnostics.csv"),
  row.names = FALSE
)

write.csv(
  metrics_df,
  file.path(output_dir, "sarima_metrics.csv"),
  row.names = FALSE
)

write.csv(
  forecasts_df,
  file.path(output_dir, "sarima_forecasts.csv"),
  row.names = FALSE
)

capture.output(
  lapply(selected_models, print),
  file = file.path(output_dir, "sarima_model_summaries.txt")
)

cat("\n============================================================\n")
cat("SARIMA CANDIDATE COMPARISON\n")
cat("============================================================\n")
print(candidate_df, row.names = FALSE)

cat("\n============================================================\n")
cat("SARIMA OUT-OF-SAMPLE METRICS\n")
cat("============================================================\n")
print(metrics_df, row.names = FALSE)

cat("\nResults saved to:\n", output_dir, "\n")

