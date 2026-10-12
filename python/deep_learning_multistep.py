
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================================
# Configuration
# =========================================================
SEED = 42
WINDOW_SIZE = 20
HORIZON = 5
N_SPLITS = 3
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 6

ASSETS = ["Oil", "Gold", "SP500"]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "python" / "deep_learning_multistep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_returns(filename):
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {filename}")
    if df[ASSETS].isna().any().any():
        raise ValueError(f"Missing values in {filename}")

    return df


def make_windows(values, window, horizon):
    X, y = [], []

    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window:i])
        y.append(values[i:i + horizon])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def build_model(model_type):
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, len(ASSETS)))

    if model_type == "LSTM":
        x = tf.keras.layers.LSTM(32)(inputs)
    elif model_type == "GRU":
        x = tf.keras.layers.GRU(32)(inputs)
    else:
        raise ValueError("Unknown model type")

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(HORIZON * len(ASSETS))(x)
    outputs = tf.keras.layers.Reshape(
        (HORIZON, len(ASSETS))
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model


def score(actual, predicted):
    return {
        "MSE": float(mean_squared_error(actual.ravel(), predicted.ravel())),
        "RMSE": float(np.sqrt(mean_squared_error(
            actual.ravel(), predicted.ravel()
        ))),
        "MAE": float(mean_absolute_error(actual.ravel(), predicted.ravel())),
    }


def fit_with_early_stopping(model, X, y):
    if len(X) < 10:
        raise ValueError("Not enough training windows")

    split = int(len(X) * 0.85)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X[:split],
        y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[callback],
        verbose=0
    )
    return model


def walk_forward_validation(train_raw, model_type):
    # Expanding-window folds. Each fold's scaler is fitted only
    # on that fold's training segment.
    fold_size = len(train_raw) // (N_SPLITS + 1)
    rows = []

    for fold in range(N_SPLITS):
        train_end = fold_size * (fold + 1)
        val_end = (
            fold_size * (fold + 2)
            if fold < N_SPLITS - 1
            else len(train_raw)
        )

        fold_train = train_raw[:train_end]
        fold_val = train_raw[train_end:val_end]

        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(fold_train)
        scaled_val = scaler.transform(fold_val)

        # Include preceding training history as context for validation.
        combined = np.vstack([scaled_train, scaled_val])
        X_all, y_all = make_windows(combined, WINDOW_SIZE, HORIZON)

        # Retain only windows whose forecast target begins in validation.
        first_val_target = len(scaled_train)
        starts = np.arange(len(X_all)) + WINDOW_SIZE
        keep = starts >= first_val_target
        X_val, y_val = X_all[keep], y_all[keep]

        X_fit, y_fit = make_windows(
            scaled_train, WINDOW_SIZE, HORIZON
        )

        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED + fold)

        model = build_model(model_type)
        fit_with_early_stopping(model, X_fit, y_fit)

        pred_scaled = model.predict(X_val, verbose=0)

        pred = scaler.inverse_transform(
            pred_scaled.reshape(-1, len(ASSETS))
        ).reshape(-1, HORIZON, len(ASSETS))

        actual = scaler.inverse_transform(
            y_val.reshape(-1, len(ASSETS))
        ).reshape(-1, HORIZON, len(ASSETS))

        result = score(actual, pred)

        rows.append({
            "Model": model_type,
            "Fold": fold + 1,
            "Train_End_Index": train_end,
            "Validation_End_Index": val_end,
            **result
        })

        print(f"Validation: {model_type}, fold {fold + 1}/{N_SPLITS}")

    return rows


def main():
    train_df = load_returns("train_returns.csv")
    test_df = load_returns("test_returns.csv")

    if train_df["Date"].max() >= test_df["Date"].min():
        raise ValueError("Training and test periods overlap")

    train_raw = train_df[ASSETS].to_numpy(dtype=np.float64)
    test_raw = test_df[ASSETS].to_numpy(dtype=np.float64)

    if len(test_raw) < HORIZON:
        raise ValueError("Test set is shorter than forecast horizon")

    # -----------------------------------------------------
    # 1. Walk-forward validation for both model families
    # -----------------------------------------------------
    validation_rows = []

    for model_type in ["LSTM", "GRU"]:
        validation_rows.extend(
            walk_forward_validation(train_raw, model_type)
        )

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        OUTPUT_DIR / "walk_forward_validation.csv",
        index=False
    )

    print("\nWalk-forward validation results:")
    print(validation_df.to_string(index=False))

    # -----------------------------------------------------
    # 2. Final evaluation on non-overlapping 5-day blocks.
    # Models are fitted on training data only.
    # -----------------------------------------------------
    final_scaler = StandardScaler()
    scaled_train = final_scaler.fit_transform(train_raw)
    X_train, y_train = make_windows(
        scaled_train, WINDOW_SIZE, HORIZON
    )

    # Start each forecast block using only information available
    # before that block. No test observations are used for fitting.
    test_blocks = []
    block_starts = list(range(0, len(test_raw) - HORIZON + 1, HORIZON))

    for model_type in ["LSTM", "GRU"]:
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)

        model = build_model(model_type)
        fit_with_early_stopping(model, X_train, y_train)

        rows = []

        for start in block_starts:
            # History may include training returns and test returns
            # observed before the current block, never future values.
            history = np.vstack([train_raw, test_raw[:start]])
            history_scaled = final_scaler.transform(history)
            last_window = history_scaled[-WINDOW_SIZE:]

            pred_scaled = model.predict(
                last_window[np.newaxis, :, :],
                verbose=0
            )[0]

            pred = final_scaler.inverse_transform(pred_scaled)
            actual = test_raw[start:start + HORIZON]

            for step in range(HORIZON):
                for asset_idx, asset in enumerate(ASSETS):
                    rows.append({
                        "Model": model_type,
                        "Block_Start": str(
                            test_df["Date"].iloc[start].date()
                        ),
                        "Date": str(
                            test_df["Date"].iloc[start + step].date()
                        ),
                        "Horizon_Step": step + 1,
                        "Asset": asset,
                        "Actual": float(actual[step, asset_idx]),
                        "Predicted": float(pred[step, asset_idx]),
                        "Baseline": 0.0
                    })

        forecast_df = pd.DataFrame(rows)
        forecast_df.to_csv(
            OUTPUT_DIR / f"{model_type.lower()}_test_forecasts.csv",
            index=False
        )

        # Metrics across all 5-step blocks and all assets.
        for asset in ASSETS:
            part = forecast_df[forecast_df["Asset"] == asset]
            rows_for_asset = {
                "Model": model_type,
                "Asset": asset,
                **score(
                    part["Actual"].to_numpy(),
                    part["Predicted"].to_numpy()
                ),
                "N_Forecasts": len(part)
            }
            baseline = score(
                part["Actual"].to_numpy(),
                part["Baseline"].to_numpy()
            )
            rows_for_asset.update({
                "Baseline_MSE": baseline["MSE"],
                "Baseline_RMSE": baseline["RMSE"],
                "Baseline_MAE": baseline["MAE"]
            })

            test_blocks.append(rows_for_asset)

        print(f"Final test evaluation completed for {model_type}")

    metrics_df = pd.DataFrame(test_blocks)
    metrics_df.to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False
    )

    summary = {
        "forecast_horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "validation_folds": N_SPLITS,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "test_blocks": len(block_starts),
        "assets": ASSETS,
        "evaluation": (
            "Direct five-step forecasts in non-overlapping blocks. "
            "Metrics are calculated over all forecast steps."
        )
    }

    with open(OUTPUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal test metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
