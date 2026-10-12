
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42
WINDOW_SIZE = 20
N_SPLITS = 3
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 6

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "python" / "deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["Oil", "Gold", "SP500"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_returns(filename):
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates found in {filename}")

    if df[ASSETS].isna().any().any():
        raise ValueError(f"Missing return values found in {filename}")

    return df


def make_sequences(values, window):
    X, y = [], []

    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_model(model_type, input_shape, n_outputs):
    inputs = tf.keras.Input(shape=input_shape)

    if model_type == "LSTM":
        x = tf.keras.layers.LSTM(32)(inputs)
    elif model_type == "GRU":
        x = tf.keras.layers.GRU(32)(inputs)
    else:
        raise ValueError("Unknown model type")

    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_outputs)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model


def metrics(y_true, y_pred):
    # MAPE is intentionally omitted: financial returns can be zero
    # or very close to zero, making percentage errors misleading.
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    train_df = load_returns("train_returns.csv")
    test_df = load_returns("test_returns.csv")

    train_dates = train_df["Date"]
    test_dates = test_df["Date"]

    if train_dates.max() >= test_dates.min():
        raise ValueError("Training and test dates overlap or are not chronological")

    train_raw = train_df[ASSETS].to_numpy(dtype=np.float64)
    test_raw = test_df[ASSETS].to_numpy(dtype=np.float64)

    if len(train_raw) <= WINDOW_SIZE + N_SPLITS:
        raise ValueError("Not enough training data")

    # Expanding-window validation. All preprocessing is fitted only
    # on the training portion of each fold.
    fold_size = len(train_raw) // (N_SPLITS + 1)
    validation_rows = []

    for model_type in ["LSTM", "GRU"]:
        for fold in range(N_SPLITS):
            fold_train_end = fold_size * (fold + 1)
            fold_val_end = (
                fold_size * (fold + 2)
                if fold < N_SPLITS - 1
                else len(train_raw)
            )

            fold_train = train_raw[:fold_train_end]
            fold_val = train_raw[fold_train_end:fold_val_end]

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(fold_train)
            scaled_val = scaler.transform(fold_val)

            # Include the preceding training observations as context
            # for the first validation prediction, without training on
            # validation observations.
            combined = np.vstack([scaled_train, scaled_val])
            X_all, y_all = make_sequences(combined, WINDOW_SIZE)

            val_start = len(scaled_train) - WINDOW_SIZE
            X_val = X_all[val_start:]
            y_val = y_all[val_start:]

            X_fit, y_fit = make_sequences(scaled_train, WINDOW_SIZE)

            tf.keras.backend.clear_session()
            tf.random.set_seed(SEED + fold)

            model = build_model(
                model_type,
                (WINDOW_SIZE, len(ASSETS)),
                len(ASSETS)
            )

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True
            )

            # Use the final portion of this fold's training data for
            # early stopping; never use the outer validation fold.
            split = int(len(X_fit) * 0.85)
            if split < 1 or split >= len(X_fit):
                raise ValueError("Insufficient sequences for inner validation")

            model.fit(
                X_fit[:split],
                y_fit[:split],
                validation_data=(X_fit[split:], y_fit[split:]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            pred_scaled = model.predict(X_val, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(y_val)

            for asset_idx, asset in enumerate(ASSETS):
                result = metrics(actual[:, asset_idx], pred[:, asset_idx])
                validation_rows.append({
                    "Model": model_type,
                    "Fold": fold + 1,
                    "Asset": asset,
                    "Train_End": str(train_dates.iloc[fold_train_end - 1].date()),
                    "Validation_Start": str(
                        train_dates.iloc[fold_train_end].date()
                    ),
                    "Validation_End": str(
                        train_dates.iloc[fold_val_end - 1].date()
                    ),
                    **result
                })

            print(f"Completed {model_type}, fold {fold + 1}/{N_SPLITS}")

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        OUTPUT_DIR / "walk_forward_validation.csv",
        index=False
    )

    # Select model family by mean validation RMSE across assets and folds.
    selection = (
        validation_df.groupby("Model")["RMSE"]
        .mean()
        .sort_values()
    )
    best_model_type = selection.index[0]

    # Final fit: scaler fitted on all training data only.
    final_scaler = StandardScaler()
    train_scaled = final_scaler.fit_transform(train_raw)

    # Test sequences use preceding training observations as history.
    combined_scaled = np.vstack([
        train_scaled,
        final_scaler.transform(test_raw)
    ])
    X_combined, y_combined = make_sequences(combined_scaled, WINDOW_SIZE)

    test_start = len(train_scaled) - WINDOW_SIZE
    X_test = X_combined[test_start:]
    y_test_scaled = y_combined[test_start:]
    y_test = final_scaler.inverse_transform(y_test_scaled)

    # Refit the selected model family using training data only.
    X_train, y_train = make_sequences(train_scaled, WINDOW_SIZE)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    final_model = build_model(
        best_model_type,
        (WINDOW_SIZE, len(ASSETS)),
        len(ASSETS)
    )

    final_split = int(len(X_train) * 0.9)
    final_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    final_model.fit(
        X_train[:final_split],
        y_train[:final_split],
        validation_data=(X_train[final_split:], y_train[final_split:]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[final_stop],
        verbose=0
    )

    pred_scaled = final_model.predict(X_test, verbose=0)
    predictions = final_scaler.inverse_transform(pred_scaled)

    test_rows = []
    for asset_idx, asset in enumerate(ASSETS):
        model_scores = metrics(y_test[:, asset_idx], predictions[:, asset_idx])
        zero_scores = metrics(
            y_test[:, asset_idx],
            np.zeros(len(y_test))
        )

        test_rows.append({
            "Model": best_model_type,
            "Asset": asset,
            **model_scores,
        })
        test_rows.append({
            "Model": "Zero-return baseline",
            "Asset": asset,
            **zero_scores,
        })

    pd.DataFrame(test_rows).to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False
    )

    forecast_df = pd.DataFrame({
        "Date": test_dates,
    })
    for i, asset in enumerate(ASSETS):
        forecast_df[f"{asset}_Actual"] = y_test[:, i]
        forecast_df[f"{asset}_Predicted"] = predictions[:, i]

    forecast_df.to_csv(
        OUTPUT_DIR / "final_test_forecasts.csv",
        index=False
    )

    with open(OUTPUT_DIR / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_model_type,
            "mean_validation_rmse_by_model": selection.to_dict(),
            "window_size": WINDOW_SIZE,
            "validation_folds": N_SPLITS,
            "training_rows": len(train_raw),
            "test_rows": len(test_raw),
            "assets": ASSETS,
            "note": (
                "One-step-ahead forecasts. Test observations are not used "
                "for model selection or fitting."
            )
        }, f, indent=2)

    print("\nValidation RMSE by model:")
    print(selection)
    print("\nFinal test metrics:")
    print(pd.DataFrame(test_rows).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
