"""
Train a neural net emulator for JOS-3.

Takes training_data.csv (from generate_training_data.py) and trains
a small feedforward network to predict JOS-3 outputs from climate +
worker inputs. Validates on held-out 20% and reports RMSE per output.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time

# --- Config ---
INPUT_COLS = ["Ta", "Tr", "RH", "Va", "PAR", "clo",
              "height", "weight", "age", "sex_male"]

OUTPUT_COLS = [
    "peak_core", "final_core",
    "peak_skin", "final_skin",
    "peak_wet", "final_wet",
    "time_to_38", "time_to_385", "time_to_39",
    "core_rate",
    "Tsk_Head", "Tsk_Chest", "Tsk_Back", "Tsk_Pelvis",
    "Tsk_LHand", "Tsk_RHand", "Tsk_LFoot", "Tsk_RFoot",
]

HIDDEN_SIZES = [128, 128, 64]
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JOS3Emulator(nn.Module):
    """Small feedforward net: inputs → JOS-3 outputs."""

    def __init__(self, n_in, n_out, hidden_sizes):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(path="emulator/training_data.csv"):
    """Load and preprocess training data."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples")

    # Encode sex as binary
    df["sex_male"] = (df["sex"] == "male").astype(float)

    # Replace time_to_* = -1 (never reached) with SIM_DURATION + 1
    for col in ["time_to_38", "time_to_385", "time_to_39"]:
        df[col] = df[col].replace(-1, 121)

    X = df[INPUT_COLS].values.astype(np.float32)
    Y = df[OUTPUT_COLS].values.astype(np.float32)

    return X, Y


def train(X, Y):
    """Train the emulator and return model + scalers."""
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    Y_train_s = y_scaler.transform(Y_train)
    Y_test_s = y_scaler.transform(Y_test)

    # Dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train_s), torch.tensor(Y_train_s)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s), torch.tensor(Y_test_s)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = JOS3Emulator(
        n_in=len(INPUT_COLS),
        n_out=len(OUTPUT_COLS),
        hidden_sizes=HIDDEN_SIZES,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}")

    # Train
    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train_s)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(X_test_s)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Train MSE: {train_loss:.6f} | "
                  f"Val MSE: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Load best
    model.load_state_dict(best_state)
    print(f"\nBest validation MSE (scaled): {best_loss:.6f}")

    # Evaluate in original units
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_s).to(DEVICE)
        Y_pred_s = model(X_test_t).cpu().numpy()

    Y_pred = y_scaler.inverse_transform(Y_pred_s)
    Y_true = Y_test

    print("\n--- Validation RMSE (original units) ---")
    rmse_results = {}
    for i, col in enumerate(OUTPUT_COLS):
        rmse = np.sqrt(np.mean((Y_pred[:, i] - Y_true[:, i]) ** 2))
        mae = np.mean(np.abs(Y_pred[:, i] - Y_true[:, i]))
        rmse_results[col] = {"rmse": round(float(rmse), 4),
                             "mae": round(float(mae), 4)}

        unit = "°C" if "core" in col or "skin" in col or "Tsk" in col else ""
        if "time" in col:
            unit = "min"
        if "wet" in col:
            unit = ""
        if "rate" in col:
            unit = "°C/min"
        print(f"  {col:20s}  RMSE: {rmse:.4f} {unit:8s}  MAE: {mae:.4f}")

    return model, x_scaler, y_scaler, rmse_results


def save_model(model, x_scaler, y_scaler, rmse_results):
    """Save model weights, scalers, and metadata."""
    torch.save(model.state_dict(), "emulator/jos3_emulator.pt")

    # Save scalers as JSON (portable)
    scaler_data = {
        "x_mean": x_scaler.mean_.tolist(),
        "x_scale": x_scaler.scale_.tolist(),
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
        "hidden_sizes": HIDDEN_SIZES,
    }
    with open("emulator/scalers.json", "w") as f:
        json.dump(scaler_data, f, indent=2)

    with open("emulator/validation_rmse.json", "w") as f:
        json.dump(rmse_results, f, indent=2)

    print("\nSaved:")
    print("  emulator/jos3_emulator.pt (model weights)")
    print("  emulator/scalers.json (input/output scalers)")
    print("  emulator/validation_rmse.json (accuracy metrics)")


def main():
    t0 = time.time()
    X, Y = load_data()
    model, x_scaler, y_scaler, rmse = train(X, Y)
    save_model(model, x_scaler, y_scaler, rmse)
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
