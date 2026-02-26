import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import dagshub
from codecarbon import EmissionsTracker
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR      = r"C:\IP\greenmlops\airflow\include\data\processed\ett"
DAGSHUB_USER  = "pramodkumar26"
DAGSHUB_REPO  = "greenmlops"
EMISSIONS_DIR = r"C:\IP\greenmlops\emissions"
RANDOM_STATE  = 42

SEQ_LEN    = 24
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2

FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL   = "OT"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("baseline_training")


def load_and_scale(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    val   = pd.read_csv(f"{data_dir}/val.csv",   parse_dates=["date"])
    test  = pd.read_csv(f"{data_dir}/test.csv",  parse_dates=["date"])

    cols = FEATURE_COLS + [TARGET_COL]

    # fit scaler on train only
    mins  = train[cols].min()
    maxs  = train[cols].max()
    rngs  = (maxs - mins).replace(0, 1e-8)

    def scale(df):
        out = df.copy()
        out[cols] = (df[cols] - mins) / rngs
        return out

    return scale(train), scale(val), scale(test), mins, maxs


class ETTDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        cols         = FEATURE_COLS + [TARGET_COL]
        self.data    = df[cols].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len, :-1]
        y = self.data[idx + self.seq_len, -1]
        return torch.tensor(x), torch.tensor(y)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse    = np.sqrt(mean_squared_error(targets, preds))
    mae     = mean_absolute_error(targets, preds)
    return rmse, mae


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_df, val_df, test_df, mins, maxs = load_and_scale(DATA_DIR)

train_ds = ETTDataset(train_df, SEQ_LEN)
val_ds   = ETTDataset(val_df,   SEQ_LEN)
test_ds  = ETTDataset(test_df,  SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

with mlflow.start_run(run_name="lstm_ett_baseline"):

    mlflow.log_params({
        "model":         "LSTM",
        "dataset":       "ett",
        "urgency_class": "LOW",
        "compute_type":  "CPU",
        "seq_len":       SEQ_LEN,
        "hidden_dim":    HIDDEN_DIM,
        "num_layers":    NUM_LAYERS,
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LR,
        "train_size":    len(train_ds),
        "val_size":      len(val_ds),
        "test_size":     len(test_ds),
    })

    model     = LSTMForecaster(len(FEATURE_COLS), HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    tracker = EmissionsTracker(
        project_name="greenmlops_ett_baseline",
        output_dir=EMISSIONS_DIR,
        output_file="emissions_ett.csv",
        log_level="error"
    )

    tracker.start()
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_rmse, val_mae = evaluate(model, val_loader, device)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_rmse":   val_rmse,
            "val_mae":    val_mae,
        }, step=epoch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss:.4f} | val_rmse: {val_rmse:.4f} | val_mae: {val_mae:.4f}")

    t_end        = time.time()
    emissions_kg = tracker.stop()

    training_time = t_end - t_start
    carbon_g      = emissions_kg * 1000 if emissions_kg else None
    energy_kwh    = tracker._total_energy.kWh if emissions_kg else None

    test_rmse, test_mae = evaluate(model, test_loader, device)

    metrics = {
        "test_rmse":             test_rmse,
        "test_mae":              test_mae,
        "training_time_seconds": training_time,
    }
    if carbon_g is not None:
        metrics["carbon_gCO2"] = carbon_g
        metrics["energy_kWh"]  = energy_kwh

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, name="model")

    print("\n--- ETT LSTM Baseline ---")
    for k, v in metrics.items():
        print(f"{k:<30} {v:.6f}")
    print(f"\nhttps://dagshub.com/{pramodk26}/{DAGSHUB_REPO}.mlflow")