import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import mlflow
import dagshub
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score, accuracy_score

DAGSHUB_USER  = "pramodkumar26"
DAGSHUB_REPO  = "greenmlops"
DATA_DIR      = "/content/drive/MyDrive/greenmlops/airflow/include/data/processed/ag_news"
EMISSIONS_DIR = "/content/drive/MyDrive/greenmlops/emissions"
MODELS_DIR    = "/content/drive/MyDrive/greenmlops/models"
RANDOM_STATE  = 42
EPOCHS        = 5
BATCH_SIZE    = 32
LR            = 2e-5
MAX_LEN       = 128

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda")
print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("baseline_training")


class AGNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_ds = AGNewsDataset(train_df, tokenizer, MAX_LEN)
val_ds   = AGNewsDataset(val_df,   tokenizer, MAX_LEN)
test_ds  = AGNewsDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

with mlflow.start_run(run_name="distilbert_agnews_baseline"):

    mlflow.log_params({
        "model":         "DistilBERT",
        "dataset":       "ag_news",
        "urgency_class": "MEDIUM",
        "compute_type":  "GPU",
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LR,
        "max_len":       MAX_LEN,
        "train_size":    len(train_ds),
        "val_size":      len(val_ds),
        "test_size":     len(test_ds),
        "pretrained":    "distilbert-base-uncased",
    })

    model     = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    tracker = EmissionsTracker(
        project_name="greenmlops_distilbert_baseline",
        output_dir=EMISSIONS_DIR,
        output_file="emissions_distilbert.csv",
        log_level="error"
    )

    tracker.start()
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1  = f1_score(all_labels, all_preds, average="macro")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                val_preds.extend(out.logits.argmax(-1).cpu().numpy())
                val_labels.extend(batch["label"].numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1  = f1_score(val_labels, val_preds, average="macro")

        mlflow.log_metrics({
            "train_loss": train_loss / len(train_loader),
            "train_acc":  train_acc,
            "train_f1":   train_f1,
            "val_acc":    val_acc,
            "val_f1":     val_f1,
        }, step=epoch)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")

    t_end        = time.time()
    emissions_kg = tracker.stop()

    training_time = t_end - t_start
    energy_kwh    = tracker._total_energy.kWh if emissions_kg else None
    carbon_g      = emissions_kg * 1000 if emissions_kg else None

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds.extend(out.logits.argmax(-1).cpu().numpy())
            test_labels.extend(batch["label"].numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average="macro")

    metrics = {
        "test_acc":              test_acc,
        "test_f1":               test_f1,
        "training_time_seconds": training_time,
        "energy_kWh":            energy_kwh,
        "carbon_gCO2":           carbon_g,
    }

    mlflow.log_metrics(metrics)
    torch.save(model.state_dict(), f"{MODELS_DIR}/distilbert_baseline.pt")
    mlflow.pytorch.log_model(model, name="model")

    print("\n--- DistilBERT AG News Baseline ---")
    for k, v in metrics.items():
        print(f"{k:<30} {v:.6f}")
    print(f"\nhttps://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")