
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import mlflow
import dagshub
from codecarbon import EmissionsTracker

DAGSHUB_USER  = "pramodkumar26"
DAGSHUB_REPO  = "greenmlops"
DATA_DIR      = "/content/drive/MyDrive/greenmlops/airflow/include/data/raw/cifar100"
PROCESSED_DIR = "/content/drive/MyDrive/greenmlops/airflow/include/data/processed/cifar100"
EMISSIONS_DIR = "/content/drive/MyDrive/greenmlops/emissions"
RANDOM_STATE  = 42
EPOCHS        = 10
BATCH_SIZE    = 128
LR            = 1e-3

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda")

print(f"Device: {device} | {torch.cuda.get_device_name(0)}")

dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("baseline_training")

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,  download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform_test)

train_indices = np.load(f"{PROCESSED_DIR}/train_indices.npy")
val_indices   = np.load(f"{PROCESSED_DIR}/val_indices.npy")

train_subset = Subset(trainset, train_indices)
val_subset   = Subset(trainset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(testset,      batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(testset)}")

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

with mlflow.start_run(run_name="resnet18_cifar100_baseline"):

    mlflow.log_params({
        "model":         "ResNet-18",
        "dataset":       "cifar100",
        "urgency_class": "MEDIUM",
        "compute_type":  "GPU",
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LR,
        "train_size":    len(train_subset),
        "val_size":      len(val_subset),
        "test_size":     len(testset),
        "pretrained":    "ImageNet",
    })

    tracker = EmissionsTracker(
        project_name="greenmlops_resnet_baseline",
        output_dir=EMISSIONS_DIR,
        output_file="emissions_resnet.csv",
        log_level="error"
    )

    tracker.start()
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += model(x).argmax(1).eq(y).sum().item()
                val_total   += y.size(0)
        val_acc = val_correct / val_total

        mlflow.log_metrics({
            "train_loss": train_loss / len(train_loader),
            "train_acc":  train_acc,
            "val_acc":    val_acc,
        }, step=epoch)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

    t_end        = time.time()
    emissions_kg = tracker.stop()

    training_time = t_end - t_start
    carbon_g      = emissions_kg * 1000 if emissions_kg else None
    energy_kwh    = tracker._total_energy.kWh if emissions_kg else None

    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_correct += model(x).argmax(1).eq(y).sum().item()
            test_total   += y.size(0)
    test_acc = test_correct / test_total

    metrics = {
        "test_acc":              test_acc,
        "training_time_seconds": training_time,
    }
    if carbon_g is not None:
        metrics["carbon_gCO2"] = carbon_g
        metrics["energy_kWh"]  = energy_kwh

    mlflow.log_metrics(metrics)
    torch.save(model.state_dict(), f"{EMISSIONS_DIR}/../models/resnet18_baseline.pt")
    mlflow.pytorch.log_model(model, name="model")

    print("\n--- ResNet-18 CIFAR-100 Baseline ---")
    for k, v in metrics.items():
        print(f"{k:<30} {v:.6f}")
    print(f"\nhttps://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
