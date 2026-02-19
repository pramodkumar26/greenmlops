# download_datasets.py

import os

# ── 1. AG NEWS (HuggingFace) ──────────────────────────────────
print("Downloading AG News...")
from datasets import load_dataset
ag_news = load_dataset("ag_news")
ag_news.save_to_disk("data/raw/ag_news")
print(f"AG News done — {len(ag_news['train'])} train, {len(ag_news['test'])} test samples")

# ── 2. CIFAR-100 (torchvision) ────────────────────────────────
print("\nDownloading CIFAR-100...")
import torchvision
cifar100_train = torchvision.datasets.CIFAR100(
    root="data/raw/cifar100", train=True, download=True
)
cifar100_test = torchvision.datasets.CIFAR100(
    root="data/raw/cifar100", train=False, download=True
)
print(f"CIFAR-100 done — {len(cifar100_train)} train, {len(cifar100_test)} test samples")

# ── 3. ETT (Electricity Transformer Temperature) ──────────────
print("\nDownloading ETT...")
import pandas as pd
import requests, os

os.makedirs("data/raw/ett", exist_ok=True)

ett_files = {
    "ETTh1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
}

for filename, url in ett_files.items():
    r = requests.get(url)
    filepath = f"data/raw/ett/{filename}"
    with open(filepath, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(filepath)
    print(f"  {filename} — {len(df)} rows, columns: {list(df.columns)}")

print("\nAll datasets downloaded successfully!")
print("\nData directory structure:")
for root, dirs, files in os.walk("data/raw"):
    level = root.replace("data/raw", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        size = os.path.getsize(os.path.join(root, f)) / 1024
        print(f"{indent}  {f} ({size:.1f} KB)")