from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import CLAMSB
from .splits import patient_level_split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureBagDataset(Dataset):
    def __init__(self, table: pd.DataFrame, task: str):
        self.table = table.reset_index(drop=True)
        self.task = task

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        with h5py.File(row.feature_path, "r") as handle:
            features = torch.from_numpy(np.asarray(handle["features"], dtype=np.float32))
        if self.task == "m1_vs_m23":
            label = 0 if row.subtype == "M1" else 1
        elif self.task == "m12_vs_m3":
            label = 1 if row.subtype == "M3" else 0
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return features, torch.tensor(label), row.slide_id


def collate_one(batch):
    return batch[0]


def make_loader(dataset: FeatureBagDataset, training: bool) -> DataLoader:
    if training:
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
        counts = np.bincount(labels, minlength=2)
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_one)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one)


@torch.inference_mode()
def evaluate(model: CLAMSB, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, probabilities = [], []
    for features, label, _ in loader:
        _, probability, _, _, _ = model(features.to(device))
        labels.append(int(label))
        probabilities.append(float(probability[0, 1]))
    predictions = [int(value >= 0.5) for value in probabilities]
    auc = roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else float("nan")
    return {"auc": auc, "accuracy": accuracy_score(labels, predictions)}


def train_task(
    split_table: pd.DataFrame,
    task: str,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    datasets = {
        name: FeatureBagDataset(split_table[split_table.split == name], task)
        for name in ("train", "validation", "test")
    }
    loaders = {name: make_loader(dataset, name == "train") for name, dataset in datasets.items()}
    model = CLAMSB(dropout=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_auc = -np.inf
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, label, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits, _, _, _, _ = model(features.to(device))
            loss = criterion(logits, label.view(1).to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, loaders["validation"], device)
        print(
            f"[{task}] epoch {epoch:03d}/{epochs} loss={np.mean(losses):.4f} "
            f"val_auc={validation['auc']:.4f} val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
        score = validation["auc"] if np.isfinite(validation["auc"]) else validation["accuracy"]
        if score > best_auc:
            best_auc = score
            torch.save(model.state_dict(), output_dir / f"clam_sb_{task}.pth")

    best = torch.load(output_dir / f"clam_sb_{task}.pth", map_location=device)
    model.load_state_dict(best)
    test_metrics = evaluate(model, loaders["test"], device)
    pd.DataFrame([{"task": task, **test_metrics}]).to_csv(
        output_dir / f"{task}_held_out_test_metrics.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the two CLAM-SB classifiers")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="runs/training")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    manifest = pd.read_csv(args.manifest)
    split_table = patient_level_split(manifest, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(output_dir / "split_70_20_10.csv", index=False)
    for task in ("m1_vs_m23", "m12_vs_m3"):
        train_task(
            split_table, task, output_dir, args.epochs,
            args.learning_rate, args.weight_decay, device,
        )


if __name__ == "__main__":
    main()

