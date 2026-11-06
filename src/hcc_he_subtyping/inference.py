from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from .decision import fuse_binary_predictions
from .model import CLAMSB, load_checkpoint


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_feature_bag(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {path}")
        features = np.asarray(handle["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(f"Expected an N x 768 feature matrix, found {features.shape} in {path}")
    return torch.from_numpy(features)


@torch.inference_mode()
def predict_bag(model: CLAMSB, features: torch.Tensor, device: torch.device):
    _, probability, prediction, attention, _ = model(features.to(device))
    return int(prediction.item()), probability.cpu().numpy()[0], attention.cpu().numpy()[0]


def run(config_path: str | Path) -> pd.DataFrame:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent
    resolve = lambda value: (base / value).resolve() if not Path(value).is_absolute() else Path(value)
    device = choose_device(config.get("device", "auto"))
    model_args = config["model"]
    model1 = load_checkpoint(CLAMSB(**model_args), resolve(config["weights"]["m1_vs_m23"]), device)
    model2 = load_checkpoint(CLAMSB(**model_args), resolve(config["weights"]["m12_vs_m3"]), device)

    rows = []
    feature_paths = sorted(resolve(config["features_dir"]).glob("*.h5"))
    if not feature_paths:
        raise FileNotFoundError("No HDF5 feature bags were found")
    for index, path in enumerate(feature_paths, 1):
        features = read_feature_bag(path)
        pred1, prob1, _ = predict_bag(model1, features, device)
        pred2, prob2, _ = predict_bag(model2, features, device)
        rows.append({
            "slide_id": path.stem,
            "m1_probability": float(prob1[0]),
            "m23_probability": float(prob1[1]),
            "m12_probability": float(prob2[0]),
            "m3_probability": float(prob2[1]),
            "m1_vs_m23_prediction": "M1" if pred1 == 0 else "M2&M3",
            "m12_vs_m3_prediction": "M1&M2" if pred2 == 0 else "M3",
            "final_subtype": fuse_binary_predictions(pred1, pred2),
        })
        print(f"[{index}/{len(feature_paths)}] {path.stem}: {rows[-1]['final_subtype']}", flush=True)

    result = pd.DataFrame(rows)
    output_path = resolve(config["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer HCC molecular ecosystem subtypes")
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

