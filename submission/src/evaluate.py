import argparse
import json
from typing import Dict, Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from dataset import (
    load_emg_dataset,
    subject_wise_split,
    EMGGestureDataset,
    create_dataloaders,
)
from model import build_model_from_config
from preprocessing import (
    load_normalization_stats,
    apply_z_score_normalization,
    design_bandpass_filter,
    apply_bandpass_filter,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def main(config_path: str, model_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]

    raw_root = data_cfg["raw_root"]
    session_names = data_cfg.get("session_names", None)
    val_subject_ids = data_cfg.get("val_subject_ids", [])

    samples, labels, subject_ids = load_emg_dataset(
        root=raw_root,
        session_names=session_names,
    )

    _, val_data = subject_wise_split(
        samples,
        labels,
        subject_ids,
        val_subject_ids=val_subject_ids,
    )

    x_val, y_val, val_subject_ids_arr = val_data

    if prep_cfg.get("bandpass", {}).get("enabled", False):
        bp_cfg = prep_cfg["bandpass"]
        b, a = design_bandpass_filter(
            low_hz=bp_cfg["low_hz"],
            high_hz=bp_cfg["high_hz"],
            sampling_rate_hz=bp_cfg["sampling_rate_hz"],
            order=bp_cfg.get("filter_order", 4),
        )
        x_val = apply_bandpass_filter(x_val, b, a)

    if prep_cfg.get("normalize", True):
        mean, std = load_normalization_stats(prep_cfg["normalization_stats_path"])
        x_val = apply_z_score_normalization(x_val, mean, std)

    val_dataset = EMGGestureDataset(x_val, y_val, val_subject_ids_arr)
    val_loader = create_dataloaders(
        val_dataset,
        val_dataset,
        batch_size=128,
        num_workers=4,
    )[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EMG gesture CNN model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/emg_gesture_model.pth",
        help="Path to trained model weights",
    )
    args = parser.parse_args()
    main(args.config, args.model)


