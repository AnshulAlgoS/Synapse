import argparse
import json
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    compute_channelwise_mean_std,
    apply_z_score_normalization,
    design_bandpass_filter,
    apply_bandpass_filter,
    save_normalization_stats,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in loader:
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)
    return metrics


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    train_cfg = config["training"]
    log_cfg = config["logging"]

    raw_root_cfg = data_cfg["raw_root"]
    if os.path.isabs(raw_root_cfg):
        raw_root = raw_root_cfg
    else:
        project_root = os.path.dirname(os.path.dirname(__file__))
        raw_root = os.path.abspath(os.path.join(project_root, raw_root_cfg))
    session_names = data_cfg.get("session_names", None)
    val_subject_ids = data_cfg.get("val_subject_ids", [])
    window_size = int(data_cfg.get("window_size", 200))
    stride = int(data_cfg.get("window_stride", 50))

    print(f"Loading EMG dataset from: {raw_root} with window_size={window_size}, stride={stride}")
    samples, labels, subject_ids = load_emg_dataset(
        root=raw_root,
        session_names=session_names,
        window_size=window_size,
        stride=stride,
    )
    print(f"Loaded samples: {samples.shape}, labels: {labels.shape}")

    train_data, val_data = subject_wise_split(
        samples,
        labels,
        subject_ids,
        val_subject_ids=val_subject_ids,
    )
    print("Completed subject-wise train/val split")

    x_train, y_train, train_subject_ids = train_data
    x_val, y_val, val_subject_ids_arr = val_data

    # Optional band-pass filtering applied symmetrically to training and validation
    if prep_cfg.get("bandpass", {}).get("enabled", False):
        print("Starting band-pass filtering...")
        bp_cfg = prep_cfg["bandpass"]
        b, a = design_bandpass_filter(
            low_hz=bp_cfg["low_hz"],
            high_hz=bp_cfg["high_hz"],
            sampling_rate_hz=bp_cfg["sampling_rate_hz"],
            order=bp_cfg.get("filter_order", 4),
        )
        x_train = apply_bandpass_filter(x_train, b, a)
        x_val = apply_bandpass_filter(x_val, b, a)
        print("Finished band-pass filtering")

    # Channel-wise normalization statistics are estimated on training set only
    if prep_cfg.get("normalize", True):
        print("Starting normalization...")
        mean, std = compute_channelwise_mean_std(x_train)
        print(f"Computed mean: {mean}, std: {std}")
        x_train = apply_z_score_normalization(x_train, mean, std)
        x_val = apply_z_score_normalization(x_val, mean, std)
        print("Finished normalization")

        stats_path = prep_cfg["normalization_stats_path"]
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        save_normalization_stats(stats_path, mean, std)

    train_dataset = EMGGestureDataset(x_train, y_train, train_subject_ids)
    val_dataset = EMGGestureDataset(x_val, y_val, val_subject_ids_arr)

    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = build_model_from_config(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 50))
    os.makedirs(log_cfg["output_dir"], exist_ok=True)
    best_model_path = log_cfg["best_model_path"]

    best_f1 = -1.0
    history = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = float(train_loss)
        history.append(val_metrics)

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"TrainLoss={train_loss:.4f} "
            f"ValLoss={val_metrics['loss']:.4f} "
            f"ValF1={val_metrics['f1_macro']:.4f}"
        )

        # Use macro F1-score as early-selection criterion
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with F1={best_f1:.4f} to {best_model_path}")

    metrics_path = log_cfg.get("metrics_path", "models/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMG gesture CNN model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)
