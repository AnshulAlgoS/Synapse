import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from model import build_model_from_config
from preprocessing import (
    load_normalization_stats,
    apply_z_score_normalization,
    design_bandpass_filter,
    apply_bandpass_filter,
)


def load_csv_window_file(path: str) -> Tuple[np.ndarray, bool]:
    """
    Load a CSV file containing flattened sEMG windows.

    If the last column appears to encode gesture labels (integer in [0, 4]),
    it is ignored during inference but returned as a flag for optional analysis.
    """
    df = pd.read_csv(path, header=None)
    data = df.values
    num_cols = data.shape[1]
    has_label = False

    if num_cols >= 9:
        last_col = data[:, -1]
        if np.issubdtype(last_col.dtype, np.integer) or np.all(
            np.in1d(np.unique(last_col), np.array([0, 1, 2, 3, 4]))
        ):
            features = data[:, :-1]
            has_label = True
        else:
            features = data
    else:
        features = data

    num_features = features.shape[1]
    if num_features % 8 != 0:
        raise ValueError(
            f"Expected number of features to be divisible by 8, got {num_features}"
        )
    time_steps = num_features // 8
    samples = features.reshape(-1, 8, time_steps)
    return samples.astype(np.float32), has_label


def main(
    config_path: str,
    model_path: str,
    input_csv: str,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prep_cfg = config["preprocessing"]

    samples, _ = load_csv_window_file(input_csv)

    if prep_cfg.get("bandpass", {}).get("enabled", False):
        bp_cfg = prep_cfg["bandpass"]
        b, a = design_bandpass_filter(
            low_hz=bp_cfg["low_hz"],
            high_hz=bp_cfg["high_hz"],
            sampling_rate_hz=bp_cfg["sampling_rate_hz"],
            order=bp_cfg.get("filter_order", 4),
        )
        samples = apply_bandpass_filter(samples, b, a)

    if prep_cfg.get("normalize", True):
        stats_path = prep_cfg["normalization_stats_path"]
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Normalization stats not found at {stats_path}. "
                f"Run training first to compute and save them."
            )
        mean, std = load_normalization_stats(stats_path)
        samples = apply_z_score_normalization(samples, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            f"Train the model before running inference."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(samples).to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    for i, cls in enumerate(preds):
        print(f"Window {i}: Predicted gesture class: {int(cls)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on EMG CSV file")
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
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing sEMG windows",
    )
    args = parser.parse_args()
    main(args.config, args.model, args.input)


