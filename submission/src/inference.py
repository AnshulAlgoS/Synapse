import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import yaml

from dataset import _load_single_csv, _discover_csv_files
from model import build_model_from_config
from preprocessing import apply_z_score_normalization


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_normalization_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    mean = data["mean"]
    std = data["std"]
    return mean, std


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_model_and_stats(config_path: str) -> Tuple[dict, torch.nn.Module, np.ndarray, np.ndarray, torch.device]:
    config = load_config(config_path)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    log_cfg = config["logging"]

    stats_path = prep_cfg["normalization_stats_path"]
    mean, std = load_normalization_stats(stats_path)

    device = select_device()
    model = build_model_from_config(config)
    model_path = log_cfg["best_model_path"]
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return config, model, mean, std, device


def _predict_windows(
    model: torch.nn.Module,
    device: torch.device,
    windows: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    all_preds: List[int] = []
    with torch.no_grad():
        num_windows = windows.shape[0]
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            batch_np = windows[start:end].astype("float32")
            batch = torch.from_numpy(batch_np).to(device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_preds, dtype=np.int64)


def run_inference_on_csv(
    csv_path: str,
    config_path: str,
    batch_size: int = 256,
) -> np.ndarray:
    config, model, mean, std, device = _prepare_model_and_stats(config_path)
    data_cfg = config["data"]

    window_size = int(data_cfg.get("window_size", 200))
    stride = int(data_cfg.get("window_stride", 50))

    windows, _ = _load_single_csv(csv_path, window_size, stride)
    if windows.shape[0] == 0:
        raise RuntimeError(f"No windows extracted from {csv_path}")

    windows = apply_z_score_normalization(windows, mean, std)
    preds = _predict_windows(model, device, windows, batch_size=batch_size)
    return preds


def run_inference_on_dataset(
    dataset_root: str,
    config_path: str,
    output_root: str,
    batch_size: int = 256,
) -> None:
    config, model, mean, std, device = _prepare_model_and_stats(config_path)
    data_cfg = config["data"]

    window_size = int(data_cfg.get("window_size", 200))
    stride = int(data_cfg.get("window_stride", 50))
    session_names = data_cfg.get("session_names", None)

    csv_files = _discover_csv_files(dataset_root, session_names=session_names)
    for csv_path, _ in csv_files:
        windows, _ = _load_single_csv(csv_path, window_size, stride)
        if windows.shape[0] == 0:
            continue
        windows_norm = apply_z_score_normalization(windows, mean, std)
        preds = _predict_windows(model, device, windows_norm, batch_size=batch_size)

        rel_path = os.path.relpath(csv_path, dataset_root)
        pred_path = os.path.join(output_root, rel_path) + ".npy"
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        np.save(pred_path, preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--csv",
        type=str,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="predictions.npy",
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="predictions",
    )
    args = parser.parse_args()

    if args.all:
        dataset_root = args.dataset_root
        if dataset_root is None:
            config = load_config(args.config)
            raw_root = config["data"]["raw_root"]
            if os.path.isabs(raw_root):
                dataset_root = raw_root
            else:
                project_root = os.path.dirname(os.path.dirname(__file__))
                dataset_root = os.path.abspath(os.path.join(project_root, raw_root))
        run_inference_on_dataset(
            dataset_root=dataset_root,
            config_path=args.config,
            output_root=args.out_dir,
        )
    else:
        if args.csv is None:
            raise SystemExit("Either --csv must be provided or --all must be set.")
        preds = run_inference_on_csv(
            csv_path=args.csv,
            config_path=args.config,
        )
        np.save(args.out, preds)
        print(f"Saved {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
