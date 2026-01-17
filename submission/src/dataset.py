import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class EMGGestureDataset(Dataset):
    """
    PyTorch Dataset for sEMG gesture classification.
    """

    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        samples: (N, C, T) array of sEMG windows
        labels: (N,) array of integer labels
        subject_ids: (N,) array of subject identifiers (for splitting)
        """
        super().__init__()
        self.samples = samples
        self.labels = labels
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        x = torch.from_numpy(self.samples[idx].astype("float32"))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _discover_csv_files(
    root: str,
    session_names: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    csv_files: List[Tuple[str, str]] = []
    if session_names is None:
        session_dirs = [d for d in os.listdir(root) if d.startswith("Session")]
    else:
        session_dirs = session_names

    for session in session_dirs:
        session_path = os.path.join(root, session)
        if not os.path.isdir(session_path):
            continue
        for subject_dir in os.listdir(session_path):
            subject_path = os.path.join(session_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            subject_id = subject_dir
            for fname in os.listdir(subject_path):
                if fname.lower().endswith(".csv"):
                    csv_files.append((os.path.join(subject_path, fname), subject_id))
    return csv_files


def _load_single_csv(
    path: str,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CSV file, extract label from filename, and slice into windows.
    """
    filename = os.path.basename(path)
    # Parse label from filename: e.g., "gesture00_trial01.csv"
    match = re.search(r"gesture(\d+)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse gesture label from filename: {filename}")
    label = int(match.group(1))

    # Read CSV
    # The files have headers like "ch1,ch2,..." so header=0 is appropriate
    df = pd.read_csv(path)
    data = df.values.astype(np.float32)

    if data.shape[1] != 8:
        # If there's an issue with features, handle or raise
        raise ValueError(f"Expected 8 channels, got {data.shape[1]} in {path}")

    # Sliding window
    num_samples = data.shape[0]
    windows = []

    if num_samples < window_size:
        # Pad with zeros if trial is shorter than window
        pad_len = window_size - num_samples
        padded = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")
        windows.append(padded.T)  # (8, T)
    else:
        # Create overlapping windows
        for start in range(0, num_samples - window_size + 1, stride):
            end = start + window_size
            window = data[start:end, :]  # (T, 8)
            windows.append(window.T)     # (8, T)

    if not windows:
        return np.empty((0, 8, window_size)), np.empty((0,))

    return np.array(windows), np.full((len(windows),), label, dtype=np.int64)


def load_emg_dataset(
    root: str,
    session_names: Optional[List[str]] = None,
    window_size: int = 200,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the entire EMG dataset from the given root directory.

    Returns:
        samples: (N, 8, T)
        labels: (N,)
        subject_ids: (N,) array of strings indicating the recording subject
    """
    csv_files = _discover_csv_files(root, session_names)
    all_samples: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subject_ids: List[str] = []
    
    for csv_path, subject_id in csv_files:
        try:
            samples, labels = _load_single_csv(csv_path, window_size, stride)
        except ValueError as exc:
            print(f"Skipping file {csv_path}: {exc}")
            continue
        
        if len(samples) > 0:
            all_samples.append(samples)
            all_labels.append(labels)
            all_subject_ids.extend([subject_id] * len(samples))

    if not all_samples:
        raise RuntimeError(f"No CSV files found under {root}")

    samples_arr = np.concatenate(all_samples, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    subject_ids_arr = np.array(all_subject_ids)

    return samples_arr, labels_arr, subject_ids_arr


def subject_wise_split(
    samples: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    val_subject_ids: List[str],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split dataset into training and validation sets based on subject IDs.

    All windows from a subject are assigned either to training or validation,
    eliminating subject-level data leakage.
    """
    val_mask = np.isin(subject_ids, val_subject_ids)
    train_mask = ~val_mask

    x_train = samples[train_mask]
    y_train = labels[train_mask]
    subj_train = subject_ids[train_mask]

    x_val = samples[val_mask]
    y_val = labels[val_mask]
    subj_val = subject_ids[val_mask]

    return (x_train, y_train, subj_train), (x_val, y_val, subj_val)
