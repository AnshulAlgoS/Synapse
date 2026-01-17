from typing import Tuple, Optional

import numpy as np
from scipy.signal import butter, filtfilt


def compute_channelwise_mean_std(
    samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and standard deviation for z-score normalization.

    samples: (N, C, T)
    Returns:
        mean: (C,)
        std: (C,)
    """
    assert samples.ndim == 3, "Expected samples of shape (N, C, T)"
    # Flatten across batch and time for each channel
    c_mean = samples.mean(axis=(0, 2))
    c_std = samples.std(axis=(0, 2))
    # Avoid division by zero
    c_std[c_std == 0.0] = 1.0
    return c_mean.astype(np.float32), c_std.astype(np.float32)


def apply_z_score_normalization(
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply channel-wise z-score normalization.

    samples: (N, C, T)
    mean, std: (C,)
    """
    assert samples.ndim == 3
    assert mean.shape[0] == samples.shape[1]
    assert std.shape[0] == samples.shape[1]
    mean_reshaped = mean.reshape(1, -1, 1)
    std_reshaped = std.reshape(1, -1, 1)
    return (samples - mean_reshaped) / std_reshaped


def design_bandpass_filter(
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float,
    order: int = 4,
):
    """
    Design a Butterworth band-pass filter for sEMG.
    """
    nyquist = 0.5 * sampling_rate_hz
    low = low_hz / nyquist
    high = high_hz / nyquist
    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid band-pass frequencies: low={low_hz}, high={high_hz}, "
            f"sampling_rate={sampling_rate_hz}"
        )
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(
    samples: np.ndarray,
    b,
    a,
) -> np.ndarray:
    """
    Apply band-pass filtering along the temporal dimension for each channel.

    samples: (N, C, T)
    """
    assert samples.ndim == 3
    n, c, t = samples.shape
    max_len = max(len(a), len(b))
    padlen = 3 * (max_len - 1)
    if t <= padlen:
        print(f"Signal length {t} <= padlen {padlen}. Skipping band-pass filter.")
        return samples.copy()
    
    # Vectorized filtering along the last axis (time)
    # filtfilt handles multidimensional arrays by filtering along axis=-1 by default
    filtered = filtfilt(b, a, samples, axis=-1)
    return filtered


def save_normalization_stats(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """
    Persist normalization statistics to disk for reuse at inference time.
    """
    np.savez(path, mean=mean, std=std)


def load_normalization_stats(
    path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization statistics from disk.
    """
    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    return mean, std

