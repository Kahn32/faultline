from __future__ import annotations

import numpy as np


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize each seismic channel over time. Input/output: (time, 3)."""
    mean = waveform.mean(axis=0, keepdims=True)
    std = waveform.std(axis=0, keepdims=True) + 1e-8
    return (waveform - mean) / std


def ensure_time_channels(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 2:
        raise ValueError("A waveform must be a 2-D array.")
    if waveform.shape[1] == 3:
        return waveform
    if waveform.shape[0] == 3:
        return waveform.T
    raise ValueError(f"Expected three channels; received shape {waveform.shape}.")
