from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .config import SignalConfig, TargetScaler
from .preprocessing import ensure_time_channels, normalize_waveform


class STEADDataset(Dataset):
    """STEAD-backed samples for classification, regression, and forecasting."""

    def __init__(self, hdf5_path: str, csv_path: str, task: str, config: SignalConfig = SignalConfig(), max_samples: int | None = None):
        self.hdf5_path = str(hdf5_path)
        self.task = task
        self.config = config
        # low_memory=False prevents noisy mixed-type warnings from unused STEAD columns.
        metadata = pd.read_csv(csv_path, low_memory=False)
        if task in {"magnitude", "forecasting"}:
            metadata = metadata[
                (metadata.trace_category == "earthquake_local")
                & metadata.p_arrival_sample.notna()
                & metadata.source_magnitude.notna()
                & metadata.source_distance_km.notna()
            ]
            if task == "forecasting":
                metadata = metadata[metadata.s_arrival_sample.notna()]
        if max_samples is not None:
            metadata = metadata.sample(min(max_samples, len(metadata)), random_state=42)
        self.metadata = metadata.reset_index(drop=True)
        self._h5_file: h5py.File | None = None
        self.target_scaler: TargetScaler | None = None
        if self.metadata.empty:
            raise ValueError(
                f"No usable {task} samples were found. Magnitude and forecasting require "
                "earthquake_local records with P-arrival and target metadata."
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def _waveform(self, row: pd.Series) -> np.ndarray:
        # Notebook/macOS uses num_workers=0. Keeping one handle per Dataset avoids
        # opening a 14+ GB HDF5 file once for every individual trace.
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
        return ensure_time_channels(self._h5_file["data"][row.trace_name][:])

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def _p_start(self, row: pd.Series, length: int, total: int) -> int:
        start = max(0, int(row.p_arrival_sample) - self.config.p_before)
        return min(start, total - length)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        waveform = self._waveform(row)
        c = self.config

        if self.task == "detection":
            is_event = int(row.trace_category == "earthquake_local")
            if is_event and pd.notna(row.p_arrival_sample):
                start = self._p_start(row, c.detector_window, len(waveform))
            else:
                rng = np.random.default_rng(index)
                start = int(rng.integers(0, len(waveform) - c.detector_window))
            window = waveform[start : start + c.detector_window]
            return torch.tensor(normalize_waveform(window)), torch.tensor(is_event, dtype=torch.long)

        start = self._p_start(row, c.estimator_window if self.task == "magnitude" else c.forecast_encoder_window + c.forecast_decoder_window, len(waveform))
        if self.task == "magnitude":
            window = normalize_waveform(waveform[start : start + c.estimator_window])
            if self.target_scaler is None:
                raise RuntimeError("Magnitude targets require a scaler fitted on the training split.")
            magnitude, distance = self.target_scaler.transform(
                float(row.source_magnitude), float(row.source_distance_km)
            )
            return torch.tensor(window), torch.tensor([magnitude, distance], dtype=torch.float32)

        segment = normalize_waveform(waveform[start : start + c.forecast_encoder_window + c.forecast_decoder_window])
        return torch.tensor(segment[: c.forecast_encoder_window]), torch.tensor(segment[c.forecast_encoder_window :])


def make_loaders(dataset: STEADDataset, batch_size: int, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Event-disjoint split: each earthquake source_id belongs to one partition."""
    meta = dataset.metadata.copy()
    fallback_groups = pd.Series("noise-" + meta.index.astype(str), index=meta.index)
    groups = meta.source_id.where(meta.source_id.notna(), fallback_groups).astype(str)
    unique_groups = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    train_end, val_end = int(.70 * len(unique_groups)), int(.85 * len(unique_groups))
    group_split = {g: "train" for g in unique_groups[:train_end]}
    group_split.update({g: "val" for g in unique_groups[train_end:val_end]})
    group_split.update({g: "test" for g in unique_groups[val_end:]})
    indices = {split: np.flatnonzero(groups.map(group_split).to_numpy() == split).tolist() for split in ("train", "val", "test")}
    if dataset.task == "magnitude":
        train_meta = meta.iloc[indices["train"]]
        dataset.target_scaler = TargetScaler.fit(
            train_meta.source_magnitude, train_meta.source_distance_km
        )
    return tuple(
        DataLoader(Subset(dataset, indices[split]), batch_size=batch_size if split == "train" else batch_size * 2, shuffle=split == "train", num_workers=0, pin_memory=False)
        for split in ("train", "val", "test")
    )
