from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from .config import TargetScaler


@torch.inference_mode()
def detector_outputs(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = [], []
    model.eval()
    for waveform, target in loader:
        probabilities.extend(torch.softmax(model(waveform.to(device)), 1)[:, 1].cpu().numpy())
        labels.extend(target.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def select_detector_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation-set threshold with the highest F1 score."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


@torch.inference_mode()
def evaluate_detector(model, loader, device="cpu", threshold: float = 0.5) -> dict[str, float]:
    labels, probabilities = detector_outputs(model, loader, device)
    predictions = probabilities >= threshold
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


@torch.inference_mode()
def evaluate_estimator(model, loader, device="cpu", scaler: TargetScaler | None = None) -> dict[str, float]:
    if scaler is None:
        scaler = loader.dataset.dataset.target_scaler
    if scaler is None:
        raise ValueError("Estimator evaluation needs the fitted TargetScaler.")
    actual_mag, actual_distance, predicted_mag, predicted_distance = [], [], [], []
    model.eval()
    for waveform, target in loader:
        mag, distance, _ = model(waveform.to(device))
        actual_m, actual_d = scaler.inverse(target[:, 0].numpy(), target[:, 1].numpy())
        predicted_m, predicted_d = scaler.inverse(mag.cpu().numpy(), distance.cpu().numpy())
        actual_mag.extend(actual_m); actual_distance.extend(actual_d)
        predicted_mag.extend(predicted_m); predicted_distance.extend(predicted_d)
    return {"magnitude_mae": mean_absolute_error(actual_mag, predicted_mag), "magnitude_rmse": mean_squared_error(actual_mag, predicted_mag) ** .5, "magnitude_r2": r2_score(actual_mag, predicted_mag), "distance_mae_km": mean_absolute_error(actual_distance, predicted_distance), "distance_rmse_km": mean_squared_error(actual_distance, predicted_distance) ** .5, "distance_r2": r2_score(actual_distance, predicted_distance)}


@torch.inference_mode()
def evaluate_forecaster(model, loader, device="cpu") -> dict[str, float]:
    mse, mae, batches = 0.0, 0.0, 0
    model.eval()
    for encoder, target in loader:
        output, _ = model(encoder.to(device), target.to(device), teacher_force=0.0); mse += F.mse_loss(output, target.to(device)).item(); mae += F.l1_loss(output, target.to(device)).item(); batches += 1
    return {"waveform_mse": mse / batches, "waveform_rmse": (mse / batches) ** .5, "waveform_mae": mae / batches}
