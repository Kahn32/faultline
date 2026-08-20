"""Compare corrected models to notebook baselines on identical event-disjoint tests."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
EARTHQUAKES = PROJECT.parent
sys.path.insert(0, str(PROJECT / "backend"))

from seismic_lab.config import SignalConfig, TargetScaler  # noqa: E402
from seismic_lab.data import STEADDataset, make_loaders  # noqa: E402
from seismic_lab.evaluate import detector_outputs, evaluate_detector, evaluate_forecaster, select_detector_threshold  # noqa: E402
from seismic_lab.models import GRUDetector, LSTMEstimator, SeismicSeq2Seq  # noqa: E402

CORRECTED = PROJECT / "artifacts" / "corrected"
APPROVED = PROJECT / "artifacts" / "approved"
BASELINE_ARCHIVE = PROJECT / "artifacts" / "baselines" / "notebook-v0.1"
EVENT_HDF5 = EARTHQUAKES / "chunk2.hdf5"
EVENT_CSV = EARTHQUAKES / "chunk2.csv"
DETECTION_HDF5 = Path("/Volumes/USB321FD/seismic-signal-lab-data/detection.hdf5")
DETECTION_CSV = Path("/Volumes/USB321FD/seismic-signal-lab-data/detection.csv")
DEVICE = torch.device("cpu")


def archive_notebook_bundle() -> None:
    BASELINE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in ("gru_detector.pt", "lstm_estimator.pt", "seq2seq.pt", "target_scaler.json", "model_metadata.json"):
        source = APPROVED / name
        destination = BASELINE_ARCHIVE / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


@torch.inference_mode()
def evaluate_estimator_with_scalers(model, loader, prediction_scaler: TargetScaler, target_scaler: TargetScaler) -> dict[str, float]:
    actual_magnitude, actual_distance, predicted_magnitude, predicted_distance = [], [], [], []
    model.eval()
    for waveform, target in loader:
        magnitude, distance, _ = model(waveform.to(DEVICE))
        actual_m, actual_d = target_scaler.inverse(target[:, 0].numpy(), target[:, 1].numpy())
        predicted_m, predicted_d = prediction_scaler.inverse(magnitude.cpu().numpy(), distance.cpu().numpy())
        actual_magnitude.extend(actual_m); actual_distance.extend(actual_d)
        predicted_magnitude.extend(predicted_m); predicted_distance.extend(predicted_d)
    actual_magnitude, actual_distance = np.asarray(actual_magnitude), np.asarray(actual_distance)
    predicted_magnitude, predicted_distance = np.asarray(predicted_magnitude), np.asarray(predicted_distance)
    return {
        "magnitude_mae": float(np.mean(np.abs(actual_magnitude - predicted_magnitude))),
        "magnitude_rmse": float(np.sqrt(np.mean((actual_magnitude - predicted_magnitude) ** 2))),
        "distance_mae_km": float(np.mean(np.abs(actual_distance - predicted_distance))),
        "distance_rmse_km": float(np.sqrt(np.mean((actual_distance - predicted_distance) ** 2))),
    }


def load(model, path: Path):
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def main() -> None:
    archive_notebook_bundle()
    report: dict[str, dict] = {}

    detector_data = STEADDataset(str(DETECTION_HDF5), str(DETECTION_CSV), "detection", max_samples=24_000)
    _, detector_val, detector_test = make_loaders(detector_data, 128, seed=42)
    detector_baseline = load(GRUDetector(), EARTHQUAKES / "gru_detector_best.pth")
    detector_candidate = load(GRUDetector(), CORRECTED / "gru_detector" / "model.pt")
    baseline_val_labels, baseline_val_probs = detector_outputs(detector_baseline, detector_val, DEVICE)
    candidate_val_labels, candidate_val_probs = detector_outputs(detector_candidate, detector_val, DEVICE)
    detector_thresholds = {
        "notebook": select_detector_threshold(baseline_val_labels, baseline_val_probs),
        "corrected": select_detector_threshold(candidate_val_labels, candidate_val_probs),
    }
    detector_scores = {
        "notebook": evaluate_detector(detector_baseline, detector_test, DEVICE, detector_thresholds["notebook"]),
        "corrected": evaluate_detector(detector_candidate, detector_test, DEVICE, detector_thresholds["corrected"]),
    }
    detector_winner = "corrected" if detector_scores["corrected"]["f1"] >= detector_scores["notebook"]["f1"] else "notebook"
    report["gru_detector"] = {"winner": detector_winner, "metrics": detector_scores, "validation_thresholds": detector_thresholds}
    detector_data.close()

    estimator_data = STEADDataset(str(EVENT_HDF5), str(EVENT_CSV), "magnitude", max_samples=16_000)
    _, _, estimator_test = make_loaders(estimator_data, 64, seed=42)
    corrected_scaler = estimator_data.target_scaler
    legacy_scaler = TargetScaler(magnitude_mean=1.0, magnitude_std=5.0, log_distance_mean=0.0, log_distance_std=10.0)
    estimator_baseline = load(LSTMEstimator(), EARTHQUAKES / "lstm_estimator_best.pth")
    estimator_candidate = load(LSTMEstimator(), CORRECTED / "lstm_estimator" / "model.pt")
    estimator_scores = {
        "notebook": evaluate_estimator_with_scalers(estimator_baseline, estimator_test, legacy_scaler, corrected_scaler),
        "corrected": evaluate_estimator_with_scalers(estimator_candidate, estimator_test, corrected_scaler, corrected_scaler),
    }
    estimator_rank = lambda metrics: metrics["magnitude_mae"] + metrics["distance_mae_km"] / 100.0
    estimator_winner = min(estimator_scores, key=lambda key: estimator_rank(estimator_scores[key]))
    report["lstm_estimator"] = {"winner": estimator_winner, "metrics": estimator_scores}
    estimator_data.close()

    forecast_data = STEADDataset(str(EVENT_HDF5), str(EVENT_CSV), "forecasting", max_samples=1_000)
    _, _, forecast_test = make_loaders(forecast_data, 8, seed=42)
    forecast_baseline = load(SeismicSeq2Seq(), EARTHQUAKES / "seq2seq_best.pth")
    forecast_candidate = load(SeismicSeq2Seq(), CORRECTED / "seq2seq" / "model.pt")
    forecast_scores = {
        "notebook": evaluate_forecaster(forecast_baseline, forecast_test, DEVICE),
        "corrected": evaluate_forecaster(forecast_candidate, forecast_test, DEVICE),
    }
    forecast_winner = "corrected" if forecast_scores["corrected"]["waveform_mae"] <= forecast_scores["notebook"]["waveform_mae"] else "notebook"
    report["seq2seq"] = {"winner": forecast_winner, "metrics": forecast_scores}
    forecast_data.close()

    sources = {
        "gru_detector": (CORRECTED / "gru_detector" / "model.pt") if detector_winner == "corrected" else (EARTHQUAKES / "gru_detector_best.pth"),
        "lstm_estimator": (CORRECTED / "lstm_estimator" / "model.pt") if estimator_winner == "corrected" else (EARTHQUAKES / "lstm_estimator_best.pth"),
        "seq2seq": (CORRECTED / "seq2seq" / "model.pt") if forecast_winner == "corrected" else (EARTHQUAKES / "seq2seq_best.pth"),
    }
    APPROVED.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, APPROVED / f"{name}.pt")
    if estimator_winner == "corrected":
        shutil.copy2(CORRECTED / "lstm_estimator" / "target_scaler.json", APPROVED / "target_scaler.json")
    else:
        legacy_scaler.save(APPROVED / "target_scaler.json")
    SignalConfig().save(APPROVED / "signal_config.json")

    metadata = {
        "version": "faultline-evaluated-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "pytorch-cpu",
        "status": "promoted-research",
        "evaluation": "identical event-disjoint test splits grouped by source_id",
        "selection_report": report,
        "detector_threshold": detector_thresholds[detector_winner],
    }
    (APPROVED / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (PROJECT / "artifacts" / "candidate_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Promoted evaluated winners to {APPROVED}")


if __name__ == "__main__":
    main()
