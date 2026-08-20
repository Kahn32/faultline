from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import torch

from .config import SignalConfig, TargetScaler
from .models import GRUDetector, LSTMEstimator, SeismicSeq2Seq
from .preprocessing import normalize_waveform


@dataclass
class ModelBundle:
    artifact_dir: Path
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        config_path = self.artifact_dir / "signal_config.json"
        self.config = SignalConfig.load(config_path) if config_path.exists() else SignalConfig()
        metadata_path = self.artifact_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"version": "untrained"}
        self.detector_threshold = float(self.metadata.get("detector_threshold", 0.5))
        scaler_path = self.artifact_dir / "target_scaler.json"
        self.target_scaler = TargetScaler.load(scaler_path) if scaler_path.exists() else None
        self.detector = GRUDetector().to(self.device)
        self.estimator = LSTMEstimator().to(self.device)
        self.forecaster = SeismicSeq2Seq().to(self.device)
        self.ready = self.target_scaler is not None and all((self.artifact_dir / name).exists() for name in ("gru_detector.pt", "lstm_estimator.pt", "seq2seq.pt"))
        if self.ready:
            self.detector.load_state_dict(torch.load(self.artifact_dir / "gru_detector.pt", map_location=self.device, weights_only=True))
            self.estimator.load_state_dict(torch.load(self.artifact_dir / "lstm_estimator.pt", map_location=self.device, weights_only=True))
            self.forecaster.load_state_dict(torch.load(self.artifact_dir / "seq2seq.pt", map_location=self.device, weights_only=True))
            self.detector.eval(); self.estimator.eval(); self.forecaster.eval()

    @torch.inference_mode()
    def analyze(self, waveform: np.ndarray) -> dict:
        if not self.ready:
            raise RuntimeError("No trained model bundle is available. Train and export the models before analysis.")
        started = time.perf_counter()
        c = self.config
        # Match training exactly: each detector candidate is normalized over its
        # own 300-sample window. Keep enough signal after every candidate for
        # both the estimator and the 300 -> 900 sample forecaster.
        required_after_detection = max(
            c.estimator_window,
            c.forecast_encoder_window + c.forecast_decoder_window,
        )
        last_start = len(waveform) - required_after_detection
        starts = list(range(0, last_start + 1, 25))
        detector_windows = torch.tensor(
            np.stack(
                [normalize_waveform(waveform[s : s + c.detector_window]) for s in starts]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        probabilities = torch.softmax(self.detector(detector_windows), dim=1)[:, 1]
        best_idx = int(probabilities.argmax().item())
        p_start = starts[best_idx]
        probability = float(probabilities[best_idx])
        detected = probability >= self.detector_threshold
        p_wave = {
            "start_sample": p_start if detected else None,
            "arrival_seconds": round((p_start + c.p_before) / c.sample_rate_hz, 3) if detected else None,
            "probability": round(probability, 5),
            "threshold": self.detector_threshold,
            "detected": detected,
        }
        if not detected:
            return {
                "model_version": self.metadata.get("version", "local"),
                "p_wave": p_wave,
                "estimates": None,
                "forecast": [],
                "attention": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "runtime": "pytorch-cpu",
            }
        estimator_window = torch.tensor(
            normalize_waveform(waveform[p_start : p_start + c.estimator_window]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mag_norm, distance_norm, importance = self.estimator(estimator_window)
        forecast_segment = normalize_waveform(
            waveform[
                p_start : p_start
                + c.forecast_encoder_window
                + c.forecast_decoder_window
            ]
        )
        encoder = torch.tensor(
            forecast_segment[: c.forecast_encoder_window],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        forecast, attention = self.forecaster.predict(encoder, steps=c.forecast_decoder_window)
        elapsed_ms = (time.perf_counter() - started) * 1000
        magnitude, distance_km = self.target_scaler.inverse(
            mag_norm.detach().cpu().numpy(), distance_norm.detach().cpu().numpy()
        )
        return {
            "model_version": self.metadata.get("version", "local"),
            "p_wave": p_wave,
            "estimates": {"magnitude": round(float(magnitude.item()), 3), "distance_km": round(float(distance_km.item()), 3)},
            "forecast": forecast.squeeze(0).cpu().numpy().round(6).tolist(),
            "attention": attention.squeeze(0).cpu().numpy().mean(axis=0).round(6).tolist(),
            "latency_ms": round(elapsed_ms, 2),
            "runtime": "pytorch-cpu",
        }
