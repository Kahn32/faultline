"""Run a local end-to-end smoke check against the promoted Faultline bundle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))

from seismic_lab.inference import ModelBundle  # noqa: E402


def synthetic_event() -> np.ndarray:
    sample = np.arange(6000, dtype=np.float32)
    onset = 1550
    envelope = np.where(sample > onset, np.exp(-(sample - onset) / 720), 0.02)
    channels = [
        np.sin(sample / (18 + channel * 4)) * 0.018
        + np.sin(sample / (5 + channel)) * envelope * 0.8
        for channel in range(3)
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def main() -> None:
    approved = PROJECT / "artifacts" / "approved"
    bundle = ModelBundle(approved)
    if not bundle.ready:
        raise RuntimeError("The approved model bundle is incomplete.")

    result = bundle.analyze(synthetic_event())
    if not result["p_wave"]["detected"]:
        raise RuntimeError("The bundled example did not clear the promoted detector threshold.")
    expected_forecast_steps = min(bundle.config.forecast_decoder_window, bundle.forecast_steps)
    if np.asarray(result["forecast"]).shape != (expected_forecast_steps, 3):
        raise RuntimeError("The forecaster returned an unexpected output shape.")
    if result["estimates"] is None:
        raise RuntimeError("The event estimator was unexpectedly skipped.")

    for name in ("gru_detector", "lstm_estimator"):
        for suffix in (".onnx", ".onnx.data"):
            if not (approved / f"{name}{suffix}").exists():
                raise RuntimeError(f"Missing deployment artifact: {name}{suffix}")

    summary = {
        "status": "ok",
        "model_version": result["model_version"],
        "p_probability": result["p_wave"]["probability"],
        "magnitude": result["estimates"]["magnitude"],
        "distance_km": result["estimates"]["distance_km"],
        "forecast_shape": [expected_forecast_steps, 3],
        "latency_ms": result["latency_ms"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
