"""Import the successful notebook checkpoints into a deployable research bundle."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent
OUTPUT = PROJECT / "artifacts" / "approved"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "gru_detector_best.pth": "gru_detector.pt",
        "lstm_estimator_best.pth": "lstm_estimator.pt",
        "seq2seq_best.pth": "seq2seq.pt",
    }
    for source_name, output_name in checkpoints.items():
        source = SOURCE / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        # Loading first catches incomplete/corrupt checkpoints before copying.
        torch.load(source, map_location="cpu", weights_only=True)
        shutil.copy2(source, OUTPUT / output_name)

    # Exact transforms used by the successful notebook checkpoint:
    # magnitude=(m-1)/5 and distance=log1p(km)/10.
    (OUTPUT / "target_scaler.json").write_text(json.dumps({
        "magnitude_mean": 1.0,
        "magnitude_std": 5.0,
        "log_distance_mean": 0.0,
        "log_distance_std": 10.0,
    }, indent=2))
    (OUTPUT / "model_metadata.json").write_text(json.dumps({
        "version": "notebook-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "pytorch-cpu",
        "status": "research-candidate",
        "split_warning": "Notebook metrics used random trace splits; corrected event-disjoint retraining is pending.",
        "reported_metrics": {
            "detector_roc_auc": 0.9946,
            "detector_average_precision": 0.9947,
            "magnitude_mae": 0.115,
            "distance_mae_km": 14.2,
            "seq2seq": "500-sample, 2-epoch prototype",
        },
    }, indent=2))
    print(f"Imported notebook bundle to {OUTPUT}")


if __name__ == "__main__":
    main()
