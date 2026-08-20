"""Export the approved PyTorch model bundle and validate ONNX parity."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import numpy as np
import onnxruntime as ort
import torch

from .models import GRUDetector, LSTMEstimator, SeismicSeq2Seq


def export_bundle(checkpoint_dir: str | Path, artifact_dir: str | Path, version: str) -> None:
    """Build one deployable bundle from the three training artifact directories.

    GRU and LSTM are exported and parity-tested through ONNX Runtime. The
    autoregressive Seq2Seq stays PyTorch in v1 because its Python decoding loop
    needs a separate fixed-shape ONNX decoder graph; this is recorded in metadata
    rather than being silently claimed as ONNX.
    """
    checkpoint_dir, artifact_dir = Path(checkpoint_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    definitions = [("gru_detector", GRUDetector(), (1, 300, 3)), ("lstm_estimator", LSTMEstimator(), (1, 500, 3))]
    manifests = {}
    for name, model, shape in definitions:
        source = checkpoint_dir / name
        model.load_state_dict(torch.load(source / "model.pt", map_location="cpu", weights_only=True)); model.eval()
        sample = torch.randn(*shape)
        torch.save(model.state_dict(), artifact_dir / f"{name}.pt")
        output_names = ["logits"] if name == "gru_detector" else ["magnitude", "distance", "attention"]
        torch.onnx.export(
            model, sample, artifact_dir / f"{name}.onnx",
            input_names=["waveform"], output_names=output_names,
            dynamic_axes={"waveform": {0: "batch"}, **{output: {0: "batch"} for output in output_names}},
            dynamo=True,
        )
        model_output = model(sample)
        torch_out = model_output[0] if isinstance(model_output, tuple) else model_output
        onnx_out = ort.InferenceSession(str(artifact_dir / f"{name}.onnx")).run(None, {"waveform": sample.numpy()})[0]
        if not np.allclose(torch_out.detach().numpy(), onnx_out, rtol=1e-3, atol=1e-4):
            raise RuntimeError(f"ONNX parity failed for {name}.")
        manifests[name] = json.loads((source / "manifest.json").read_text())

    seq_source = checkpoint_dir / "seq2seq"
    forecaster = SeismicSeq2Seq()
    forecaster.load_state_dict(torch.load(seq_source / "model.pt", map_location="cpu", weights_only=True))
    torch.save(forecaster.state_dict(), artifact_dir / "seq2seq.pt")
    manifests["seq2seq"] = json.loads((seq_source / "manifest.json").read_text())
    scaler = checkpoint_dir / "lstm_estimator" / "target_scaler.json"
    if not scaler.exists():
        raise FileNotFoundError("The estimator bundle is missing target_scaler.json.")
    shutil.copy2(scaler, artifact_dir / "target_scaler.json")
    (artifact_dir / "model_metadata.json").write_text(json.dumps({
        "version": version,
        "runtime": "pytorch-cpu + onnxruntime",
        "onnx_validated_models": ["gru_detector", "lstm_estimator"],
        "pytorch_only_models": ["seq2seq"],
        "training_manifests": manifests,
    }, indent=2))
