"""Export and parity-check the approved notebook GRU/LSTM for ONNX/Triton."""
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
from seismic_lab.models import GRUDetector, LSTMEstimator  # noqa: E402

ARTIFACTS = PROJECT / "artifacts" / "approved"

def export(name: str, model, shape: tuple[int, ...], output_names: list[str]) -> None:
    model.load_state_dict(torch.load(ARTIFACTS / f"{name}.pt", map_location="cpu", weights_only=True))
    model.eval()
    sample = torch.randn(*shape)
    output_path = ARTIFACTS / f"{name}.onnx"
    torch.onnx.export(
        model, sample, output_path, input_names=["waveform"], output_names=output_names,
        dynamic_axes={"waveform": {0: "batch"}, **{item: {0: "batch"} for item in output_names}},
        opset_version=18,
    )
    pytorch_outputs = model(sample)
    if not isinstance(pytorch_outputs, tuple):
        pytorch_outputs = (pytorch_outputs,)
    onnx_outputs = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"]).run(None, {"waveform": sample.numpy()})
    for expected, actual in zip(pytorch_outputs, onnx_outputs):
        if not np.allclose(expected.detach().numpy(), actual, rtol=1e-3, atol=1e-4):
            raise RuntimeError(f"ONNX parity failed for {name}")
    print(f"{name}: ONNX parity passed")

def main() -> None:
    export("gru_detector", GRUDetector(), (1, 300, 3), ["logits"])
    export("lstm_estimator", LSTMEstimator(), (1, 500, 3), ["magnitude", "distance", "attention"])

if __name__ == "__main__":
    main()
