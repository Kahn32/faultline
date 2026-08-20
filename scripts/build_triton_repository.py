"""Create an NVIDIA Triton model repository from validated ONNX artifacts."""
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts" / "approved"
REPOSITORY = PROJECT / "infra" / "triton" / "model_repository"

def install(name: str) -> None:
    source = ARTIFACTS / f"{name}.onnx"
    if not source.exists():
        raise FileNotFoundError(f"Missing validated ONNX model: {source}")
    version_dir = REPOSITORY / name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, version_dir / "model.onnx")
    external_weights = ARTIFACTS / f"{name}.onnx.data"
    if external_weights.exists():
        shutil.copy2(external_weights, version_dir / external_weights.name)

def main() -> None:
    install("gru_detector")
    install("lstm_estimator")
    print(f"Triton repository ready at {REPOSITORY}")
    print("Start on an NVIDIA host with: docker compose --profile nvidia up triton")

if __name__ == "__main__":
    main()
