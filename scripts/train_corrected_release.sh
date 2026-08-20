#!/bin/zsh
set -u
set -o pipefail

PROJECT_DIR="/Users/kahn/Downloads/Machine_Learning/LSTM/Earthquakes/faultline"
PYTHON_BIN="/Users/kahn/Downloads/Machine_Learning/Data_Preprocessing/Feature_Eng/.venv/bin/python"
DETECTION_HDF5="/Volumes/USB321FD/seismic-signal-lab-data/detection.hdf5"
DETECTION_CSV="/Volumes/USB321FD/seismic-signal-lab-data/detection.csv"
EVENT_HDF5="/Users/kahn/Downloads/Machine_Learning/LSTM/Earthquakes/chunk2.hdf5"
EVENT_CSV="/Users/kahn/Downloads/Machine_Learning/LSTM/Earthquakes/chunk2.csv"

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=backend
export PYTHONUNBUFFERED=1
mkdir -p artifacts/logs

failed=0
echo "[Faultline] Training corrected GRU detector..."
"$PYTHON_BIN" backend/train.py detection --hdf5 "$DETECTION_HDF5" --csv "$DETECTION_CSV" --max-samples 24000 --epochs 10 --batch-size 128 --patience 4 --output artifacts/corrected 2>&1 | tee artifacts/logs/gru_detector.log || failed=1

echo "[Faultline] Training corrected LSTM estimator..."
"$PYTHON_BIN" backend/train.py magnitude --hdf5 "$EVENT_HDF5" --csv "$EVENT_CSV" --max-samples 16000 --epochs 10 --batch-size 64 --patience 4 --output artifacts/corrected 2>&1 | tee artifacts/logs/lstm_estimator.log || failed=1

echo "[Faultline] Training corrected Seq2Seq forecaster..."
"$PYTHON_BIN" backend/train.py forecasting --hdf5 "$EVENT_HDF5" --csv "$EVENT_CSV" --max-samples 1000 --epochs 5 --batch-size 8 --patience 3 --output artifacts/corrected 2>&1 | tee artifacts/logs/seq2seq.log || failed=1

if [[ "$failed" -eq 0 ]]; then
  echo "[Faultline] Comparing candidates and promoting winners..."
  "$PYTHON_BIN" scripts/evaluate_and_promote.py 2>&1 | tee artifacts/logs/evaluate_and_promote.log || failed=1
fi

if [[ "$failed" -eq 0 ]]; then
  echo "[Faultline] Rebuilding ONNX and Triton artifacts..."
  "$PYTHON_BIN" scripts/export_notebook_onnx.py 2>&1 | tee artifacts/logs/export_onnx.log || failed=1
  "$PYTHON_BIN" scripts/build_triton_repository.py 2>&1 | tee artifacts/logs/build_triton.log || failed=1
fi

echo "[Faultline] Release pipeline finished with status=$failed"
exit "$failed"
