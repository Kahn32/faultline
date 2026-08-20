# Faultline

Faultline is a CPU-first seismic early-warning research platform built from the original `Earthquake_LSTM.ipynb` models. It analyzes supplied three-channel waveforms with a GRU detector, attention-based LSTM estimator, and Seq2Seq waveform forecaster.

> This is a research and portfolio project. It is not emergency guidance, a public alerting system, or a system that predicts earthquakes before signals arrive.

## What it does

```text
CSV waveform → validate → P-wave scan → magnitude/distance estimate → waveform forecast
```

The upload CSV must contain exactly 6,000 samples at 100 Hz and three columns:

```csv
east_west,north_south,vertical
```

## Architecture

```text
React dashboard → FastAPI / ONNX-ready inference → Redis cache
                                              ↘ ClickHouse history
Training dataset → PyTorch models → MLflow runs → versioned artifacts
Prometheus ← API metrics
```

## Local development

1. The preserved research notebook is at [notebooks/Earthquake_LSTM_working_2026-08-18.ipynb](notebooks/Earthquake_LSTM_working_2026-08-18.ipynb). Keep using it for experiments; the package is the reproducible training/deployment path.
2. Install `backend/requirements.txt` into the same virtual environment as your notebook. The training command records MLflow runs, uses event-disjoint splits, saves a training-only regression scaler, and writes a checkpoint manifest.
3. Run the full corrected release pipeline under `caffeinate` (from this folder):

```bash
caffeinate -dimsu ./scripts/train_corrected_release.sh
```

It trains all three candidates, evaluates both notebook and corrected models on identical event-disjoint test splits, promotes the winning model for each task, and rebuilds the ONNX/Triton artifacts. The equivalent individual commands are:

```bash
# Model 1 — detector. Use merged noise + event data on the USB.
PYTHONPATH=backend python backend/train.py detection \
  --hdf5 /Volumes/USB321FD/seismic-signal-lab-data/detection.hdf5 \
  --csv /Volumes/USB321FD/seismic-signal-lab-data/detection.csv \
  --max-samples 24000 --epochs 10 --batch-size 128

# Model 2 — magnitude/distance estimator. Local earthquake-only chunk.
PYTHONPATH=backend python backend/train.py magnitude \
  --hdf5 ../chunk2.hdf5 --csv ../chunk2.csv \
  --max-samples 16000 --epochs 10 --batch-size 64

# Model 3 — deliberately modest; autoregressive forecasting is CPU-expensive.
PYTHONPATH=backend python backend/train.py forecasting \
  --hdf5 ../chunk2.hdf5 --csv ../chunk2.csv \
  --max-samples 1000 --epochs 5 --batch-size 8
```

4. The Seq2Seq forecaster remains a PyTorch model in v1: its autoregressive decode loop is recorded as such in the bundle metadata instead of being inaccurately called ONNX. It is an explicit Triton/ONNX v2 target.

### NVIDIA Triton profile

The detector and estimator can be exported to a Triton model repository after ONNX parity passes:

```bash
python scripts/export_notebook_onnx.py
python scripts/build_triton_repository.py
docker compose --profile nvidia up triton
```

This profile targets an NVIDIA Linux/CUDA host. The normal Compose stack remains CPU-first on macOS; Triton is serving infrastructure, not a training dependency. The repository uses Triton's required `model-name/1/model.onnx` layout.

5. The API deliberately returns `503` until `artifacts/approved/` exists with all real artifacts.
6. Run `docker compose up --build`.
7. Open the dashboard on `http://localhost:3000`.

## CPU API deployment

`Dockerfile.api` is the standalone inference image and `render.yaml` describes a free Render web service with `/health` checks. The free service can sleep after inactivity, so the first request after a quiet period may be slow. Set the public frontend's `NEXT_PUBLIC_API_URL` to the resulting `https://...onrender.com` URL before publishing.

## Detection dataset

Create a single balanced detection source from the original noise and local-earthquake chunks without loading waveforms into memory:

```bash
python scripts/merge_detection_dataset.py \
  --hdf5 ../chunk1.hdf5 /Volumes/USB321FD/chunk2/chunk2.hdf5 \
  --csv ../chunk1.csv /Volumes/USB321FD/chunk2/chunk2.csv \
  --output-dir /Volumes/USB321FD/seismic-signal-lab-data
```

The script creates `detection.hdf5` and `detection.csv`, verifies trace and metadata counts, and refuses to overwrite existing outputs.

## Model claims and evaluation

The project reports detector F1/ROC-AUC/PR-AUC, regression MAE/RMSE/R² in real units, and forecast MAE/RMSE. It should only be described as an early-warning **research prototype**. CUDA, ROCm, Kafka, and NVIDIA Triton are documented v2 extensions after measured implementation.

The detector threshold is selected only from validation predictions. If an upload does not clear that threshold, the API intentionally skips magnitude, distance, and waveform forecasting rather than displaying event estimates for noise.

### Promoted release (`faultline-evaluated-v1`)

Every comparison below used the same deterministic test records from event-disjoint splits grouped by STEAD `source_id`.

| Model | Promoted source | Notebook baseline | Corrected candidate |
|---|---|---:|---:|
| GRU detector | corrected | F1 0.9698 | **F1 0.9884** |
| LSTM magnitude | corrected | MAE 0.411 | **MAE 0.265** |
| LSTM distance | corrected | MAE 15.27 km | **MAE 10.08 km** |
| Seq2Seq forecast | notebook | **MAE 0.6272** | MAE 0.6361 |

The selection report is saved in `artifacts/candidate_comparison.json`; the promoted model provenance and detector threshold are saved in `artifacts/approved/model_metadata.json`. See [docs/model-card.md](docs/model-card.md) for scope and limitations.

## Resume summary

Built Faultline, a containerized seismic ML platform that evaluates and promotes GRU, attention-LSTM, and Seq2Seq models on event-disjoint STEAD splits; improved detector F1 from 0.970 to 0.988 and distance MAE from 15.27 km to 10.08 km, exported parity-checked ONNX models to an NVIDIA Triton repository, and served CPU inference through FastAPI with Redis, ClickHouse, Prometheus, and a waveform-first React interface.
