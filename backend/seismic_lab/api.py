from __future__ import annotations

import hashlib
import csv
import io
import os
import asyncio
from pathlib import Path
import time
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.concurrency import run_in_threadpool

from .config import SignalConfig
from .inference import ModelBundle
from .persistence import AnalysisStore
from .preprocessing import ensure_time_channels

ARTIFACT_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))
bundle = ModelBundle(ARTIFACT_DIR)
config = SignalConfig()
store = AnalysisStore()
ANALYSES = Counter("seismic_analyses_total", "Completed waveform analyses", ["cache"])
LATENCY = Histogram("seismic_analysis_seconds", "End-to-end waveform analysis latency")
INFERENCE_SLOTS = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCE", "2")))

app = FastAPI(title="Faultline API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=False)
app.mount("/metrics", make_asgi_app())


def parse_csv(payload: bytes) -> np.ndarray:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        missing = set(config.channels) - set(reader.fieldnames or ())
        if missing:
            raise HTTPException(422, f"CSV must include these channel columns: {', '.join(config.channels)}.")
        rows = [
            [float(row[channel]) for channel in config.channels]
            for row in reader
        ]
        waveform = np.asarray(rows, dtype=np.float32)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read CSV: {exc}") from exc
    if waveform.shape != (config.total_samples, len(config.channels)):
        raise HTTPException(422, f"CSV must contain exactly {config.total_samples} rows and three channel columns; received {waveform.shape[0]} rows.")
    if not np.isfinite(waveform).all():
        raise HTTPException(422, "CSV contains missing or non-numeric waveform values.")
    return ensure_time_channels(waveform)


@app.get("/health")
def health() -> dict:
    return {"status": "ready" if bundle.ready else "model_artifacts_missing", "model_version": bundle.metadata.get("version", "untrained"), "runtime": "pytorch-cpu", "services": store.status}


@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        raise HTTPException(415, "Upload a CSV waveform file.")
    if not bundle.ready:
        raise HTTPException(503, "Models are not trained/exported yet. Run the training pipeline first.")
    payload = await file.read(5_000_001)
    if len(payload) > 5_000_000:
        raise HTTPException(413, "CSV upload is too large; maximum size is 5 MB.")
    key = hashlib.sha256(payload).hexdigest()
    cached = store.get(key)
    if cached is not None:
        ANALYSES.labels(cache="hit").inc()
        return {**cached, "cache": "hit"}
    waveform = parse_csv(payload)
    async with INFERENCE_SLOTS:
        with LATENCY.time():
            response = await run_in_threadpool(bundle.analyze, waveform)
    store.save(key, response, file.filename or "waveform.csv")
    ANALYSES.labels(cache="miss").inc()
    return {**response, "cache": "miss"}
