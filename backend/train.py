"""Reproducible CPU-first training for the three original notebook models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from seismic_lab.data import STEADDataset, make_loaders
from seismic_lab.config import SignalConfig
from seismic_lab.evaluate import evaluate_detector, evaluate_estimator, evaluate_forecaster
from seismic_lab.models import GRUDetector, LSTMEstimator, SeismicSeq2Seq


MODEL_NAMES = {"detection": "gru_detector", "magnitude": "lstm_estimator", "forecasting": "seq2seq"}


def loss_for(task: str, model, batch, device: torch.device, teacher_force: float = 0.0):
    x, target = (item.to(device) for item in batch)
    if task == "detection":
        return F.cross_entropy(model(x), target, label_smoothing=0.02)
    if task == "magnitude":
        magnitude, distance, _ = model(x)
        # Huber loss is less dominated by rare, very distant events than MSE.
        return F.smooth_l1_loss(magnitude, target[:, 0], beta=0.5) + 0.5 * F.smooth_l1_loss(distance, target[:, 1], beta=0.5)
    predicted, _ = model(x, target, teacher_force=teacher_force)
    # Point loss alone encourages flat average waveforms. Temporal-gradient and
    # peak terms explicitly reward phase changes and strong-motion amplitude.
    point_loss = F.mse_loss(predicted, target)
    gradient_loss = F.l1_loss(predicted[:, 1:] - predicted[:, :-1], target[:, 1:] - target[:, :-1])
    peak_loss = F.l1_loss(predicted.abs().amax(dim=1), target.abs().amax(dim=1))
    return point_loss + 0.20 * gradient_loss + 0.05 * peak_loss


def build_model(task: str):
    return {"detection": GRUDetector(), "magnitude": LSTMEstimator(), "forecasting": SeismicSeq2Seq()}[task]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODEL_NAMES)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=12_000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = STEADDataset(args.hdf5, args.csv, args.task, max_samples=args.max_samples)
    train_loader, val_loader, test_loader = make_loaders(dataset, args.batch_size, args.seed)
    model = build_model(args.task).to(device)
    learning_rate = {"detection": 1e-3, "magnitude": 5e-4, "forecasting": 3e-4}[args.task]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    output = Path(args.output) / MODEL_NAMES[args.task]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    best_loss, stale_epochs, history = float("inf"), 0, []

    mlflow.set_experiment("faultline")
    with mlflow.start_run(run_name=f"{MODEL_NAMES[args.task]}-{device.type}"):
        mlflow.log_params(vars(args) | {"device": str(device), "parameters": sum(p.numel() for p in model.parameters())})
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            teacher_force = max(0.1, 0.9 - epoch * 0.025) if args.task == "forecasting" else 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = loss_for(args.task, model, batch, device, teacher_force)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    val_loss += loss_for(args.task, model, batch, device).item()
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "teacher_force": teacher_force}
            history.append(row)
            mlflow.log_metrics(row, step=epoch)
            print(
                f"{MODEL_NAMES[args.task]} {epoch + 1}/{args.epochs}: "
                f"train={train_loss:.5f} val={val_loss:.5f}",
                flush=True,
            )

            if val_loss < best_loss:
                best_loss, stale_epochs = val_loss, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {epoch + 1} epochs.", flush=True)
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        metrics = {
            "detection": evaluate_detector,
            "magnitude": evaluate_estimator,
            "forecasting": evaluate_forecaster,
        }[args.task](model, test_loader, device)
        mlflow.log_metrics(metrics)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        SignalConfig().save(output / "signal_config.json")
        if dataset.target_scaler is not None:
            dataset.target_scaler.save(output / "target_scaler.json")
        manifest = {
            "model_name": MODEL_NAMES[args.task],
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "task": args.task,
            "framework": "pytorch",
            "device": device.type,
            "metrics": metrics,
            "dataset": {"hdf5_path": str(Path(args.hdf5).resolve()), "csv_path": str(Path(args.csv).resolve()), "samples": len(dataset), "split": "event_disjoint_by_source_id"},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        mlflow.log_artifacts(str(output), artifact_path="model_bundle")
        print(
            json.dumps({"checkpoint": str(checkpoint_path), "test_metrics": metrics}, indent=2),
            flush=True,
        )
    dataset.close()


if __name__ == "__main__":
    main()
