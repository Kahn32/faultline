# Faultline model card

## Intended use

Faultline is a portfolio and research prototype for analyzing a supplied 60-second, three-channel seismic waveform. It detects a likely P-wave window, estimates event magnitude and source distance, and forecasts a subsequent waveform segment.

It is not an earthquake-prediction system, public alerting system, or source of emergency guidance.

## Promoted bundle

Version: `faultline-evaluated-v1`

| Task | Model selected | Held-out result |
|---|---|---:|
| Event detection | corrected bidirectional GRU | F1 0.9884; ROC-AUC 0.9983; PR-AUC 0.9985 |
| Magnitude regression | corrected attention LSTM | MAE 0.265; RMSE 0.405 |
| Distance regression | corrected attention LSTM | MAE 10.08 km; RMSE 22.17 km |
| Waveform forecasting | original notebook Seq2Seq | MAE 0.6272; RMSE 0.9966 normalized units |

The corrected Seq2Seq candidate was not promoted because its waveform MAE (0.6361) was worse than the original notebook model (0.6272) on the identical test split.

## Evaluation protocol

- Data: STEAD local-earthquake and noise records.
- Splits: deterministic 70/15/15 partitions grouped by `source_id`, preventing one seismic event from crossing train, validation, and test.
- Detector operating threshold: chosen on validation F1, then frozen for the test set and deployment.
- Regression scaling: fitted only on training targets and saved beside the model weights.
- Promotion: each corrected candidate was compared directly with its notebook checkpoint on identical held-out records.

## Deployment

- PyTorch CPU is the complete v1 runtime.
- Detector and estimator have ONNX exports with numerical parity checks.
- The Triton repository contains the ONNX models for a future NVIDIA Linux deployment profile.
- Seq2Seq stays in PyTorch because its autoregressive loop was not inaccurately presented as a complete ONNX export.

## Limitations

- Results are measured on a finite STEAD subset and may not generalize to new stations, regions, sensors, sampling rates, or rare event types.
- A strong detector score does not make this an operational warning system.
- Magnitude and distance are inferred only after the detector threshold is cleared.
- Forecast error remains substantial; forecast plots are research outputs, not ground-motion guarantees.
- The synthetic examples are interface demonstrations, not benchmark records.
