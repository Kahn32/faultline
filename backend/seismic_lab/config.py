from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class SignalConfig:
    sample_rate_hz: int = 100
    total_samples: int = 6000
    channels: tuple[str, str, str] = ("east_west", "north_south", "vertical")
    detector_window: int = 300
    estimator_window: int = 500
    forecast_encoder_window: int = 300
    forecast_decoder_window: int = 900
    p_before: int = 50

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SignalConfig":
        values = json.loads(Path(path).read_text())
        if "channels" in values:
            values["channels"] = tuple(values["channels"])
        return cls(**values)


@dataclass
class TargetScaler:
    """Training-set-only transforms for the magnitude/distance regressor."""

    magnitude_mean: float = 0.0
    magnitude_std: float = 1.0
    log_distance_mean: float = 0.0
    log_distance_std: float = 1.0

    @classmethod
    def fit(cls, magnitudes, distances) -> "TargetScaler":
        import numpy as np

        magnitudes = np.asarray(magnitudes, dtype=np.float32)
        log_distances = np.log1p(np.asarray(distances, dtype=np.float32))
        return cls(
            magnitude_mean=float(magnitudes.mean()),
            magnitude_std=float(max(magnitudes.std(), 1e-6)),
            log_distance_mean=float(log_distances.mean()),
            log_distance_std=float(max(log_distances.std(), 1e-6)),
        )

    def transform(self, magnitude: float, distance_km: float) -> tuple[float, float]:
        import numpy as np

        return (
            (float(magnitude) - self.magnitude_mean) / self.magnitude_std,
            (float(np.log1p(distance_km)) - self.log_distance_mean) / self.log_distance_std,
        )

    def inverse(self, magnitude, log_distance):
        import numpy as np

        return (
            np.asarray(magnitude) * self.magnitude_std + self.magnitude_mean,
            np.expm1(np.asarray(log_distance) * self.log_distance_std + self.log_distance_mean),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TargetScaler":
        return cls(**json.loads(Path(path).read_text()))
