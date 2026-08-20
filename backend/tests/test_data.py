import numpy as np
import pandas as pd
import pytest

from seismic_lab.config import TargetScaler
from seismic_lab.data import make_loaders
from seismic_lab.evaluate import select_detector_threshold
from seismic_lab.preprocessing import ensure_time_channels, normalize_waveform


def test_channel_orientation_is_normalized():
    assert ensure_time_channels(np.zeros((3, 6000), dtype=np.float32)).shape == (6000, 3)


def test_normalization_is_channelwise():
    waveform = np.arange(30, dtype=np.float32).reshape(10, 3)
    assert np.allclose(normalize_waveform(waveform).mean(axis=0), 0, atol=1e-6)


def test_rejects_non_three_channel_signal():
    with pytest.raises(ValueError):
        ensure_time_channels(np.zeros((100, 2), dtype=np.float32))


def test_target_scaler_round_trip_uses_real_units():
    scaler = TargetScaler.fit([2.0, 3.0, 4.0], [10.0, 40.0, 100.0])
    magnitude, distance = scaler.transform(3.2, 55.0)
    restored_magnitude, restored_distance = scaler.inverse(magnitude, distance)
    assert restored_magnitude == pytest.approx(3.2)
    assert restored_distance == pytest.approx(55.0)


def test_source_ids_do_not_cross_splits():
    class MetadataOnlyDataset:
        task = "detection"
        target_scaler = None
        metadata = pd.DataFrame({"source_id": ["a", "a", "b", "c", "d", "e", "f", "g"]})

        def __len__(self):
            return len(self.metadata)

    dataset = MetadataOnlyDataset()
    loaders = make_loaders(dataset, batch_size=2, seed=7)
    split_groups = [set(dataset.metadata.iloc[loader.dataset.indices].source_id) for loader in loaders]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])


def test_detector_threshold_is_selected_from_validation_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.35, 0.40, 0.90])
    threshold = select_detector_threshold(labels, probabilities)
    predictions = probabilities >= threshold
    assert predictions.tolist() == labels.astype(bool).tolist()
