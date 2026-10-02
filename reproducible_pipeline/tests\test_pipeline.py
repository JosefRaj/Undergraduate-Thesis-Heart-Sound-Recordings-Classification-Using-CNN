import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sound_ml.audio import fixed_segments, normalize_peak
from heart_sound_ml.baseline import LogisticBaseline
from heart_sound_ml.evaluate import binary_metrics
from heart_sound_ml.features import log_spectrum_features
from heart_sound_ml.split import stratified_group_split
from heart_sound_ml.synthetic import synthetic_recording


class PipelineTests(unittest.TestCase):
    def test_segmentation_and_padding(self):
        x = np.arange(13, dtype=float)
        segments = fixed_segments(x, sample_rate=2, seconds=2, pad_last=True)
        self.assertEqual([len(s) for s in segments], [4, 4, 4, 4])

    def test_normalization_is_bounded(self):
        x = normalize_peak(np.array([-2.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(np.max(np.abs(x))), 1.0)

    def test_features_are_finite_and_fixed_width(self):
        x = synthetic_recording(0, seed=1)
        feat = log_spectrum_features(x, 2000, bands=20)
        self.assertEqual(feat.shape, (44,))
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_group_split_has_no_leakage(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        groups = np.array(["a", "a", "b", "c", "d", "d", "e", "f"])
        train, test = stratified_group_split(labels, groups, seed=3)
        self.assertEqual(len(set(groups[train]) & set(groups[test])), 0)

    def test_baseline_learns_separable_synthetic_data(self):
        labels = np.array([0] * 12 + [1] * 12)
        groups = np.array([f"g{i}" for i in range(24)])
        x = np.stack(
            [log_spectrum_features(synthetic_recording(int(y), seed=i), 2000) for i, y in enumerate(labels)]
        )
        train, test = stratified_group_split(labels, groups, seed=5)
        model = LogisticBaseline().fit(x[train], labels[train])
        metrics = binary_metrics(labels[test], model.predict(x[test]))
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
