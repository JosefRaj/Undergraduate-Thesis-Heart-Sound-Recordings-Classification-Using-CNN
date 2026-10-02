from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sound_ml.baseline import LogisticBaseline
from heart_sound_ml.evaluate import binary_metrics, write_metrics
from heart_sound_ml.features import log_spectrum_features
from heart_sound_ml.split import stratified_group_split
from heart_sound_ml.synthetic import synthetic_recording


def main() -> None:
    sample_rate = 2000
    labels = np.array([0] * 16 + [1] * 16)
    groups = np.array([f"recording-{i:02d}" for i in range(len(labels))])
    features = np.stack(
        [log_spectrum_features(synthetic_recording(int(label), sample_rate, seed=i), sample_rate) for i, label in enumerate(labels)]
    )
    train_idx, test_idx = stratified_group_split(labels, groups, seed=17)
    model = LogisticBaseline().fit(features[train_idx], labels[train_idx])
    metrics = binary_metrics(labels[test_idx], model.predict(features[test_idx]))
    report_dir = ROOT / "reports" / "synthetic_demo"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, report_dir / "metrics.json")
    print(json.dumps(metrics, indent=2))
    print("Synthetic demo only - not a medical or CV performance result.")


if __name__ == "__main__":
    main()
