from __future__ import annotations

import numpy as np


def synthetic_recording(label: int, sample_rate: int = 2000, seconds: float = 5.0, seed: int = 0) -> np.ndarray:
    """Generate a deterministic heart-like signal for pipeline tests only."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    heart_rate_hz = 1.2 + 0.15 * label
    envelope = np.maximum(0.0, np.sin(2 * np.pi * heart_rate_hz * t)) ** 8
    carrier = np.sin(2 * np.pi * (55 + 25 * label) * t)
    harmonic = 0.35 * label * np.sin(2 * np.pi * 170 * t)
    noise = rng.normal(0.0, 0.035 + 0.01 * label, size=len(t))
    return 0.75 * envelope * carrier + harmonic + noise
