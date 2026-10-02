from __future__ import annotations

import numpy as np

from .audio import normalize_peak


def _frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    starts = range(0, len(samples) - frame_length + 1, hop_length)
    return np.stack([samples[s : s + frame_length] for s in starts])


def log_spectrum_features(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 64.0,
    hop_ms: float = 32.0,
    bands: int = 24,
) -> np.ndarray:
    """Return compact time-frequency summary features using only NumPy.

    The vector contains per-band log-power mean and standard deviation plus
    basic waveform statistics. It is intentionally transparent for a baseline.
    """
    if sample_rate <= 0 or bands < 2:
        raise ValueError("sample_rate must be positive and bands >= 2")
    x = normalize_peak(np.asarray(samples, dtype=np.float64))
    frame_length = max(32, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frames = _frame_signal(x, frame_length, hop_length)
    window = np.hanning(frame_length)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    edges = np.linspace(0, power.shape[1], bands + 1, dtype=int)
    band_power = np.stack(
        [power[:, edges[i] : max(edges[i] + 1, edges[i + 1])].mean(axis=1) for i in range(bands)],
        axis=1,
    )
    log_power = np.log1p(band_power)
    zero_crossing = float(np.mean(x[:-1] * x[1:] < 0)) if len(x) > 1 else 0.0
    waveform = np.array([np.mean(x), np.std(x), np.max(np.abs(x)), zero_crossing])
    return np.concatenate([log_power.mean(axis=0), log_power.std(axis=0), waveform])
