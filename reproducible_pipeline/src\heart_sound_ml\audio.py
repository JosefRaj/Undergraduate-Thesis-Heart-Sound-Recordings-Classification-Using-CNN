from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file and return normalized mono samples."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported in the draft pipeline")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    return samples, sample_rate


def normalize_peak(samples: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return samples.copy() if peak < eps else samples / peak


def fixed_segments(
    samples: np.ndarray,
    sample_rate: int,
    seconds: float = 5.0,
    overlap: float = 0.0,
    pad_last: bool = False,
) -> list[np.ndarray]:
    """Split a recording into deterministic fixed-length segments."""
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    width = int(round(sample_rate * seconds))
    step = max(1, int(round(width * (1.0 - overlap))))
    x = np.asarray(samples, dtype=np.float64)
    result = [x[start : start + width] for start in range(0, max(0, len(x) - width + 1), step)]
    consumed = (len(result) - 1) * step + width if result else 0
    if pad_last and consumed < len(x):
        tail_start = len(result) * step
        tail = x[tail_start : tail_start + width]
        result.append(np.pad(tail, (0, width - len(tail))))
    elif pad_last and not result and len(x) > 0:
        result.append(np.pad(x, (0, width - len(x))))
    return result
