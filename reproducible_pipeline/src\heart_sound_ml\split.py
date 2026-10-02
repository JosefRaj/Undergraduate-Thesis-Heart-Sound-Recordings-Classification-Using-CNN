from __future__ import annotations

import numpy as np


def stratified_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split group identifiers while approximately preserving both classes."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    train_groups: list[object] = []
    test_groups: list[object] = []
    for cls in np.unique(labels):
        cls_groups = np.unique(groups[labels == cls])
        rng.shuffle(cls_groups)
        n_test = max(1, int(round(len(cls_groups) * test_fraction))) if len(cls_groups) > 1 else 0
        test_groups.extend(cls_groups[:n_test].tolist())
        train_groups.extend(cls_groups[n_test:].tolist())
    test_mask = np.isin(groups, np.asarray(test_groups, dtype=groups.dtype))
    train_mask = ~test_mask
    if np.any(np.intersect1d(groups[train_mask], groups[test_mask])):
        raise RuntimeError("group leakage detected")
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)
