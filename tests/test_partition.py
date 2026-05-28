"""Tests for the MNIST partitioning utilities (specifications.md section 2).

These tests assume MNIST-like targets (10 classes, balanced). They use
a synthetic targets array rather than downloading MNIST to keep the
test suite fast and hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl.datasets.mnist_partition import dirichlet, iid, label_skew


@pytest.fixture
def synthetic_targets(tiny_dataset_size: int) -> np.ndarray:
    # 60 samples per class, 10 classes.
    targets = np.repeat(np.arange(10), tiny_dataset_size // 10)
    return targets


def test_iid_no_loss(synthetic_targets: np.ndarray) -> None:
    parts = iid(synthetic_targets.tolist(), num_clients=5, seed=0)
    flat = sorted(i for sub in parts for i in sub)
    assert flat == list(range(len(synthetic_targets)))
    assert all(abs(len(p) - len(synthetic_targets) // 5) <= 1 for p in parts)


def test_label_skew_constraint(synthetic_targets: np.ndarray) -> None:
    parts = label_skew(
        synthetic_targets.tolist(), num_clients=10, classes_per_client=2, seed=0
    )
    for indices in parts:
        client_classes = set(synthetic_targets[indices].tolist())
        assert len(client_classes) <= 2


def test_dirichlet_skew_strength(synthetic_targets: np.ndarray) -> None:
    """Lower alpha should produce more skewed client distributions."""
    parts_low = dirichlet(synthetic_targets.tolist(), num_clients=10, alpha=0.1, seed=0)
    parts_high = dirichlet(synthetic_targets.tolist(), num_clients=10, alpha=100.0, seed=0)

    def max_class_share(part):
        if not part:
            return 0.0
        labels = synthetic_targets[part]
        counts = np.bincount(labels, minlength=10)
        return counts.max() / len(part)

    avg_skew_low = float(np.mean([max_class_share(p) for p in parts_low]))
    avg_skew_high = float(np.mean([max_class_share(p) for p in parts_high]))
    assert avg_skew_low > avg_skew_high, "alpha=0.1 should be more skewed than alpha=100"
