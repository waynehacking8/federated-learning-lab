"""Tests for FedAvg aggregation (specifications.md section 4)."""

from __future__ import annotations

import torch

from fl.algorithms.fedavg import aggregate


def test_equal_weight_average() -> None:
    """Two identical-size clients should produce the elementwise mean."""
    sd1 = {"w": torch.zeros(3), "b": torch.ones(2)}
    sd2 = {"w": torch.ones(3) * 4, "b": torch.ones(2) * -2}
    aggregated = aggregate([sd1, sd2], sample_sizes=[100, 100])
    torch.testing.assert_close(aggregated["w"], torch.full((3,), 2.0))
    torch.testing.assert_close(aggregated["b"], torch.full((2,), -0.5))


def test_weighted_average_by_sample_size() -> None:
    """A client with 3x the samples gets 3x the weight."""
    sd1 = {"w": torch.zeros(2)}
    sd2 = {"w": torch.ones(2) * 4}
    aggregated = aggregate([sd1, sd2], sample_sizes=[100, 300])
    # Expected: (0*1 + 4*3) / 4 = 3
    torch.testing.assert_close(aggregated["w"], torch.full((2,), 3.0))


def test_skip_integer_tensors() -> None:
    """Integer-typed tensors (e.g., BN num_batches_tracked) should pass through unchanged."""
    sd1 = {"w": torch.ones(2), "n": torch.tensor(5, dtype=torch.long)}
    sd2 = {"w": torch.ones(2) * 3, "n": torch.tensor(7, dtype=torch.long)}
    aggregated = aggregate([sd1, sd2], sample_sizes=[1, 1])
    torch.testing.assert_close(aggregated["w"], torch.full((2,), 2.0))
    # Integer entries: implementation-defined, but must not crash.
    assert "n" in aggregated
