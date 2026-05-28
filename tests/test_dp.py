"""Tests for the DP-SGD primitives (specifications.md section 7)."""

from __future__ import annotations

import torch

from privacy.dp import add_gaussian_noise, clip_per_sample_gradients


def test_clip_caps_norm() -> None:
    # Three "per-sample gradients" of differing norms.
    grads = torch.stack([
        torch.tensor([3.0, 4.0]),  # norm 5.0
        torch.tensor([0.5, 0.0]),  # norm 0.5
        torch.tensor([1.0, 1.0]),  # norm sqrt(2) ≈ 1.414
    ])
    clipped = clip_per_sample_gradients(grads, max_norm=1.0)
    norms = clipped.norm(dim=1)
    assert torch.all(norms <= 1.0 + 1e-6)
    # The 0.5-norm grad should be untouched.
    torch.testing.assert_close(clipped[1], grads[1])


def test_gaussian_noise_zero_mean() -> None:
    """Averaging many noise draws should yield approximately zero mean."""
    torch.manual_seed(0)
    grad = torch.zeros(1000)
    samples = torch.stack([add_gaussian_noise(grad, noise_scale=1.0) for _ in range(100)])
    assert samples.mean().abs() < 0.05
