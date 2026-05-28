"""Differential Privacy via DP-SGD (Abadi et al., 2016).

Two primitives:
    1. Per-sample gradient clipping with norm ``C``.
    2. Gaussian noise with scale ``σ * C / batch_size``.

The combined effect is that each parameter update is (ε, δ)-
differentially-private with respect to single-sample changes in the
dataset.

This module is independent of the federation algorithm; it can be
wrapped around any client's local training loop.

For production-grade ε accounting, use Opacus's RDP accountant —
the accountant in this prototype is approximate.

TODO:
    - Implement per-sample gradient clipping.
    - Implement Gaussian noise injection.
    - Add a simple (ε, δ) estimator based on standard composition
      (not RDP).
"""

from __future__ import annotations


def clip_per_sample_gradients(per_sample_grads, max_norm: float):
    """Clip each per-sample gradient to ``max_norm``."""
    raise NotImplementedError("To be implemented in Phase 5.1")


def add_gaussian_noise(aggregated_grad, noise_scale: float):
    """Add Gaussian noise N(0, σ²·I) to the aggregated gradient."""
    raise NotImplementedError("To be implemented in Phase 5.1")
