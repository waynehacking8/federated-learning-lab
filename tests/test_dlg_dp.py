"""Regression tests for the DP post-processing inside the DLG demo.

The original implementation nested the noise branch inside the clipping
branch, so dp_sigma > 0 with dp_clip_C=None silently returned the raw
gradient -- the demo would claim noise was applied when it was not.
"""

from __future__ import annotations

import pytest
import torch

from privacy.dlg import _apply_dp, _flat_grad


def _grads():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(4, 3, generator=g), torch.randn(7, generator=g)]


def test_noop_when_no_dp_configured() -> None:
    grads = _grads()
    out = _apply_dp(grads, clip_C=None, sigma=0.0, gen=torch.Generator())
    for a, b in zip(out, grads):
        torch.testing.assert_close(a, b)


def test_clip_bounds_norm() -> None:
    grads = _grads()
    out = _apply_dp(grads, clip_C=0.5, sigma=0.0, gen=torch.Generator())
    assert _flat_grad(out).norm().item() <= 0.5 + 1e-5


def test_noise_is_actually_applied() -> None:
    grads = _grads()
    gen = torch.Generator().manual_seed(1)
    out = _apply_dp(grads, clip_C=1.0, sigma=1.0, gen=gen)
    clipped = _apply_dp(grads, clip_C=1.0, sigma=0.0, gen=torch.Generator())
    diff = (_flat_grad(out) - _flat_grad(clipped)).norm().item()
    assert diff > 0.1, "sigma=1 noise must visibly perturb the gradient"


def test_sigma_without_clip_raises() -> None:
    with pytest.raises(ValueError):
        _apply_dp(_grads(), clip_C=None, sigma=1.0, gen=torch.Generator())
