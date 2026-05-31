"""Per-sample clipping correctness (external review finding #3).

The (eps, delta) guarantee is only valid if clipping is genuinely
per-sample -- never on a microbatch mean. This test pins that: the
library clip must equal a manual per-row clip, and every clipped row
must have L2 norm <= C.
"""

from __future__ import annotations

import torch

from privacy.dp import clip_per_sample_gradients


def test_clip_matches_manual_per_sample() -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 50)
    C = 1.0
    lib = clip_per_sample_gradients(g, C)
    manual = torch.stack([row * min(1.0, C / (row.norm().item() + 1e-12)) for row in g])
    torch.testing.assert_close(lib, manual)


def test_clipped_rows_respect_norm_bound() -> None:
    torch.manual_seed(1)
    g = torch.randn(16, 30) * 5.0  # large norms so clipping actually bites
    C = 0.7
    out = clip_per_sample_gradients(g, C)
    assert bool((out.norm(dim=1) <= C + 1e-5).all())


def test_small_grads_unchanged() -> None:
    g = torch.full((4, 10), 0.01)
    out = clip_per_sample_gradients(g, max_norm=1.0)
    torch.testing.assert_close(out, g)  # norm < C, no scaling
