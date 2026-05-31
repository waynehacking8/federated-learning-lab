"""Test SCAFFOLD global step size eta_g (specifications.md section 6, D15).

eta_g scales the server-side model update: x <- x + eta_g * mean(delta_w)
(Karimireddy 2020 Algorithm 1). eta_g=1 is the paper default; eta_g<1
dampens the stationary-distribution oscillation under severe skew.
"""

from __future__ import annotations

import torch

from fl.algorithms.scaffold import ScaffoldAggregator


class _FakeClient:
    """Minimal stand-in carrying a SCAFFOLD round-output buffer."""

    def __init__(self, delta_w, delta_c, w_global, n):
        self._scaffold_round_outputs = {
            "delta_w": delta_w,
            "delta_c": delta_c,
            "n_samples": n,
            "w_global_broadcast": w_global,
        }
        self._scaffold_c_global_ref = None


def _setup(global_lr):
    w_global = {"w": torch.zeros(4)}
    # Two clients, each proposing delta_w = +1 on every coordinate.
    clients = [
        _FakeClient({"w": torch.ones(4)}, {"w": torch.zeros(4)}, {"w": torch.zeros(4)}, 10),
        _FakeClient({"w": torch.ones(4)}, {"w": torch.zeros(4)}, {"w": torch.zeros(4)}, 10),
    ]
    agg = ScaffoldAggregator(clients=clients, global_lr=global_lr)
    return agg, clients


def test_global_lr_one_is_full_step() -> None:
    agg, clients = _setup(global_lr=1.0)
    out = agg.aggregate([c._scaffold_round_outputs["delta_w"] for c in clients], [10, 10])
    # mean delta_w = +1, eta_g=1 -> new w = 0 + 1*1 = 1.
    torch.testing.assert_close(out["w"], torch.ones(4))


def test_global_lr_half_is_damped_step() -> None:
    agg, clients = _setup(global_lr=0.5)
    out = agg.aggregate([c._scaffold_round_outputs["delta_w"] for c in clients], [10, 10])
    # mean delta_w = +1, eta_g=0.5 -> new w = 0 + 0.5*1 = 0.5.
    torch.testing.assert_close(out["w"], torch.full((4,), 0.5))


def test_default_global_lr_is_one() -> None:
    agg, _ = _setup(global_lr=1.0)
    assert agg.global_lr == 1.0
