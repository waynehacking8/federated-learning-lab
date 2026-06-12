"""Tests for FedOpt server-side adaptive optimizer (specifications.md section 13)."""

from __future__ import annotations

import torch

from fl.algorithms.fedopt import FedOptAggregator


def _init():
    return {"w": torch.zeros(5), "b": torch.zeros(3), "n": torch.tensor(0)}


def test_moves_toward_client_consensus() -> None:
    """If all clients agree on a direction, the server should move that way."""
    init = _init()
    agg = FedOptAggregator(init, optimizer="adam", server_lr=0.1)
    # Clients all moved +1 on w.
    c = {"w": torch.ones(5), "b": torch.zeros(3), "n": torch.tensor(0)}
    out = agg.aggregate([c, c], [1, 1])
    assert (out["w"] > 0).all(), "server should step toward the client consensus"


def test_integer_key_passthrough() -> None:
    init = _init()
    agg = FedOptAggregator(init, optimizer="adam")
    c = {"w": torch.ones(5), "b": torch.zeros(3), "n": torch.tensor(9)}
    out = agg.aggregate([c], [1])
    assert "n" in out


def test_output_matches_internal_state() -> None:
    init = _init()
    agg = FedOptAggregator(init, optimizer="yogi", server_lr=0.05)
    c = {"w": torch.ones(5) * 0.5, "b": torch.ones(3), "n": torch.tensor(0)}
    out = agg.aggregate([c], [1])
    torch.testing.assert_close(out["w"].float(), agg.w["w"])
    torch.testing.assert_close(out["b"].float(), agg.w["b"])


def test_round1_update_matches_reddi_2021() -> None:
    """Hand-check the first FedAdam step against Algorithm 2 of
    Reddi et al. 2021: v_0 = tau^2, NO bias correction."""
    import math

    init = {"w": torch.zeros(2)}
    lr, b1, b2, tau = 0.1, 0.9, 0.99, 1e-3
    agg = FedOptAggregator(init, optimizer="adam", server_lr=lr, beta1=b1, beta2=b2, tau=tau)
    out = agg.aggregate([{"w": torch.ones(2)}], [1])

    g = -1.0  # pseudo-gradient of a +1 client delta
    m1 = (1 - b1) * g
    v1 = b2 * tau**2 + (1 - b2) * g * g
    expected = 0.0 - lr * m1 / (math.sqrt(v1) + tau)
    torch.testing.assert_close(out["w"], torch.full((2,), expected))


def test_adagrad_accumulator_is_running_sum() -> None:
    """FedAdagrad's v is a running sum (no beta2 decay, no bias correction)."""
    init = {"w": torch.zeros(1)}
    agg = FedOptAggregator(init, optimizer="adagrad", server_lr=0.1, tau=1e-3)
    agg.aggregate([{"w": torch.ones(1)}], [1])
    v_after_1 = agg.v["w"].clone()
    torch.testing.assert_close(v_after_1, torch.tensor([1e-6 + 1.0]))


def test_unknown_optimizer_raises() -> None:
    try:
        FedOptAggregator(_init(), optimizer="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
