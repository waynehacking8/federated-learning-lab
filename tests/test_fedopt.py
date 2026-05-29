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


def test_unknown_optimizer_raises() -> None:
    try:
        FedOptAggregator(_init(), optimizer="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
