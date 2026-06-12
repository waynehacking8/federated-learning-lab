"""Additive-secret-sharing invariants for privacy/secagg.py."""

from __future__ import annotations

import pytest
import torch

from privacy.secagg import aggregate_shares, simulate_secagg_round, split_update


def test_shares_sum_to_update() -> None:
    update = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    shares = split_update(update, num_peers=5, seed=0)
    assert len(shares) == 5
    torch.testing.assert_close(aggregate_shares(shares), update, atol=1e-5, rtol=1e-5)


def test_individual_share_is_not_the_update() -> None:
    update = torch.ones(100)
    shares = split_update(update, num_peers=4, seed=0)
    for s in shares:
        assert not torch.allclose(s, update, atol=1e-2), (
            "an individual share should not reveal the update"
        )


def test_round_recovers_sum_of_updates() -> None:
    g = torch.Generator().manual_seed(3)
    updates = [torch.randn(46, generator=g) for _ in range(6)]
    recovered = simulate_secagg_round(updates, seed=0)
    torch.testing.assert_close(
        recovered, torch.stack(updates).sum(dim=0), atol=1e-4, rtol=1e-5
    )


def test_fewer_than_two_peers_raises() -> None:
    with pytest.raises(ValueError):
        split_update(torch.ones(3), num_peers=1)
