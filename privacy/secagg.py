"""Secure aggregation skeleton -- additive secret sharing.

If each client splits its update into ``num_peers`` additive shares,
sends one share to each peer, and each peer sums the shares it
received, then the server-side sum of peer-sums recovers
``sum_k update_k`` without ever revealing any individual update.

This is NOT the full SecAgg protocol -- no DH key exchange, no Shamir
threshold sharing, no malicious-client tolerance. It only demonstrates
the core additive-secret-sharing primitive.
"""

from __future__ import annotations

import torch


def split_update(update: torch.Tensor, num_peers: int, seed: int | None = None) -> list[torch.Tensor]:
    """Split ``update`` into ``num_peers`` additive shares.

    The first ``num_peers - 1`` shares are sampled from a standard
    normal; the last share is set so that the shares sum to ``update``.
    """
    if num_peers < 2:
        raise ValueError("need at least 2 peers for secret sharing")
    generator = torch.Generator(device=update.device)
    if seed is not None:
        generator.manual_seed(seed)
    shares: list[torch.Tensor] = []
    running_sum = torch.zeros_like(update)
    for _ in range(num_peers - 1):
        r = torch.randn(update.shape, generator=generator, dtype=update.dtype, device=update.device)
        shares.append(r)
        running_sum = running_sum + r
    shares.append(update - running_sum)
    return shares


def aggregate_shares(shares: list[torch.Tensor]) -> torch.Tensor:
    """Sum a list of shares (server-side operation)."""
    if not shares:
        raise ValueError("aggregate_shares() called with empty list")
    out = torch.zeros_like(shares[0])
    for s in shares:
        out = out + s
    return out


def simulate_secagg_round(updates: list[torch.Tensor], seed: int = 0) -> torch.Tensor:
    """Simulate one round of additive-secret-sharing SecAgg.

    For each client k, split update_k across N peers, route share j to
    peer j, then have each peer sum what it received. Server sums the
    N peer-sums. Returns the recovered ``sum_k update_k``.
    """
    n_clients = len(updates)
    # Routing matrix: shares[k][j] = share j of client k's update.
    all_shares: list[list[torch.Tensor]] = [
        split_update(u, num_peers=n_clients, seed=seed + k) for k, u in enumerate(updates)
    ]
    # Peer j receives shares[0][j], shares[1][j], ..., shares[K-1][j].
    peer_sums: list[torch.Tensor] = [
        aggregate_shares([all_shares[k][j] for k in range(n_clients)])
        for j in range(n_clients)
    ]
    return aggregate_shares(peer_sums)
