"""Secure aggregation skeleton — additive secret sharing.

Pedagogical implementation of the core primitive behind Bonawitz et
al.'s SecAgg (2017): if each client splits its update into K-1 random
shares plus a final share that completes the sum, and sends each share
to a different peer, then the server can only see the sum of all
client updates, not individual contributions.

This is NOT the full SecAgg protocol:
    - No Diffie-Hellman key exchange.
    - No Shamir threshold sharing for dropout tolerance.
    - No verification against malicious clients.

It only demonstrates the *concept* — server sees the sum, individual
contributions are masked.

TODO:
    - Implement additive splitting.
    - Implement peer-side aggregation simulation.
    - Demonstrate the property via a small 3-client example in a
      notebook.
"""

from __future__ import annotations


def split_update(update, num_peers: int):
    """Split ``update`` into ``num_peers`` additive shares."""
    raise NotImplementedError("To be implemented in Phase 6.1")


def aggregate_shares(shares):
    """Sum the shares (server-side operation)."""
    raise NotImplementedError("To be implemented in Phase 6.1")
