"""Federated learning server / aggregator.

The server maintains the global model state, selects clients each round,
broadcasts the model, and aggregates returned updates. The aggregation
rule is plugged in by the algorithm module (FedAvg, FedProx, SCAFFOLD).

Round protocol:
    1. ``select_clients(rate)`` — sample participating clients.
    2. ``broadcast(global_state)`` — send a copy of the current model.
    3. Each client runs ``local_update`` (see ``fl.client.Client``).
    4. ``aggregate(client_updates, sample_sizes)`` — combine into the
       new global state.
    5. ``evaluate(test_loader)`` — global test metric.

TODO:
    - Implement Server class with the above hooks.
    - Decouple aggregation rule via a strategy protocol (Aggregator).
    - Persist per-round metrics to disk for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Aggregator(Protocol):
    """Aggregation strategy: combines client updates into a new global state."""

    def aggregate(self, client_updates: list, sample_sizes: list) -> dict: ...


@dataclass
class Server:
    """Federation server. To be implemented."""

    def run_round(self, round_index: int) -> dict:
        raise NotImplementedError("To be implemented in Phase 1.2")
