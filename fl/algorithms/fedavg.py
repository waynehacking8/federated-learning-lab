"""FedAvg aggregation (McMahan et al., 2017).

Weighted average of client states by sample count:

    w_{t+1} = Σ_k (n_k / n) * w_t^k

This module exposes one function: ``aggregate(client_states, sizes)``.

TODO:
    - Implement weighted state-dict averaging.
    - Add type hints for PyTorch state_dict-style dicts.
"""

from __future__ import annotations


def aggregate(client_states: list[dict], sample_sizes: list[int]) -> dict:
    """Weighted average of client state dicts, weighted by sample sizes."""
    raise NotImplementedError("To be implemented in Phase 1.3")
