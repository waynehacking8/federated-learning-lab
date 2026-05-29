"""FedAvg aggregation (McMahan et al., 2017).

Weighted average of client state dicts by sample count:

    w_{t+1} = sum_k (n_k / n) * w_t^k

Average only floating-point tensors; integer tensors (e.g. BatchNorm
``num_batches_tracked``) are passed through using the first client's
value because averaging integer counters is meaningless.
"""

from __future__ import annotations

import torch


class FedAvgAggregator:
    """Stateless aggregator object satisfying the Aggregator protocol."""

    def aggregate(
        self,
        client_states: list[dict],
        sample_sizes: list[int],
    ) -> dict:
        return aggregate(client_states, sample_sizes)


def aggregate(client_states: list[dict], sample_sizes: list[int]) -> dict:
    """Weighted average of client state dicts, weighted by sample sizes."""
    if not client_states:
        raise ValueError("aggregate() called with empty client_states")
    if len(client_states) != len(sample_sizes):
        raise ValueError(
            f"client_states ({len(client_states)}) and sample_sizes "
            f"({len(sample_sizes)}) length mismatch"
        )

    total = float(sum(sample_sizes))
    if total <= 0:
        raise ValueError("sample_sizes sum to zero")
    weights = [n / total for n in sample_sizes]

    reference_keys = client_states[0].keys()
    for state in client_states[1:]:
        if state.keys() != reference_keys:
            raise ValueError("client state_dicts have mismatched keys")

    out: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        first = client_states[0][key]
        if first.is_floating_point():
            acc = torch.zeros_like(first, dtype=torch.float32)
            for w, state in zip(weights, client_states):
                acc = acc + state[key].to(torch.float32) * w
            out[key] = acc.to(first.dtype)
        else:
            # Integer tensors (e.g. BN num_batches_tracked) are not averaged.
            out[key] = first.clone()
    return out
