"""FedPer -- personalized FL by partial parameter sharing
(Arivazhagan et al., 2019).

Idea: split the network into a SHARED feature-extractor body and a
PER-CLIENT classifier head. Only the body is federated (FedAvg);
each client keeps its own head across rounds and never sends it to
the server.

    shared_{t+1} = sum_k (n_k / n) * shared_t^k     # FedAvg over body
    head_{t+1}^k = head_t^k                          # head stays local

The head names are discovered programmatically (the parameters of the
final ``nn.Linear`` in the model) rather than hard-coded, via
``head_param_names`` / ``is_shared`` -- so the split survives small
architecture changes.
"""

from __future__ import annotations

import torch
from torch import nn


def head_param_names(model: nn.Module) -> set[str]:
    """Return the state_dict keys belonging to the final Linear layer.

    The "head" is the last nn.Linear module in iteration order; every
    other parameter (conv layers + earlier Linear layers) is shared.
    """
    last_linear_prefix = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_prefix = name
    if last_linear_prefix is None:
        raise ValueError("model has no nn.Linear layer to use as a head")
    # state_dict keys for that module are "<prefix>.weight" / "<prefix>.bias".
    state_keys = set(model.state_dict().keys())
    return {k for k in state_keys if k.startswith(last_linear_prefix + ".")}


def is_shared(param_name: str, head_names: set[str]) -> bool:
    """True if a parameter is part of the shared body (federated)."""
    return param_name not in head_names


def attach_fedper(clients: list, head_names: set[str]) -> None:
    """Patch each client so it keeps its own classifier head across rounds.

    On every round the server broadcasts a state dict whose body is the
    federated average and whose head is some arbitrary client's head
    (meaningless globally). Each patched client overwrites the head keys
    with its own persisted head before training, and stashes the freshly
    trained head afterwards. The body still trains and federates normally.
    """
    for client in clients:
        client._fedper_head = None  # persisted local head, set after round 1
        _patch_fedper_client(client, head_names)


def _patch_fedper_client(client, head_names: set[str]) -> None:
    original_local_update = client.local_update

    def patched_local_update(model: nn.Module, global_state: dict):
        # Inject this client's own head (if it has one yet) over the
        # broadcast head, so local training continues from the personal head.
        state = dict(global_state)
        if client._fedper_head is not None:
            for k in head_names:
                state[k] = client._fedper_head[k].clone()
        new_state, n = original_local_update(model, state)
        # Persist the trained head for next round.
        client._fedper_head = {k: new_state[k].detach().cpu().clone() for k in head_names}
        return new_state, n

    client.local_update = patched_local_update  # type: ignore[assignment]


class FedPerAggregator:
    """FedAvg over the shared body only; head keys are passed through.

    The aggregated "global" head is meaningless under FedPer (each
    client owns its head), so for the head keys we simply keep the
    first client's tensor to produce a complete state dict. Clients
    overwrite the head with their own persisted copy in local_update.
    """

    def __init__(self, head_names: set[str]) -> None:
        self.head_names = head_names

    def aggregate(self, client_states: list[dict], sample_sizes: list[int]) -> dict:
        if not client_states:
            raise ValueError("aggregate() called with empty client_states")
        total = float(sum(sample_sizes))
        if total <= 0:
            raise ValueError("sample_sizes sum to zero")
        weights = [n / total for n in sample_sizes]

        out: dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            first = client_states[0][key]
            if key in self.head_names or not first.is_floating_point():
                # Head params and integer counters: pass through (not federated).
                out[key] = first.clone()
            else:
                acc = torch.zeros_like(first, dtype=torch.float32)
                for w, state in zip(weights, client_states):
                    acc = acc + state[key].to(torch.float32) * w
                out[key] = acc.to(first.dtype)
        return out
