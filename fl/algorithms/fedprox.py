"""FedProx -- proximal-term-augmented local loss (Li et al., 2018).

Local objective:

    L_k(w) = F_k(w) + (mu / 2) * || w - w_global ||^2

Aggregation is identical to FedAvg; the proximal term lives on the
client side. Rather than recomputing it inside the loss, we add the
proximal *gradient*

    grad_prox = mu * (w_local - w_global)

directly to ``p.grad`` after ``loss.backward()`` and before
``optimizer.step()``. This is mathematically equivalent and avoids
constructing an extra graph node.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn


def proximal_term(local_params, global_params, mu: float) -> float:
    """Return (mu / 2) * || local - global ||^2 (for diagnostics only)."""
    sq = 0.0
    for lp, gp in zip(local_params, global_params):
        sq += float(((lp.detach() - gp.detach()) ** 2).sum().item())
    return 0.5 * mu * sq


def _snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def attach_proximal_hook(client, mu: float) -> None:
    """Attach a per-batch grad_hook that adds the proximal gradient.

    The hook captures ``w_global`` from the model's parameters at the
    start of ``local_update`` (right after ``load_state_dict``) by
    wrapping ``local_update`` so the snapshot is taken once per round.
    """
    if mu <= 0:
        return  # FedProx with mu=0 degenerates to FedAvg.

    original_local_update = client.local_update

    def patched_local_update(model: nn.Module, global_state: dict):
        # Snapshot of w_global aligned with model.parameters() order. The
        # snapshot lives on the client's device because the local model
        # is moved to GPU by the hook below before training begins.
        global_params: list[torch.Tensor] = []

        def grad_hook(local_model: nn.Module) -> None:
            # Lazily initialize the snapshot the first time the hook fires,
            # i.e. after the local model has been moved to client.device.
            if not global_params:
                for p in local_model.parameters():
                    global_params.append(p.detach().clone())
            for p, gp in zip(local_model.parameters(), global_params):
                if p.grad is not None:
                    p.grad.add_(p.detach() - gp, alpha=mu)

        client.grad_hook = grad_hook
        try:
            return original_local_update(model, global_state)
        finally:
            client.grad_hook = None

    client.local_update = patched_local_update  # type: ignore[assignment]
