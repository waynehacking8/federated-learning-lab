"""SCAFFOLD -- control-variate-corrected federated learning
(Karimireddy et al., 2019).

State:
    c_global: server-side control variate (state-dict-shaped)
    c_local : per-client control variate (same shape)

Local update for each batch:
    g_tilde = g - c_local + c_global
    w_local <- w_local - lr * g_tilde

End-of-round (Option II, the simpler form):
    c_local_new = c_local - c_global + (1 / (K * lr)) * (w_global - w_local)
    delta_c     = c_local_new - c_local
    delta_w     = w_local - w_global

Server-side:
    w_global <- w_global + (1 / N_selected) * sum(delta_w_k)
    c_global <- c_global + (S / N_total) * mean(delta_c_k)

This module wires SCAFFOLD into the existing Server/Client scaffold
via two hooks:
    attach_scaffold(clients)
        - Patches each client's local_update to apply the corrected
          gradient during SGD and to return (delta_w, delta_c).
    ScaffoldAggregator(clients)
        - Reads the deltas attached to each client, applies the SCAFFOLD
          server-side update, and returns the new global state.

The aggregator side communicates with patched clients through a small
per-client buffer ``client._scaffold_round_outputs`` to avoid changing
the public Aggregator protocol (which only sees state dicts).
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn


def corrected_gradient(grad, c_local, c_global):
    """Return grad - c_local + c_global (elementwise)."""
    return grad - c_local + c_global


def _zeros_like_state(state: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in state.items() if v.is_floating_point()}


def _state_to_cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def attach_scaffold(clients: list) -> None:
    """Initialize c_local on each client and patch local_update.

    After this call, each client's local_update returns the standard
    ``(new_state_dict, n_samples)`` tuple (so the Server protocol is
    unchanged) and additionally stores the SCAFFOLD-specific deltas
    on ``client._scaffold_round_outputs``.
    """
    # We initialise c_local lazily on first round (we need an example
    # state dict to know shapes). c_global is held by the aggregator
    # since it is round-stateful and shared across clients.
    for client in clients:
        client._scaffold_c_local = None
        client._scaffold_c_global_ref = None  # populated by aggregator before round
        client._scaffold_round_outputs = None
        _patch_client(client)


def _patch_client(client) -> None:
    original_local_update = client.local_update

    def patched_local_update(model: nn.Module, global_state: dict):
        device = client.device
        # Lazy initialization of c_local in CPU, shape-matched to model state.
        if client._scaffold_c_local is None:
            client._scaffold_c_local = _zeros_like_state(_state_to_cpu(global_state))

        # Bring c_local and c_global to device-side parameter list.
        c_local_dev = {
            k: v.to(device) for k, v in client._scaffold_c_local.items()
        }
        c_global_cpu = client._scaffold_c_global_ref  # may be None on round 1
        c_global_dev = (
            {k: v.to(device) for k, v in c_global_cpu.items()}
            if c_global_cpu is not None
            else {k: torch.zeros_like(v) for k, v in c_local_dev.items()}
        )

        # Build a parallel list aligned to model.named_parameters() since
        # state_dict() and named_parameters() may not iterate in the same
        # order. Filter to floating-point keys only (matches c_local).
        def aligned(named_params, source: dict) -> list[torch.Tensor]:
            out = []
            for name, _ in named_params:
                if name in source:
                    out.append(source[name])
                else:
                    out.append(None)
            return out

        # We need to track total optimizer steps K for the round-end update.
        step_count = {"k": 0}

        def grad_hook(local_model: nn.Module) -> None:
            named = list(local_model.named_parameters())
            c_loc = aligned(named, c_local_dev)
            c_glb = aligned(named, c_global_dev)
            for (name, p), cl, cg in zip(named, c_loc, c_glb):
                if p.grad is None:
                    continue
                if cl is not None and cg is not None:
                    p.grad.add_(cg - cl)
            step_count["k"] += 1

        client.grad_hook = grad_hook
        try:
            new_state, n = original_local_update(model, global_state)
        finally:
            client.grad_hook = None

        # End-of-round c_local update (Option II):
        # c_local_new = c_local - c_global + (1 / (K * lr)) * (w_global - w_local)
        K = max(1, step_count["k"])
        lr = client.local_lr
        c_local_new: dict[str, torch.Tensor] = {}
        delta_c: dict[str, torch.Tensor] = {}
        delta_w: dict[str, torch.Tensor] = {}
        for key, w_local in new_state.items():
            if not w_local.is_floating_point():
                continue
            w_global_t = global_state[key]
            cl = client._scaffold_c_local.get(key)
            cg = c_global_cpu.get(key) if c_global_cpu is not None else torch.zeros_like(w_local)
            if cl is None:
                cl = torch.zeros_like(w_local)
            update = (w_global_t - w_local) / (K * lr)
            c_local_new[key] = cl - cg + update
            delta_c[key] = c_local_new[key] - cl
            delta_w[key] = w_local - w_global_t

        client._scaffold_c_local = c_local_new  # persist for next round
        client._scaffold_round_outputs = {
            "delta_w": delta_w,
            "delta_c": delta_c,
            "n_samples": n,
            "w_global_broadcast": {
                k: v.clone() for k, v in global_state.items()
            },
        }
        return new_state, n

    client.local_update = patched_local_update  # type: ignore[assignment]


class ScaffoldAggregator:
    """Server-side SCAFFOLD aggregation.

    Maintains c_global. Reads per-client deltas from each client's
    ``_scaffold_round_outputs`` buffer (set by the patched local_update).
    """

    def __init__(self, clients: list, global_lr: float = 1.0) -> None:
        self.clients = clients
        self.c_global: dict[str, torch.Tensor] | None = None
        # Global step size eta_g (Karimireddy 2020, Algorithm 1 line 17:
        # x <- x + eta_g * (1/|S|) sum(y_i - x)). The paper's experiments use
        # eta_g = 1, which is the default here. eta_g < 1 dampens the
        # round-to-round oscillation that control variates can induce under
        # severe skew + many local epochs (the global iterates form a Markov
        # chain converging to a stationary distribution; a smaller eta_g
        # shrinks that distribution's variance).
        self.global_lr = global_lr

    def aggregate(
        self,
        client_states: list[dict],
        sample_sizes: list[int],
    ) -> dict:
        # Identify clients with fresh outputs (those that participated).
        participated = [
            c for c in self.clients if c._scaffold_round_outputs is not None
        ]
        if len(participated) != len(client_states):
            # Should not happen with synchronous selection, but guard anyway.
            pass

        # Lazy init c_global from the first delta_c we see.
        if self.c_global is None:
            template = participated[0]._scaffold_round_outputs["delta_c"]
            self.c_global = {k: torch.zeros_like(v) for k, v in template.items()}

        # Recover w_global from any participating client's stored broadcast
        # (the Aggregator protocol does not give us the server state directly,
        # so each patched client caches the global_state it received).
        w_global = {
            k: v.clone()
            for k, v in participated[0]._scaffold_round_outputs["w_global_broadcast"].items()
        }
        first_delta = participated[0]._scaffold_round_outputs["delta_w"]

        # Aggregate delta_w by simple mean across selected clients (the
        # SCAFFOLD paper uses 1/|S|, not sample-weighted; sample-weighted
        # is also valid and often used in practice).
        S = len(participated)
        N = len(self.clients)

        mean_delta_w = {
            k: torch.zeros_like(v) for k, v in first_delta.items()
        }
        mean_delta_c = {k: torch.zeros_like(v) for k, v in mean_delta_w.items()}
        for c in participated:
            for k in mean_delta_w:
                mean_delta_w[k] += c._scaffold_round_outputs["delta_w"][k]
                mean_delta_c[k] += c._scaffold_round_outputs["delta_c"][k]
        for k in mean_delta_w:
            mean_delta_w[k] /= S
            mean_delta_c[k] /= S

        # w_global <- w_global + eta_g * mean(delta_w). Non-float keys (BN
        # counters, etc.) pass through from the broadcast state.
        new_global: dict[str, torch.Tensor] = {}
        for k, v in w_global.items():
            if k in mean_delta_w and v.is_floating_point():
                new_global[k] = v + self.global_lr * mean_delta_w[k]
            else:
                new_global[k] = v.clone()

        # c_global <- c_global + (S / N) * mean(delta_c) ... but mean is
        # already mean (sum/S), so equivalent server update is:
        # c_global <- c_global + (S / N) * mean_delta_c
        for k in self.c_global:
            self.c_global[k] = self.c_global[k] + (S / N) * mean_delta_c[k]

        # Broadcast c_global to all clients for next round and clear outputs.
        for c in self.clients:
            c._scaffold_c_global_ref = {k: v.clone() for k, v in self.c_global.items()}
            c._scaffold_round_outputs = None

        return new_global
