"""FedOpt -- server-side adaptive optimizers (Reddi et al., 2020).

Reframes federated aggregation as a single optimizer step on the
server. The weighted-average client delta is treated as a
"pseudo-gradient":

    delta_bar = sum_k (n_k / n) (w_local_k - w_global)
    g_tilde   = -delta_bar          # gradient = negative of the step we took

A server-side optimizer (Adam / Yogi / Adagrad) then updates the global
model from g_tilde with its own server learning rate and momentum
buffers. With server-Adam this is FedAdam; with Yogi, FedYogi.

    m_t = b1 m_{t-1} + (1-b1) g_tilde
    v_t = (Adam)     b2 v_{t-1} + (1-b2) g_tilde^2
          (Yogi)     v_{t-1} - (1-b2) g_tilde^2 sign(v_{t-1} - g_tilde^2)
          (Adagrad)  v_{t-1} + g_tilde^2
    w_{t+1} = w_t - lr_server * m_t / (sqrt(v_t) + tau)

tau = 1e-3 is the adaptivity floor from Reddi 2020 (larger than vanilla
Adam's epsilon). The aggregator is stateful (holds m, v, and its own
copy of the global weights) and is pluggable on top of FedAvg-style
client updates.
"""

from __future__ import annotations

import torch


class FedOptAggregator:
    """Server-side adaptive optimizer over the averaged client delta."""

    def __init__(
        self,
        init_state: dict,
        optimizer: str = "adam",
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        if optimizer not in ("adam", "yogi", "adagrad"):
            raise ValueError(f"unknown optimizer: {optimizer}")
        self.optimizer = optimizer
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        # Float keys only; integer counters are passed through.
        self.float_keys = [k for k, v in init_state.items() if v.is_floating_point()]
        self.w = {k: init_state[k].detach().cpu().to(torch.float32).clone() for k in self.float_keys}
        self.m = {k: torch.zeros_like(self.w[k]) for k in self.float_keys}
        self.v = {k: torch.zeros_like(self.w[k]) for k in self.float_keys}
        self._template = {k: v.clone() for k, v in init_state.items()}

    def aggregate(self, client_states: list[dict], sample_sizes: list[int]) -> dict:
        total = float(sum(sample_sizes))
        if total <= 0:
            raise ValueError("sample_sizes sum to zero")
        weights = [n / total for n in sample_sizes]

        out: dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            first = client_states[0][key]
            if key not in self.float_keys:
                out[key] = first.clone()
                continue

            # Weighted-average client delta relative to current server weights.
            delta = torch.zeros_like(self.w[key])
            for wt, state in zip(weights, client_states):
                delta = delta + (state[key].to(torch.float32) - self.w[key]) * wt
            g = -delta  # pseudo-gradient

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            g2 = g * g
            if self.optimizer == "adam":
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g2
            elif self.optimizer == "yogi":
                self.v[key] = self.v[key] - (1 - self.beta2) * g2 * torch.sign(self.v[key] - g2)
            else:  # adagrad
                self.v[key] = self.v[key] + g2

            self.w[key] = self.w[key] - self.server_lr * self.m[key] / (self.v[key].sqrt() + self.tau)
            out[key] = self.w[key].to(first.dtype)
        return out
