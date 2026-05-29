"""FedLoRA -- federated parameter-efficient fine-tuning (Phase 10).

Federate only LoRA adapter weights (+ the classifier head) of a frozen
base transformer. Two aggregation strategies:

    FedIT      : FedAvg over all adapter params (A and B) + classifier.
                 (Sun et al. 2024 -- the straightforward baseline.)
    FedSA-LoRA : share only the A matrices (+ classifier); each client
                 keeps its own B matrices across rounds.
                 (Guo et al. ICLR 2025 -- A learns shared structure,
                 B learns client-specific structure.)

Both operate on dicts of {param_name: tensor} restricted to the
trainable LoRA + classifier parameters, so payload accounting is just
the summed tensor sizes of whatever is shared.
"""

from __future__ import annotations

import torch


def is_lora_A(name: str) -> bool:
    return "lora_A" in name


def is_lora_B(name: str) -> bool:
    return "lora_B" in name


def is_classifier(name: str) -> bool:
    return "classifier" in name or "pre_classifier" in name


def _weighted_avg(states: list[dict], weights: list[float], keys) -> dict:
    out = {}
    for k in keys:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for w, s in zip(weights, states):
            acc = acc + s[k].to(torch.float32) * w
        out[k] = acc.to(states[0][k].dtype)
    return out


class FedITAggregator:
    """Weighted-average ALL shared params (A, B, classifier)."""

    def aggregate(self, client_states: list[dict], sample_sizes: list[int]) -> dict:
        total = float(sum(sample_sizes))
        weights = [n / total for n in sample_sizes]
        return _weighted_avg(client_states, weights, list(client_states[0].keys()))

    def shared_keys(self, all_keys) -> list[str]:
        return list(all_keys)  # everything is shared


class FedSALoRAAggregator:
    """Average only the A matrices; B matrices AND the task head stay local.

    Per Guo et al. (ICLR 2025), A learns shared/general structure while B
    captures client-specific structure -- so only A is aggregated. The
    classification head is likewise client-specific under sharp label skew
    (each client sees a different label subset), so it is kept local too:
    averaging a head across clients with disjoint label supports pulls
    every client's predictions toward labels it never sees and destroys
    its own-distribution accuracy. (Sharing the head was measured to drop
    per-client accuracy by ~30pp -- see results/fedlora and D14.)

    The aggregate() output contains only the shared keys (the A matrices);
    the experiment runner keeps each client's own B and head across rounds.
    """

    def aggregate(self, client_states: list[dict], sample_sizes: list[int]) -> dict:
        total = float(sum(sample_sizes))
        weights = [n / total for n in sample_sizes]
        shared = [k for k in client_states[0].keys() if is_lora_A(k)]
        return _weighted_avg(client_states, weights, shared)

    def shared_keys(self, all_keys) -> list[str]:
        return [k for k in all_keys if is_lora_A(k)]
