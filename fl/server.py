"""Federated learning server / aggregator.

Holds the global model and orchestrates one round at a time:

    1. select_clients(rate)   -- sample participating clients
    2. broadcast(global_state) -- send a CPU copy of the model
    3. each selected client returns (new_state_dict, n_samples)
    4. aggregate(...)         -- combine into the new global state
    5. evaluate(test_loader)  -- compute test loss/accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl.client import Client


def _state_to_cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


class Aggregator(Protocol):
    """Aggregation strategy: combines client updates into a new global state."""

    def aggregate(
        self,
        client_states: list[dict],
        sample_sizes: list[int],
    ) -> dict: ...


@dataclass
class Server:
    global_model: nn.Module
    aggregator: Aggregator
    clients: list[Client]
    test_loader: DataLoader
    device: torch.device
    participation_rate: float = 1.0
    rng: torch.Generator = field(default_factory=torch.Generator)

    def __post_init__(self) -> None:
        self.global_model.to(self.device)

    def _global_state_cpu(self) -> dict:
        return _state_to_cpu(self.global_model.state_dict())

    def select_clients(self) -> list[Client]:
        k = max(1, int(round(len(self.clients) * self.participation_rate)))
        idx = torch.randperm(len(self.clients), generator=self.rng)[:k].tolist()
        return [self.clients[i] for i in idx]

    def run_round(self, round_index: int) -> dict:
        global_state = self._global_state_cpu()
        selected = self.select_clients()

        client_states: list[dict] = []
        sample_sizes: list[int] = []
        for client in selected:
            new_state, n = client.local_update(self.global_model, global_state)
            client_states.append(new_state)
            sample_sizes.append(n)

        new_global = self.aggregator.aggregate(client_states, sample_sizes)
        # Load aggregated CPU state into the GPU-resident model.
        self.global_model.load_state_dict(
            {k: v.to(self.device) for k, v in new_global.items()}
        )

        loss, acc = self.evaluate()
        return {
            "round": round_index,
            "n_clients": len(selected),
            "test_loss": loss,
            "test_acc": acc,
        }

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.global_model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for xb, yb in self.test_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.global_model(xb)
            total_loss += loss_fn(logits, yb).item()
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += yb.numel()
        self.global_model.train()
        return total_loss / total_n, total_correct / total_n
