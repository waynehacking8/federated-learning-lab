"""Federated learning client / local trainer.

Each client holds a fixed list of dataset indices into a shared training
set and a stateless training routine. On each round it:

    1. Loads the broadcast global state into a fresh local model copy.
    2. Runs ``local_epochs`` of SGD on its local data.
    3. Returns the updated CPU state dict and the number of samples used.

The client does not own a persistent optimizer; SGD state (momentum
buffers) is intentionally re-initialized each round so that local
training is a clean re-anchor on the broadcast model. This matches
FedAvg / FedProx as described in the original papers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fl.models.cnn import make_mnist_cnn


def _state_to_cpu(state: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


@dataclass
class Client:
    """Per-client state: dataset slice + local training hyperparameters."""

    client_id: int
    local_indices: list[int]
    train_dataset: Dataset
    device: torch.device
    local_epochs: int = 5
    local_lr: float = 0.01
    batch_size: int = 32
    momentum: float = 0.0
    # Optional gradient post-processor: receives the model after backward()
    # and may modify .grad in-place. Used by FedProx for the proximal term.
    grad_hook: Optional[Callable[[nn.Module], None]] = None

    def __post_init__(self) -> None:
        self._loader_cache: Optional[DataLoader] = None
        # Persistent device-side model so we avoid deep-copying per round.
        # Re-initialized lazily on first local_update so we can match the
        # broadcast architecture exactly via load_state_dict.
        self._local_model: Optional[nn.Module] = None

    def _loader(self) -> DataLoader:
        if self._loader_cache is None:
            subset = Subset(self.train_dataset, self.local_indices)
            self._loader_cache = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
        return self._loader_cache

    def n_samples(self) -> int:
        return len(self.local_indices)

    def local_update(
        self,
        model: nn.Module,
        global_state: dict,
    ) -> tuple[dict, int]:
        """Train ``self._local_model`` from ``global_state`` and return CPU state.

        ``model`` is the server's reference architecture; we ignore the
        instance itself and reuse a persistent local model on
        ``self.device`` to avoid per-round deepcopy. The first call
        instantiates the local model.
        """
        if self._local_model is None:
            self._local_model = make_mnist_cnn().to(self.device)
        local_model = self._local_model
        local_model.load_state_dict(
            {k: v.to(self.device, non_blocking=True) for k, v in global_state.items()}
        )
        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.local_lr, momentum=self.momentum
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = self._loader()

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = local_model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if self.grad_hook is not None:
                    self.grad_hook(local_model)
                optimizer.step()

        return _state_to_cpu(local_model.state_dict()), self.n_samples()
