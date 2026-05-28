"""Federated learning client / local trainer.

Each client holds a private local dataset and a current copy of the
global model. On each round it receives the latest global state, runs
``local_epochs`` of SGD on its data, and returns the local model state
(or the delta) plus the number of training samples used.

TODO:
    - Implement Client class with ``local_update`` method.
    - Make the local optimizer configurable (SGD, SGD+momentum, AdamW).
    - Support optional DP-SGD wrapping (delegated to ``privacy.dp``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Client:
    """Per-client state. To be implemented."""

    client_id: int

    def local_update(self, global_state: dict, local_epochs: int) -> tuple[dict, int]:
        """Return updated local state and the number of training samples used."""
        raise NotImplementedError("To be implemented in Phase 1.2")
