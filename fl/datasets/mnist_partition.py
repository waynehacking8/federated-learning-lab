"""MNIST partitioning utilities for federated experiments.

Provides four partition schemes that systematically vary Non-IID
severity:

    iid(num_clients)
        Random equal split.

    label_skew(num_clients, classes_per_client)
        Each client sees ``classes_per_client`` of the 10 MNIST classes.

    quantity_skew(num_clients, alpha)
        Client dataset sizes follow a power-law distribution.

    dirichlet(num_clients, alpha)
        Class proportions per client drawn from Dir(α). Smaller α gives
        more skew; α = 100 approximates IID.

Returns a list of length num_clients, where each entry is a list of
sample indices into the source MNIST dataset.

TODO:
    - Implement all four partition schemes.
    - Add a visualization helper that plots the per-client class
      distribution as a stacked bar chart.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def iid(targets: Sequence[int], num_clients: int, seed: int = 0) -> list[list[int]]:
    """Equal-size random split of dataset indices across clients."""
    raise NotImplementedError("To be implemented in Phase 1.1")


def label_skew(targets: Sequence[int], num_clients: int, classes_per_client: int, seed: int = 0) -> list[list[int]]:
    """Each client receives samples from a subset of classes only."""
    raise NotImplementedError("To be implemented in Phase 2.1")


def dirichlet(targets: Sequence[int], num_clients: int, alpha: float, seed: int = 0) -> list[list[int]]:
    """Dirichlet-α partition (Hsu, Qi, Brown 2019)."""
    raise NotImplementedError("To be implemented in Phase 2.1")
