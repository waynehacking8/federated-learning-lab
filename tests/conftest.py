"""Shared pytest fixtures for the federated-learning-lab test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Seed all RNGs we care about for reproducibility."""
    import random

    random.seed(0)
    np.random.seed(0)

    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except ImportError:
        pass


@pytest.fixture
def tiny_dataset_size() -> int:
    """Synthetic dataset size used by partition tests (kept small for speed)."""
    return 600  # 60 samples per class * 10 classes
