"""Small CNN for MNIST (~21k parameters).

Architecture (kept small enough that federated training is feasible
in-process across ~100 clients):

    Conv2d(1, 16, 5) → ReLU → MaxPool2d(2)
    Conv2d(16, 32, 5) → ReLU → MaxPool2d(2)
    Flatten → Linear(512, 64) → ReLU → Linear(64, 10)

TODO:
    - Implement the model class.
    - Add a parameter-count sanity check.
"""

from __future__ import annotations


def make_mnist_cnn():
    """Return a small CNN suitable for MNIST classification."""
    raise NotImplementedError("To be implemented in Phase 1.1")
