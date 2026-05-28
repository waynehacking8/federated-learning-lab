"""SCAFFOLD — control-variate-corrected federated learning
(Karimireddy et al., 2019).

Maintains:
    - ``c`` — server-side global control variate.
    - ``c_k`` — per-client control variate.

Each client's effective gradient is corrected:

    g̃ = g − c_k + c

At round end:
    - Client updates its ``c_k`` based on its trajectory.
    - Server updates ``c`` to the mean of returned client ``c_k`` deltas.

TODO:
    - Implement control-variate state structures.
    - Implement the corrected-gradient hook for the client optimizer.
    - Implement the c, c_k update rules at round boundaries.

Note: SCAFFOLD's communication cost is ~2× FedAvg because c_k must
also be transmitted. This is the price for faster convergence.
"""

from __future__ import annotations


def corrected_gradient(grad, c_local, c_global):
    """Return grad − c_local + c_global."""
    raise NotImplementedError("To be implemented in Phase 4.1")
