"""FedProx — proximal-term-augmented local loss (Li et al., 2018).

Local objective becomes:

    L_k(w) = F_k(w) + (μ/2) * ‖w − w_t‖²

The aggregation step is identical to FedAvg; the proximal term lives
on the client side. This module exposes:

    - ``proximal_term(local_params, global_params, mu)`` for clients
      to add to their loss.
    - ``aggregate(...)`` re-uses FedAvg's weighted averaging.

TODO:
    - Implement the proximal term computation.
    - Add a helper that wraps a client's optimizer with the proximal
      gradient adjustment.
"""

from __future__ import annotations


def proximal_term(local_params, global_params, mu: float) -> float:
    """Return (μ/2) * ‖local_params - global_params‖²."""
    raise NotImplementedError("To be implemented in Phase 3.1")
