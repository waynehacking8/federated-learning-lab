"""End-to-end FedAvg experiment on MNIST.

CLI:
    python -m scripts.run_fedavg_mnist \\
        --num-clients 10 \\
        --partition dirichlet \\
        --alpha 0.5 \\
        --rounds 50 \\
        --local-epochs 5 \\
        --participation-rate 1.0 \\
        --output-dir results/fedavg_dirichlet_0.5

Sequence:
    1. Load MNIST.
    2. Partition into ``num_clients`` according to ``partition`` scheme.
    3. Build Server and Clients.
    4. Loop ``rounds`` times; record test accuracy each round.
    5. Save the learning curve to ``output_dir/curve.png`` and metrics
       to ``output_dir/metrics.json``.

TODO:
    - Wire up argument parsing with typer.
    - Implement the experiment loop using fl.server.Server and
      fl.client.Client (once implemented).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("To be implemented in Phase 1.4")


if __name__ == "__main__":
    main()
