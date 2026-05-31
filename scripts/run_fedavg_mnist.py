"""End-to-end FedAvg experiment on MNIST.

Examples:
    python -m scripts.run_fedavg_mnist --partition iid --rounds 50
    python -m scripts.run_fedavg_mnist --partition dirichlet --alpha 0.1 --rounds 50

Produces ``<output_dir>/metrics.json`` and ``<output_dir>/curve.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment


def _parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--partition", choices=["iid", "label_skew", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--classes-per-client", type=int, default=2)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--local-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    if args.output_dir is None:
        tag = args.partition + (f"_a{args.alpha}" if args.partition == "dirichlet" else "")
        args.output_dir = str(Path("results") / f"fedavg_{tag}")

    return ExperimentConfig(
        algorithm="fedavg",
        partition=args.partition,
        num_clients=args.num_clients,
        classes_per_client=args.classes_per_client,
        alpha=args.alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        batch_size=args.batch_size,
        participation_rate=args.participation_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )


def main() -> None:
    cfg = _parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
