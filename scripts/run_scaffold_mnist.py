"""End-to-end SCAFFOLD experiment on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--partition", choices=["iid", "label_skew", "dirichlet"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--classes-per-client", type=int, default=2)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--local-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path("results") / f"scaffold_{args.partition}_a{args.alpha}")

    cfg = ExperimentConfig(
        algorithm="scaffold",
        partition=args.partition,
        num_clients=args.num_clients,
        classes_per_client=args.classes_per_client,
        alpha=args.alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
