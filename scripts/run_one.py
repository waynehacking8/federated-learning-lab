"""Run a single experiment config from the CLI.

Used by the unified-round sweep so multiple configs can be launched as
independent processes in parallel (the GPU sits at low utilisation for
this small CNN, so the bottleneck is per-process data loading -- running
several at once raises throughput without exceeding VRAM).
"""

from __future__ import annotations

import argparse

from scripts.experiment import ExperimentConfig, run_experiment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True, choices=["fedavg", "fedprox", "scaffold"])
    p.add_argument("--partition", default="label_skew", choices=["iid", "label_skew", "dirichlet"])
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--classes-per-client", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--local-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = ExperimentConfig(
        algorithm=args.algorithm,
        partition=args.partition,
        num_clients=args.num_clients,
        classes_per_client=args.classes_per_client,
        alpha=args.alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        batch_size=args.batch_size,
        participation_rate=args.participation_rate,
        mu=args.mu,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
