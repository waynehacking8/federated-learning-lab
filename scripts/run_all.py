"""Run all phase experiments sequentially in one process.

Shares MNIST loading across runs (saves a few seconds each). Suitable
for a single "complete the demo" pass; for parameter sweeps, run the
individual ``run_*.py`` scripts.

Each experiment writes ``results/<name>/{metrics.json, curve.png, REPORT.md}``.
The three-way comparison and a SUMMARY.md are produced at the end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment

# Defaults tuned for "convergence visible, GPU wall clock reasonable".
COMMON = dict(num_clients=10, local_epochs=5, local_lr=0.01, batch_size=64, seed=0)


def main() -> None:
    out_root = Path("results")
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = [
        # Phase 1.4 -- IID baseline. Converges fast.
        ExperimentConfig(
            algorithm="fedavg",
            partition="iid",
            rounds=15,
            output_dir=str(out_root / "fedavg_iid"),
            **COMMON,
        ),
        # Phase 2 -- Non-IID FedAvg (Dir alpha=0.1). Reference for drift.
        ExperimentConfig(
            algorithm="fedavg",
            partition="dirichlet",
            alpha=0.1,
            rounds=25,
            output_dir=str(out_root / "fedavg_dirichlet_a0.1"),
            **COMMON,
        ),
        # Phase 3 -- FedProx on the same Non-IID setting.
        ExperimentConfig(
            algorithm="fedprox",
            partition="dirichlet",
            alpha=0.1,
            mu=0.01,
            rounds=25,
            output_dir=str(out_root / "fedprox_dirichlet_a0.1_mu0.01"),
            **COMMON,
        ),
        # Phase 4 -- SCAFFOLD on the same Non-IID setting.
        ExperimentConfig(
            algorithm="scaffold",
            partition="dirichlet",
            alpha=0.1,
            rounds=25,
            output_dir=str(out_root / "scaffold_dirichlet_a0.1"),
            **COMMON,
        ),
    ]

    t0 = time.time()
    for cfg in experiments:
        print(f"\n========== {cfg.algorithm} / {cfg.partition} alpha={cfg.alpha} ==========",
              flush=True)
        run_experiment(cfg)
        print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    # Comparison plot + summary.
    from scripts.make_comparison_plots import main as make_plots
    sys.argv = [
        "make_comparison_plots",
        "--runs", "fedavg_iid", "fedavg_dirichlet_a0.1",
        "fedprox_dirichlet_a0.1_mu0.01", "scaffold_dirichlet_a0.1",
        "--out", "results/three_way_comparison.png",
        "--report", "results/THREE_WAY_REPORT.md",
    ]
    make_plots()

    from scripts.make_summary import main as make_summary
    make_summary()

    print(f"\nALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
