"""label_skew(2 classes/client) comparison: FedAvg vs FedProx vs SCAFFOLD.

The Dir(alpha=0.1) partition turned out moderate enough that FedAvg
itself reaches >= 0.97 on MNIST -- so FedProx and SCAFFOLD have no
visible margin to improve. label_skew with 2 of 10 classes per client
is much more severe (most clients can not recognize the 8 classes
they never see) and lets the drift-mitigation algorithms show their
effect.

Runs each algorithm sequentially, saves per-run artifacts, then
produces a label_skew comparison plot + report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment

COMMON = dict(
    num_clients=10,
    classes_per_client=2,
    local_epochs=5,
    local_lr=0.01,
    batch_size=64,
    seed=0,
)


def main() -> None:
    out_root = Path("results")
    experiments = [
        ExperimentConfig(
            algorithm="fedavg",
            partition="label_skew",
            rounds=25,
            output_dir=str(out_root / "fedavg_labelskew_2"),
            **COMMON,
        ),
        # Same mu as Dir run for direct comparison.
        ExperimentConfig(
            algorithm="fedprox",
            partition="label_skew",
            mu=0.01,
            rounds=25,
            output_dir=str(out_root / "fedprox_labelskew_2_mu0.01"),
            **COMMON,
        ),
        # Stronger anchor; label_skew is severe.
        ExperimentConfig(
            algorithm="fedprox",
            partition="label_skew",
            mu=0.1,
            rounds=25,
            output_dir=str(out_root / "fedprox_labelskew_2_mu0.1"),
            **COMMON,
        ),
        ExperimentConfig(
            algorithm="scaffold",
            partition="label_skew",
            rounds=25,
            output_dir=str(out_root / "scaffold_labelskew_2"),
            **COMMON,
        ),
    ]

    t0 = time.time()
    for cfg in experiments:
        print(f"\n========== {cfg.algorithm} / label_skew(2) "
              f"{'mu=' + str(cfg.mu) if cfg.algorithm == 'fedprox' else ''} ==========",
              flush=True)
        run_experiment(cfg)
        print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    from scripts.make_comparison_plots import main as make_plots
    sys.argv = [
        "make_comparison_plots",
        "--runs",
        "fedavg_labelskew_2",
        "fedprox_labelskew_2_mu0.01",
        "fedprox_labelskew_2_mu0.1",
        "scaffold_labelskew_2",
        "--out", "results/labelskew_comparison.png",
        "--report", "results/LABELSKEW_REPORT.md",
    ]
    make_plots()

    print(f"\nLABEL_SKEW SWEEP DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
