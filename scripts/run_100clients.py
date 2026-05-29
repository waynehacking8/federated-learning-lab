"""100-client severity test: FedAvg vs FedProx vs SCAFFOLD on label_skew(2).

At 100 clients each client averages 600 samples; with label_skew(2 of
10 classes) each client sees only 2 classes -- the canonical "severe
Non-IID" regime where FedProx and SCAFFOLD are supposed to clearly
beat FedAvg.

Reduced to 3 algorithms (skip FedProx mu=0.01, which we already
observed to be too mild) and 20 rounds to keep wall clock under an
hour. Participation rate stays at 1.0 (all 100 clients per round) so
we measure pure algorithmic difference, not sampling variance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment

COMMON = dict(
    num_clients=100,
    classes_per_client=2,
    local_epochs=5,
    local_lr=0.01,
    batch_size=64,
    seed=0,
    participation_rate=1.0,
)


def main() -> None:
    out_root = Path("results")
    t0 = time.time()
    runs = [
        ExperimentConfig(
            algorithm="fedavg",
            partition="label_skew",
            rounds=20,
            output_dir=str(out_root / "fedavg_labelskew_2_K100"),
            **COMMON,
        ),
        ExperimentConfig(
            algorithm="fedprox",
            partition="label_skew",
            mu=0.1,
            rounds=20,
            output_dir=str(out_root / "fedprox_labelskew_2_K100_mu0.1"),
            **COMMON,
        ),
        ExperimentConfig(
            algorithm="scaffold",
            partition="label_skew",
            rounds=20,
            output_dir=str(out_root / "scaffold_labelskew_2_K100"),
            **COMMON,
        ),
    ]
    for cfg in runs:
        tag = cfg.algorithm
        if cfg.algorithm == "fedprox":
            tag += f" mu={cfg.mu}"
        print(f"\n========== K=100 {tag} / label_skew(2) ==========", flush=True)
        run_experiment(cfg)
        print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    from scripts.make_comparison_plots import main as make_plots
    sys.argv = [
        "make_comparison_plots",
        "--runs",
        "fedavg_labelskew_2_K100",
        "fedprox_labelskew_2_K100_mu0.1",
        "scaffold_labelskew_2_K100",
        "--out", "results/K100_labelskew_comparison.png",
        "--report", "results/K100_LABELSKEW_REPORT.md",
    ]
    make_plots()

    from scripts.make_summary import main as make_summary
    make_summary()
    print(f"\nK=100 LABEL_SKEW SWEEP DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
