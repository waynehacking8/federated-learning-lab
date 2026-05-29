"""label_skew comparison sweep, then DP-FedAvg.

Chained because GPU is the bottleneck and these can not run in parallel.
Sequence:
    1. label_skew(2 classes per client) sweep:
       FedAvg, FedProx mu=0.01, FedProx mu=0.1, SCAFFOLD.
    2. DP-FedAvg on IID with the spec's main config (C=1, sigma=1).
    3. DP-FedAvg on IID with higher noise (C=1, sigma=5) to show the
       monotonic accuracy degradation acceptance from spec section 7.
    4. Regenerate results/SUMMARY.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment
from scripts.run_dp_fedavg import DPExperimentConfig, run as run_dp

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
    t0 = time.time()

    # 1) label_skew sweep.
    label_skew_runs = [
        ExperimentConfig(
            algorithm="fedavg",
            partition="label_skew",
            rounds=25,
            output_dir=str(out_root / "fedavg_labelskew_2"),
            **COMMON,
        ),
        ExperimentConfig(
            algorithm="fedprox",
            partition="label_skew",
            mu=0.01,
            rounds=25,
            output_dir=str(out_root / "fedprox_labelskew_2_mu0.01"),
            **COMMON,
        ),
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
    for cfg in label_skew_runs:
        tag = cfg.algorithm
        if cfg.algorithm == "fedprox":
            tag += f" mu={cfg.mu}"
        print(f"\n========== {tag} / label_skew(2) ==========", flush=True)
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

    # 2) DP-FedAvg main (sigma=1).
    print(f"\n========== DP-FedAvg IID  (C=1, sigma=1) ==========", flush=True)
    run_dp(DPExperimentConfig(
        partition="iid",
        num_clients=10,
        rounds=20,
        local_epochs=1,
        local_lr=0.05,
        batch_size=32,
        clip_C=1.0,
        noise_sigma=1.0,
        seed=0,
        output_dir="results/dp_fedavg_iid_C1_s1",
    ))
    print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    # 3) DP-FedAvg high-noise (sigma=5) to demonstrate monotonic degradation.
    print(f"\n========== DP-FedAvg IID  (C=1, sigma=5) ==========", flush=True)
    run_dp(DPExperimentConfig(
        partition="iid",
        num_clients=10,
        rounds=20,
        local_epochs=1,
        local_lr=0.05,
        batch_size=32,
        clip_C=1.0,
        noise_sigma=5.0,
        seed=0,
        output_dir="results/dp_fedavg_iid_C1_s5",
    ))
    print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    # 4) Cross-experiment summary.
    from scripts.make_summary import main as make_summary
    make_summary()

    print(f"\nLABEL_SKEW + DP CHAIN DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
