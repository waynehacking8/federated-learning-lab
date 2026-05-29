"""Resume the killed label_skew + DP chain, running ONLY the missing pieces.

The original run_labelskew_and_dp.py chain was killed mid-SCAFFOLD. The
three completed label_skew runs (fedavg, fedprox mu=0.01, fedprox mu=0.1)
are intact on disk, so re-running them would waste GPU time. This script
picks up exactly where the kill happened:

    1. SCAFFOLD on label_skew(2)  -> results/scaffold_labelskew_2
    2. label_skew comparison plot + report (all four runs now present)
    3. DP-FedAvg IID, C=1 sigma=1  -> results/dp_fedavg_iid_C1_s1
    4. DP-FedAvg IID, C=1 sigma=5  -> results/dp_fedavg_iid_C1_s5
    5. regenerate results/SUMMARY.md

Idempotent: each step skips itself if its metrics.json already exists.
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


def _done(name: str) -> bool:
    return (Path("results") / name / "metrics.json").exists()


def main() -> None:
    out_root = Path("results")
    t0 = time.time()

    # 1) SCAFFOLD label_skew(2) -- the run that was killed.
    if _done("scaffold_labelskew_2"):
        print("[skip] scaffold_labelskew_2 already complete", flush=True)
    else:
        print("\n========== scaffold / label_skew(2) ==========", flush=True)
        run_experiment(
            ExperimentConfig(
                algorithm="scaffold",
                partition="label_skew",
                rounds=25,
                output_dir=str(out_root / "scaffold_labelskew_2"),
                **COMMON,
            )
        )
        print(f"--- elapsed total: {time.time() - t0:.0f}s ---", flush=True)

    # 2) label_skew comparison plot + report.
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

    # 3) DP-FedAvg main (sigma=1).
    if _done("dp_fedavg_iid_C1_s1"):
        print("[skip] dp_fedavg_iid_C1_s1 already complete", flush=True)
    else:
        print("\n========== DP-FedAvg IID  (C=1, sigma=1) ==========", flush=True)
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

    # 4) DP-FedAvg high-noise (sigma=5).
    if _done("dp_fedavg_iid_C1_s5"):
        print("[skip] dp_fedavg_iid_C1_s5 already complete", flush=True)
    else:
        print("\n========== DP-FedAvg IID  (C=1, sigma=5) ==========", flush=True)
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

    # 5) Cross-experiment summary.
    from scripts.make_summary import main as make_summary

    make_summary()
    print(f"\nRESUME label_skew + DP DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
