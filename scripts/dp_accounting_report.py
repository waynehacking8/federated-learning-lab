"""Rigorous (epsilon, delta) accounting for the DP-FedAvg runs.

The naive estimator in privacy/dp.py uses strong composition and ignores
privacy amplification by subsampling -- it overshoots by ~2-3 orders of
magnitude. This script computes the tight bound with the subsampled
Gaussian RDP accountant (Google's dp_accounting, the same engine Opacus
wraps) and cross-checks it against the analytic per-step Gaussian RDP
composition.

Mechanism actually run (privacy/dp.py DPSGDClient):
  - Per-sample grads clipped to L2 norm C, summed; Gaussian noise with
    std = sigma * C added to the sum; result averaged by batch size.
    => standard DP-SGD with noise multiplier z = sigma, sensitivity C.
  - Each client: IID partition, n_k = 60000/10 = 6000 samples,
    batch B = 32  => sampling rate q = B / n_k.
  - local_epochs E = 1, rounds R = 20
    => steps per client = E * ceil(n_k / B) = 188; total = R * 188 = 3760.
  - Privacy is per client (each client runs its own DP-SGD); the reported
    epsilon is the per-client guarantee, which is what DP-FedAvg protects.

Caveat: the DataLoader shuffles and cuts fixed-size batches (sampling
without replacement); the amplification theorem assumes Poisson
sampling. This is the same approximation Opacus makes by default; the
difference is negligible at this q and is noted in the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dp_accounting import dp_event, pld, rdp

N_TRAIN = 60000
N_CLIENTS = 10
BATCH = 32
EPOCHS = 1
ROUNDS = 20
DELTA = 1e-5

N_K = N_TRAIN // N_CLIENTS  # 6000
Q = BATCH / N_K
STEPS_PER_ROUND = EPOCHS * math.ceil(N_K / BATCH)
TOTAL_STEPS = ROUNDS * STEPS_PER_ROUND

# Standard RDP orders used by Opacus / TF-Privacy.
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def rdp_epsilon(noise_multiplier: float, delta: float = DELTA) -> float:
    """Tight subsampled-Gaussian RDP epsilon over TOTAL_STEPS."""
    accountant = rdp.RdpAccountant(ORDERS)
    event = dp_event.PoissonSampledDpEvent(
        Q, dp_event.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, TOTAL_STEPS)
    return accountant.get_epsilon(delta)


def pld_epsilon(noise_multiplier: float, delta: float = DELTA) -> float:
    """Independent cross-check via the Privacy Loss Distribution accountant.

    A different algorithm from RDP; the two should agree closely (PLD is
    typically a touch tighter). Agreement is evidence the bound is correct,
    not a single-library artefact.
    """
    accountant = pld.PLDAccountant()
    event = dp_event.PoissonSampledDpEvent(
        Q, dp_event.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, TOTAL_STEPS)
    return accountant.get_epsilon(delta)


def analytic_unsubsampled_epsilon(noise_multiplier: float, delta: float = DELTA) -> float:
    """Cross-check: composition WITHOUT subsampling amplification.

    Per-step Gaussian RDP is alpha / (2 z^2); composing T steps gives
    RDP(alpha) = T * alpha / (2 z^2); convert to (eps, delta) and minimise
    over alpha. This must be >= the subsampled bound (no amplification),
    and << the naive strong-composition number.
    """
    z = noise_multiplier
    best = float("inf")
    for alpha in ORDERS:
        if alpha <= 1:
            continue
        rdp_alpha = TOTAL_STEPS * alpha / (2 * z * z)
        eps = rdp_alpha + math.log1p(-1.0 / alpha) - math.log(delta * alpha) / (alpha - 1)
        best = min(best, eps)
    return best


def main() -> None:
    print(f"q={Q:.6f}  steps/round={STEPS_PER_ROUND}  total_steps={TOTAL_STEPS}  delta={DELTA}")
    print(f"{'sigma':>6} | {'RDP eps':>9} | {'PLD eps':>9} | {'analytic no-sub':>16} | {'naive (old)':>12}")
    print("-" * 70)

    results = {}
    root = Path("results")
    for sigma in (1.0, 5.0):
        eps_tight = rdp_epsilon(sigma)
        eps_pld = pld_epsilon(sigma)
        eps_analytic = analytic_unsubsampled_epsilon(sigma)
        tag = f"s{int(sigma)}"
        run_dir = root / f"dp_fedavg_iid_C1_{tag}"
        naive = None
        mj = run_dir / "metrics.json"
        if mj.exists():
            naive = json.loads(mj.read_text()).get("naive_epsilon")
        print(f"{sigma:>6.1f} | {eps_tight:>9.3f} | {eps_pld:>9.3f} | {eps_analytic:>16.3f} | "
              f"{(round(naive,1) if naive else 'n/a'):>12}")
        results[tag] = {
            "sigma": sigma,
            "rdp_epsilon": eps_tight,
            "pld_epsilon": eps_pld,
            "analytic_no_subsampling_epsilon": eps_analytic,
            "naive_epsilon": naive,
            "q": Q,
            "total_steps": TOTAL_STEPS,
            "delta": DELTA,
        }

    (root / "dp_accounting.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {root}/dp_accounting.json")


if __name__ == "__main__":
    main()
