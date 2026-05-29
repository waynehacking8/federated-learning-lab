"""Phase 8.1/8.2 experiment: robust aggregation under a sign-flip attack.

IID MNIST, n=10 clients, f=2 Byzantine. Byzantine clients return a
sign-flipped, norm-matched version of an honest update (credible-looking
but poisonous). Compares FedAvg (no defense) against coordinate-median,
Krum, Multi-Krum, trimmed-mean, and Bulyan.

Acceptance gate: median / Krum stay within 5pp of the no-attack
baseline; FedAvg degrades by >= 20pp.

Liu et al. (ICML 2023) caveat is documented: on IID data the
distance-based guarantee holds; under strong Non-IID it would not.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl.algorithms.fedavg import FedAvgAggregator
from fl.algorithms.robust import (
    BulyanAggregator, CoordinateMedianAggregator, KrumAggregator, TrimmedMeanAggregator,
)
from fl.client import Client
from fl.datasets.mnist_partition import iid
from fl.models.cnn import make_mnist_cnn
from fl.server import Server

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ROUNDS = 30
NUM_CLIENTS = 10
F_BYZ = 2
SEED = 0


def _set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    return train, test


SIGNFLIP_SCALE = 10.0  # amplification of the reversed update (standard attack strength)


class SignFlipClient(Client):
    """Byzantine client: trains honestly, then returns an amplified,
    sign-flipped update: w_byz = w_global - scale * (w_local - w_global).

    The honest update is (w_local - w_global). This reverses its direction
    and amplifies it by ``scale`` so the poison actually moves the average
    (a pure reflection, scale=1, is too weak when the honest delta is tiny
    after a couple of local epochs). This is the canonical sign-flip /
    gradient-scaling attack robust aggregators are evaluated against.
    """

    def local_update(self, model, global_state):
        new_state, n = super().local_update(model, global_state)
        flipped = {}
        for k, v in new_state.items():
            if v.is_floating_point():
                gw = global_state[k]
                flipped[k] = gw - SIGNFLIP_SCALE * (v - gw)
            else:
                flipped[k] = v
        return flipped, n


def _build_clients(parts, train_set, device, byzantine: bool):
    clients = []
    for cid, idx in enumerate(parts):
        if not idx:
            continue
        cls = SignFlipClient if (byzantine and cid < F_BYZ) else Client
        clients.append(cls(client_id=cid, local_indices=idx, train_dataset=train_set,
                           device=device, local_epochs=2, local_lr=0.05, batch_size=64))
    return clients


def run_variant(agg_name, parts, train_set, test_loader, device, byzantine: bool):
    _set_seeds(SEED)
    clients = _build_clients(parts, train_set, device, byzantine)
    if agg_name == "fedavg":
        agg = FedAvgAggregator()
    elif agg_name == "median":
        agg = CoordinateMedianAggregator()
    elif agg_name == "krum":
        agg = KrumAggregator(f=F_BYZ, multi_m=1)
    elif agg_name == "multikrum":
        agg = KrumAggregator(f=F_BYZ, multi_m=NUM_CLIENTS - F_BYZ)
    elif agg_name == "trimmed":
        agg = TrimmedMeanAggregator(beta=F_BYZ)
    elif agg_name == "bulyan":
        agg = BulyanAggregator(f=F_BYZ)
    else:
        raise ValueError(agg_name)

    rng = torch.Generator(); rng.manual_seed(SEED)
    server = Server(global_model=make_mnist_cnn(), aggregator=agg, clients=clients,
                    test_loader=test_loader, device=device, participation_rate=1.0, rng=rng)
    accs = []
    for r in range(1, ROUNDS + 1):
        m = server.run_round(r)
        accs.append(m["test_acc"])
    return accs[-1], max(accs)


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set = _load()
    parts = iid(list(train_set.targets.numpy()), NUM_CLIENTS, seed=SEED)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    aggs = ["fedavg", "median", "krum", "multikrum", "trimmed", "bulyan"]
    rows = {}
    t0 = time.time()
    # Baseline (no attack) only needs FedAvg as the reference point.
    print("=== no-attack baseline (FedAvg) ===", flush=True)
    base_final, base_best = run_variant("fedavg", parts, train_set, test_loader, device, byzantine=False)
    print(f"baseline FedAvg final={base_final:.4f}", flush=True)

    for a in aggs:
        print(f"=== under attack: {a} ===", flush=True)
        final, best = run_variant(a, parts, train_set, test_loader, device, byzantine=True)
        rows[a] = {"final": final, "best": best, "drop_vs_baseline": base_final - final}
        print(f"{a}: final={final:.4f} (drop {base_final-final:+.4f})", flush=True)

    # Gates.
    fedavg_drop = rows["fedavg"]["drop_vs_baseline"]
    median_ok = abs(rows["median"]["drop_vs_baseline"]) <= 0.05
    krum_ok = abs(rows["krum"]["drop_vs_baseline"]) <= 0.05
    fedavg_degrades = fedavg_drop >= 0.20
    gate = median_ok and krum_ok and fedavg_degrades

    out = Path("results/robust"); out.mkdir(parents=True, exist_ok=True)
    summary = {"rounds": ROUNDS, "num_clients": NUM_CLIENTS, "f": F_BYZ, "seed": SEED,
               "baseline_fedavg_final": base_final, "under_attack": rows,
               "median_within_5pp": bool(median_ok), "krum_within_5pp": bool(krum_ok),
               "fedavg_degrades_20pp": bool(fedavg_degrades), "gate_pass": bool(gate),
               "wall_seconds": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(rows.keys())
    finals = [rows[a]["final"] for a in names]
    bars = ax.bar(names, finals, color=["#d62728"] + ["#2ca02c"] * (len(names) - 1))
    ax.axhline(base_final, ls="--", color="gray", label=f"no-attack baseline {base_final:.3f}")
    ax.axhline(base_final - 0.05, ls=":", color="green", alpha=0.5, label="baseline - 5pp")
    ax.set_ylabel("final test accuracy (under f=2 sign-flip attack)")
    ax.set_title(f"Robust aggregation vs sign-flip attack (IID, n={NUM_CLIENTS}, f={F_BYZ})")
    ax.set_ylim(0, 1.0); ax.legend(loc="lower right"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("results/robust_aggregation.png", dpi=120); plt.close(fig)

    lines = [
        "# Byzantine-robust aggregation (Phase 8.1/8.2)",
        "",
        f"IID MNIST, n={NUM_CLIENTS} clients, f={F_BYZ} Byzantine (sign-flip),",
        f"{ROUNDS} rounds, seed {SEED}.",
        "",
        f"No-attack FedAvg baseline: **{base_final:.4f}**.",
        "",
        "| Aggregator | Final acc (under attack) | Drop vs baseline |",
        "|---|---|---|",
    ]
    for a in names:
        lines.append(f"| {a} | {rows[a]['final']:.4f} | {rows[a]['drop_vs_baseline']:+.4f} |")
    lines += [
        "",
        f"**Acceptance gate: {'PASS' if gate else 'FAIL'}**",
        f"- Median within 5pp of baseline: {'PASS' if median_ok else 'fail'} "
        f"({rows['median']['drop_vs_baseline']:+.4f})",
        f"- Krum within 5pp of baseline: {'PASS' if krum_ok else 'fail'} "
        f"({rows['krum']['drop_vs_baseline']:+.4f})",
        f"- FedAvg degrades >= 20pp: {'PASS' if fedavg_degrades else 'fail'} "
        f"({fedavg_drop:+.4f})",
        "",
        "![robust](../robust_aggregation.png)",
        "",
        "## Liu et al. (ICML 2023) caveat",
        "",
        "These results are on IID data, where the distance-based guarantee of",
        "Krum/Bulyan holds: honest updates cluster, the attacker is an outlier.",
        "Under strong Non-IID data (e.g. Dir(0.1)), honest-client divergence",
        "becomes comparable to attacker divergence, so distance-based",
        "aggregators silently lose their guarantee and can discard honest",
        "minorities. Robust aggregation and heterogeneity-robustness are",
        "distinct problems; classical aggregators solve only the former.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\ngate={'PASS' if gate else 'FAIL'} (FedAvg drop {fedavg_drop:+.4f}, "
          f"median {rows['median']['drop_vs_baseline']:+.4f}, krum {rows['krum']['drop_vs_baseline']:+.4f})")
    print("saved results/robust/ and results/robust_aggregation.png")


if __name__ == "__main__":
    main()
