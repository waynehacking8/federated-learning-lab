"""Phase 9 experiment: server-side adaptive optimizer (FedAdam).

Compares on Dirichlet(alpha=0.1), K=10, E=5:
    - FedAvg              (plain weighted averaging)
    - FedAdam            (FedAvg client updates + server-side Adam)
    - FedProx + FedAdam  (proximal client updates + server-side Adam)

Acceptance gate: server-side adaptive optimizer reaches the FedAvg
target accuracy in >= 20% fewer rounds, OR exceeds FedAvg final by
>= 1 percentage point.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl.algorithms.fedavg import FedAvgAggregator
from fl.algorithms.fedopt import FedOptAggregator
from fl.algorithms.fedprox import attach_proximal_hook
from fl.client import Client
from fl.datasets.mnist_partition import dirichlet
from fl.models.cnn import make_mnist_cnn
from fl.server import Server

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ROUNDS = 50
NUM_CLIENTS = 10
ALPHA = 0.1
SEED = 0
SERVER_LR = 0.05
TARGET = 0.95  # rounds-to-target accuracy


def _set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    return train, test


def _clients(parts, train_set, device, mu=None):
    cs = []
    for cid, idx in enumerate(parts):
        if not idx:
            continue
        c = Client(client_id=cid, local_indices=idx, train_dataset=train_set,
                   device=device, local_epochs=5, local_lr=0.01, batch_size=64)
        cs.append(c)
    if mu is not None:
        for c in cs:
            attach_proximal_hook(c, mu=mu)
    return cs


def run_variant(name, parts, train_set, test_loader, device):
    _set_seeds(SEED)
    if name == "fedavg":
        clients = _clients(parts, train_set, device)
        agg = FedAvgAggregator()
    elif name == "fedadam":
        clients = _clients(parts, train_set, device)
        agg = FedOptAggregator(make_mnist_cnn().state_dict(), optimizer="adam", server_lr=SERVER_LR)
    elif name == "fedprox_adam":
        clients = _clients(parts, train_set, device, mu=0.01)
        agg = FedOptAggregator(make_mnist_cnn().state_dict(), optimizer="adam", server_lr=SERVER_LR)
    else:
        raise ValueError(name)

    rng = torch.Generator(); rng.manual_seed(SEED)
    server = Server(global_model=make_mnist_cnn(), aggregator=agg, clients=clients,
                    test_loader=test_loader, device=device, participation_rate=1.0, rng=rng)
    history = []
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        m = server.run_round(r)
        history.append(m)
        if r % 5 == 0 or r == ROUNDS:
            print(f"[{name}] round {r:3d} acc={m['test_acc']:.4f}", flush=True)
    wall = time.time() - t0
    accs = [h["test_acc"] for h in history]
    r2t = next((h["round"] for h in history if h["test_acc"] >= TARGET), None)
    return {"history": history, "final": accs[-1], "best": max(accs),
            "rounds_to_target": r2t, "wall_seconds": wall}


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set = _load()
    parts = dirichlet(list(train_set.targets.numpy()), NUM_CLIENTS, ALPHA, seed=SEED)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    results = {}
    for name in ("fedavg", "fedadam", "fedprox_adam"):
        print(f"\n========== {name} / Dir(0.1) ==========", flush=True)
        results[name] = run_variant(name, parts, train_set, test_loader, device)

    fa = results["fedavg"]; fad = results["fedadam"]
    # Gate: >=20% fewer rounds to target OR +1pp final.
    speed_gate = (fa["rounds_to_target"] is not None and fad["rounds_to_target"] is not None
                  and fad["rounds_to_target"] <= 0.8 * fa["rounds_to_target"])
    acc_gate = fad["final"] >= fa["final"] + 0.01
    gate = speed_gate or acc_gate

    out = Path("results/fedopt_comparison"); out.mkdir(parents=True, exist_ok=True)
    summary = {"rounds": ROUNDS, "alpha": ALPHA, "num_clients": NUM_CLIENTS,
               "server_lr": SERVER_LR, "target": TARGET, "results": results,
               "speed_gate": bool(speed_gate), "acc_gate": bool(acc_gate), "gate_pass": bool(gate)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {"fedavg": "FedAvg", "fedadam": "FedAdam (server Adam)",
              "fedprox_adam": "FedProx + server Adam"}
    for name, res in results.items():
        rounds = [h["round"] for h in res["history"]]
        accs = [h["test_acc"] for h in res["history"]]
        ax.plot(rounds, accs, marker=".", label=labels[name])
    ax.axhline(TARGET, ls=":", color="gray", alpha=0.6, label=f"target {TARGET}")
    ax.set_xlabel("round"); ax.set_ylabel("test accuracy")
    ax.set_title("FedOpt: server-side adaptive optimizer on Dir(0.1)")
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("results/fedopt_comparison.png", dpi=120); plt.close(fig)

    def r2t(res): return res["rounds_to_target"] if res["rounds_to_target"] else "not reached"
    lines = [
        "# FedOpt -- server-side adaptive optimizer (Phase 9)",
        "",
        f"Dirichlet(alpha={ALPHA}), {NUM_CLIENTS} clients, E=5, {ROUNDS} rounds, "
        f"server_lr={SERVER_LR}, target={TARGET}.",
        "",
        "| Variant | Final acc | Best acc | Rounds to target |",
        "|---|---|---|---|",
    ]
    for name, res in results.items():
        lines.append(f"| {labels[name]} | {res['final']:.4f} | {res['best']:.4f} | {r2t(res)} |")
    lines += [
        "",
        f"**Acceptance gate: {'PASS' if gate else 'FAIL'}**",
        f"- Speed gate (FedAdam rounds-to-target <= 0.8x FedAvg): "
        f"{'PASS' if speed_gate else 'fail'} "
        f"(FedAvg {r2t(fa)} vs FedAdam {r2t(fad)})",
        f"- Accuracy gate (FedAdam final >= FedAvg + 1pp): "
        f"{'PASS' if acc_gate else 'fail'} "
        f"({fad['final']:.4f} vs {fa['final']:.4f}, {(fad['final']-fa['final'])*100:+.2f}pp)",
        "",
        "![comparison](../fedopt_comparison.png)",
        "",
        "## Interpretation",
        "",
        "Treating the averaged client delta as a pseudo-gradient lets the",
        "server apply Adam's per-coordinate adaptive step sizes. Under Non-IID",
        "data the averaged delta is noisy and ill-scaled across coordinates;",
        "Adam's second-moment normalization (with the tau=1e-3 adaptivity",
        "floor from Reddi 2020) damps the noisy coordinates and accelerates",
        "the well-behaved ones.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nFedAdam final={fad['final']:.4f} (FedAvg {fa['final']:.4f}), "
          f"r2target {r2t(fad)} vs {r2t(fa)}, gate={'PASS' if gate else 'FAIL'}")
    print("saved results/fedopt_comparison/ and results/fedopt_comparison.png")


if __name__ == "__main__":
    main()
