"""DP-FedAvg experiment on MNIST.

Runs FedAvg with DP-SGD wrapping the local update step.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl.algorithms.fedavg import FedAvgAggregator
from fl.datasets.mnist_partition import dirichlet, iid
from fl.models.cnn import make_mnist_cnn
from fl.server import Server
from privacy.dp import DPConfig, DPSGDClient, naive_epsilon

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class DPExperimentConfig:
    partition: str = "iid"
    alpha: float = 0.1
    num_clients: int = 10
    rounds: int = 20
    local_epochs: int = 1
    local_lr: float = 0.05
    batch_size: int = 32
    clip_C: float = 1.0
    noise_sigma: float = 1.0
    seed: int = 0
    output_dir: str = "results/dp_fedavg_iid"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg: DPExperimentConfig) -> dict:
    _set_seeds(cfg.seed)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)

    targets = list(train.targets.numpy())
    if cfg.partition == "iid":
        parts = iid(targets, cfg.num_clients, seed=cfg.seed)
    elif cfg.partition == "dirichlet":
        parts = dirichlet(targets, cfg.num_clients, cfg.alpha, seed=cfg.seed)
    else:
        raise ValueError(cfg.partition)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DPConfig(clip_C=cfg.clip_C, noise_sigma=cfg.noise_sigma)
    clients = [
        DPSGDClient(
            client_id=cid,
            local_indices=indices,
            train_dataset=train,
            device=device,
            dp=dp,
            local_epochs=cfg.local_epochs,
            local_lr=cfg.local_lr,
            batch_size=cfg.batch_size,
        )
        for cid, indices in enumerate(parts)
        if indices
    ]

    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    server = Server(
        global_model=make_mnist_cnn(),
        aggregator=FedAvgAggregator(),
        clients=clients,  # type: ignore[arg-type]
        test_loader=test_loader,
        device=device,
        participation_rate=1.0,
        rng=rng,
    )

    history: list[dict] = []
    t0 = time.time()
    for r in range(1, cfg.rounds + 1):
        m = server.run_round(r)
        history.append(m)
        print(
            f"[dp-fedavg/{cfg.partition} C={cfg.clip_C} s={cfg.noise_sigma}] "
            f"round {r:3d} acc={m['test_acc']:.4f} loss={m['test_loss']:.4f}"
        )
    wall = time.time() - t0

    # Naive epsilon estimate.
    # Steps per client per round = local_epochs * ceil(n_k / batch_size).
    avg_n = float(np.mean([c.n_samples() for c in clients]))
    steps_per_round = cfg.local_epochs * int(np.ceil(avg_n / cfg.batch_size))
    total_steps = cfg.rounds * steps_per_round
    sample_rate = cfg.batch_size / max(1, avg_n)
    eps_est = naive_epsilon(cfg.noise_sigma, sample_rate, total_steps)

    summary = {
        "config": asdict(cfg),
        "history": history,
        "final_acc": history[-1]["test_acc"],
        "wall_seconds": wall,
        "naive_epsilon": eps_est,
        "total_steps": total_steps,
    }
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Curve.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = [h["round"] for h in history]
    accs = [h["test_acc"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, accs, marker=".")
    ax.set_xlabel("round")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"DP-FedAvg / {cfg.partition} / C={cfg.clip_C} sigma={cfg.noise_sigma}")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=120)
    plt.close(fig)

    # Report.
    final = accs[-1]
    best = max(accs)
    r90 = next((h["round"] for h in history if h["test_acc"] >= 0.90), None)
    lines = [
        f"# DP-FedAvg report -- {cfg.partition}",
        "",
        "## Configuration",
        "",
        "| Key | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in asdict(cfg).items()],
        "",
        "## Privacy (naive, not RDP)",
        "",
        f"- Noise sigma: {cfg.noise_sigma}",
        f"- Clip C: {cfg.clip_C}",
        f"- Total local SGD steps: {total_steps}",
        f"- Approx sample rate: {sample_rate:.4f}",
        f"- Naive epsilon estimate (delta=1e-5): **{eps_est:.2f}**",
        "  - Note: Abadi's RDP accountant gives much tighter bounds.",
        "",
        "## Results",
        "",
        f"- Final accuracy (round {cfg.rounds}): **{final:.4f}**",
        f"- Best accuracy: {best:.4f}",
        f"- Rounds to 0.90: {r90 if r90 is not None else 'not reached'}",
        f"- Wall clock: {wall:.1f}s",
        "",
        "## History",
        "",
        "| Round | Acc | Loss |",
        "|---|---|---|",
        *[f"| {h['round']} | {h['test_acc']:.4f} | {h['test_loss']:.4f} |" for h in history],
        "",
        "![curve](curve.png)",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"saved {out}/  (wall={wall:.1f}s, naive eps approx {eps_est:.2f})")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-C", type=float, default=1.0)
    p.add_argument("--noise-sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/dp_fedavg_{args.partition}_C{args.clip_C}_s{args.noise_sigma}"
    cfg = DPExperimentConfig(
        partition=args.partition,
        alpha=args.alpha,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        batch_size=args.batch_size,
        clip_C=args.clip_C,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
