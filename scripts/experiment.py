"""Shared experiment scaffolding for FedAvg / FedProx / SCAFFOLD runs.

Exposes one function: ``run_experiment(config) -> metrics_dict``.
Used by the per-algorithm CLI scripts (run_fedavg_mnist, etc.) so
that all algorithms see identical data, partitions, seeds, and
evaluation logic. The only thing that varies across algorithms is
how clients update locally and how the server aggregates.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl.client import Client
from fl.datasets.mnist_partition import dirichlet, iid, label_skew
from fl.models.cnn import make_mnist_cnn

Algorithm = Literal["fedavg", "fedprox", "scaffold"]
Partition = Literal["iid", "label_skew", "dirichlet"]

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class ExperimentConfig:
    algorithm: Algorithm = "fedavg"
    partition: Partition = "iid"
    num_clients: int = 10
    classes_per_client: int = 2  # for label_skew
    alpha: float = 0.1  # for dirichlet
    rounds: int = 50
    local_epochs: int = 5
    local_lr: float = 0.01
    batch_size: int = 32
    participation_rate: float = 1.0
    mu: float = 0.01  # FedProx proximal strength
    global_lr: float = 1.0  # SCAFFOLD server-side step size eta_g
    seed: int = 0
    device: str = "cuda"
    output_dir: Optional[str] = None
    log_every: int = 1


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    return train, test


def _partition(cfg: ExperimentConfig, targets) -> list[list[int]]:
    if cfg.partition == "iid":
        return iid(targets, cfg.num_clients, seed=cfg.seed)
    if cfg.partition == "label_skew":
        return label_skew(targets, cfg.num_clients, cfg.classes_per_client, seed=cfg.seed)
    if cfg.partition == "dirichlet":
        return dirichlet(targets, cfg.num_clients, cfg.alpha, seed=cfg.seed)
    raise ValueError(f"unknown partition: {cfg.partition}")


def _make_clients(
    cfg: ExperimentConfig,
    train_dataset,
    parts: list[list[int]],
    device: torch.device,
) -> list[Client]:
    clients: list[Client] = []
    for cid, indices in enumerate(parts):
        if not indices:
            continue
        clients.append(
            Client(
                client_id=cid,
                local_indices=indices,
                train_dataset=train_dataset,
                device=device,
                local_epochs=cfg.local_epochs,
                local_lr=cfg.local_lr,
                batch_size=cfg.batch_size,
            )
        )
    return clients


def _make_server(cfg: ExperimentConfig, clients, test_loader, device):
    # Imports here keep the top-level import graph small and avoid pulling
    # algorithm modules until they're needed.
    from fl.server import Server

    if cfg.algorithm == "fedavg":
        from fl.algorithms.fedavg import FedAvgAggregator

        aggregator = FedAvgAggregator()
    elif cfg.algorithm == "fedprox":
        from fl.algorithms.fedavg import FedAvgAggregator
        from fl.algorithms.fedprox import attach_proximal_hook

        # FedProx is FedAvg aggregation + a client-side proximal gradient.
        aggregator = FedAvgAggregator()
        for client in clients:
            attach_proximal_hook(client, mu=cfg.mu)
    elif cfg.algorithm == "scaffold":
        from fl.algorithms.scaffold import ScaffoldAggregator, attach_scaffold

        attach_scaffold(clients)
        aggregator = ScaffoldAggregator(clients=clients, global_lr=cfg.global_lr)
    else:
        raise ValueError(f"unknown algorithm: {cfg.algorithm}")

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    return Server(
        global_model=make_mnist_cnn(),
        aggregator=aggregator,
        clients=clients,
        test_loader=test_loader,
        device=device,
        participation_rate=cfg.participation_rate,
        rng=rng,
    )


def run_experiment(cfg: ExperimentConfig) -> dict:
    _set_seeds(cfg.seed)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    train_set, test_set = _load_mnist()
    parts = _partition(cfg, list(train_set.targets.numpy()))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    clients = _make_clients(cfg, train_set, parts, device)
    server = _make_server(cfg, clients, test_loader, device)

    history: list[dict] = []
    t0 = time.time()
    for r in range(1, cfg.rounds + 1):
        metrics = server.run_round(r)
        history.append(metrics)
        if r % cfg.log_every == 0 or r == cfg.rounds:
            print(
                f"[{cfg.algorithm}/{cfg.partition}] round {r:3d} "
                f"acc={metrics['test_acc']:.4f} loss={metrics['test_loss']:.4f} "
                f"clients={metrics['n_clients']}"
            )

    wall = time.time() - t0
    summary = {
        "config": asdict(cfg),
        "history": history,
        "final_acc": history[-1]["test_acc"],
        "wall_seconds": wall,
    }

    if cfg.output_dir:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "metrics.json").open("w") as f:
            json.dump(summary, f, indent=2)
        title = f"{cfg.algorithm} / {cfg.partition}"
        _save_curve(history, out / "curve.png", title=title)
        _save_report(summary, parts, out / "REPORT.md")
        print(f"saved {out}/metrics.json, curve.png, REPORT.md  (wall={wall:.1f}s)")
    return summary


def _save_report(summary: dict, parts: list[list[int]], path: Path) -> None:
    cfg = summary["config"]
    history = summary["history"]
    sizes = [len(p) for p in parts if p]
    accs = [h["test_acc"] for h in history]
    losses = [h["test_loss"] for h in history]
    final = accs[-1]
    best = max(accs)
    best_round = accs.index(best) + 1
    rounds_to_90 = next((h["round"] for h in history if h["test_acc"] >= 0.90), None)
    rounds_to_95 = next((h["round"] for h in history if h["test_acc"] >= 0.95), None)

    lines = [
        f"# Experiment report -- {cfg['algorithm']} / {cfg['partition']}",
        "",
        "## Configuration",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for k, v in cfg.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Partition",
        "",
        f"- Number of clients with data: **{len(sizes)}**",
        f"- Samples per client: min={min(sizes)}, median={int(np.median(sizes))}, "
        f"max={max(sizes)}, total={sum(sizes)}",
        "",
        "## Results",
        "",
        f"- Final test accuracy (round {cfg['rounds']}): **{final:.4f}**",
        f"- Best test accuracy: **{best:.4f}** at round {best_round}",
        f"- Final test loss: {losses[-1]:.4f}",
        f"- Rounds to 0.90 acc: {rounds_to_90 if rounds_to_90 is not None else 'not reached'}",
        f"- Rounds to 0.95 acc: {rounds_to_95 if rounds_to_95 is not None else 'not reached'}",
        f"- Wall clock: {summary['wall_seconds']:.1f}s",
        "",
        "## Per-round history",
        "",
        "| Round | Test acc | Test loss | Clients |",
        "|---|---|---|---|",
    ]
    for h in history:
        lines.append(
            f"| {h['round']} | {h['test_acc']:.4f} | {h['test_loss']:.4f} | {h['n_clients']} |"
        )
    lines += ["", "![curve](curve.png)", ""]
    path.write_text("\n".join(lines))


def _save_curve(history: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [h["round"] for h in history]
    accs = [h["test_acc"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, accs, marker=".", linewidth=1)
    ax.set_xlabel("round")
    ax.set_ylabel("test accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
