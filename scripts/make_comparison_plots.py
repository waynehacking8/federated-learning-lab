"""Aggregate comparison plots across algorithm/partition runs.

Reads ``results/*/metrics.json`` for any combination of algorithms and
produces side-by-side plots and a summary report.

Usage:
    python -m scripts.make_comparison_plots \\
        --runs fedavg_iid fedavg_dirichlet_a0.1 fedprox_dirichlet_a0.1_mu0.01 \\
               scaffold_dirichlet_a0.1 \\
        --out results/three_way_comparison.png \\
        --report results/THREE_WAY_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_run(name: str) -> dict:
    path = Path("results") / name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run the experiment first")
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="results/three_way_comparison.png")
    ap.add_argument("--report", type=str, default="results/THREE_WAY_REPORT.md")
    args = ap.parse_args()

    runs = [(name, _load_run(name)) for name in args.runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, run in runs:
        rounds = [h["round"] for h in run["history"]]
        accs = [h["test_acc"] for h in run["history"]]
        cfg = run["config"]
        label = f"{cfg['algorithm']} ({cfg['partition']}"
        if cfg.get("partition") == "dirichlet":
            label += f", a={cfg['alpha']}"
        if cfg.get("algorithm") == "fedprox":
            label += f", mu={cfg['mu']}"
        label += ")"
        ax.plot(rounds, accs, marker=".", linewidth=1.2, label=label)

    ax.set_xlabel("communication round")
    ax.set_ylabel("test accuracy")
    ax.set_title("Federated learning algorithms -- convergence comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"saved {args.out}")

    # Report.
    lines = [
        "# Three-way algorithm comparison",
        "",
        "## Runs",
        "",
        "| Run | Algorithm | Partition | Rounds | Final acc | Best acc | Round to 0.90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs:
        cfg = run["config"]
        history = run["history"]
        accs = [h["test_acc"] for h in history]
        final = accs[-1]
        best = max(accs)
        r90 = next((h["round"] for h in history if h["test_acc"] >= 0.90), None)
        part = cfg["partition"]
        if part == "dirichlet":
            part += f" (alpha={cfg['alpha']})"
        lines.append(
            f"| {name} | {cfg['algorithm']} | {part} | {cfg['rounds']} | "
            f"{final:.4f} | {best:.4f} | {r90 if r90 is not None else '-'} |"
        )

    lines += [
        "",
        f"![comparison]({Path(args.out).name})",
        "",
        "## Observations",
        "",
        "- For Non-IID partitions (Dirichlet alpha=0.1), FedAvg drifts; FedProx",
        "  anchors via the proximal term (mu); SCAFFOLD corrects drift via",
        "  control variates.",
        "- Plot accuracy vs *round*, not wall-clock: SCAFFOLD pays ~2x",
        "  communication per round in return for fewer rounds to a target.",
        "",
    ]
    Path(args.report).write_text("\n".join(lines))
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
