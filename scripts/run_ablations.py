"""Phase 2.2 + 3.2 ablations.

E-sweep: FedAvg on Dir(0.1) with local_epochs in {1, 5, 10}. More local
work per round increases client drift under Non-IID, so larger E should
reach a *lower* plateau (or be less stable) despite doing more compute
per round -- the "larger E -> more drift" claim (notes section 1.6).

mu-sweep: FedProx on Dir(0.1) across the cross-silo mu range
{0.001, 0.01} and the cross-device range {0.05, 0.1}, reporting
rounds-to-target -- the mu-table from notes section 10.1.3.

Both reuse scripts.experiment.run_experiment. Outputs:
    results/E_SWEEP_REPORT.md + results/fedavg_noniid.png
    results/MU_SWEEP_REPORT.md + results/fedavg_vs_fedprox.png
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiment import ExperimentConfig, run_experiment

ROUNDS = 40
TARGET = 0.95


def _rounds_to(history, target):
    return next((h["round"] for h in history if h["test_acc"] >= target), None)


def e_sweep():
    runs = {}
    for E in (1, 5, 10):
        cfg = ExperimentConfig(
            algorithm="fedavg", partition="dirichlet", alpha=0.1, num_clients=10,
            rounds=ROUNDS, local_epochs=E, local_lr=0.01, batch_size=64, seed=0,
            output_dir=f"results/ablation_E{E}",
        )
        print(f"\n========== E-sweep: FedAvg Dir(0.1) E={E} ==========", flush=True)
        summary = run_experiment(cfg)
        runs[E] = summary
    return runs


def mu_sweep():
    runs = {}
    for mu in (0.001, 0.01, 0.05, 0.1):
        cfg = ExperimentConfig(
            algorithm="fedprox", partition="dirichlet", alpha=0.1, num_clients=10,
            rounds=ROUNDS, local_epochs=5, local_lr=0.01, batch_size=64, mu=mu, seed=0,
            output_dir=f"results/ablation_mu{mu}",
        )
        print(f"\n========== mu-sweep: FedProx Dir(0.1) mu={mu} ==========", flush=True)
        runs[mu] = run_experiment(cfg)
    # FedAvg reference (mu=0 equivalent).
    cfg = ExperimentConfig(
        algorithm="fedavg", partition="dirichlet", alpha=0.1, num_clients=10,
        rounds=ROUNDS, local_epochs=5, local_lr=0.01, batch_size=64, seed=0,
        output_dir="results/ablation_mu_fedavg",
    )
    print("\n========== mu-sweep: FedAvg reference ==========", flush=True)
    runs["fedavg"] = run_experiment(cfg)
    return runs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- E-sweep ----
    e_runs = e_sweep()
    fig, ax = plt.subplots(figsize=(8, 5))
    for E, s in e_runs.items():
        h = s["history"]
        ax.plot([x["round"] for x in h], [x["test_acc"] for x in h], marker=".", label=f"E={E}")
    ax.set_xlabel("round"); ax.set_ylabel("test accuracy")
    ax.set_title("FedAvg on Dir(0.1): local-epochs (E) sweep -- larger E, more drift")
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("results/fedavg_noniid.png", dpi=120); plt.close(fig)

    lines = [
        "# E-sweep: local epochs vs client drift (Phase 2.2)",
        "",
        f"FedAvg on Dirichlet(0.1), 10 clients, {ROUNDS} rounds.",
        "",
        "| E (local epochs) | Final acc | Best acc | Round to 0.90 |",
        "|---|---|---|---|",
    ]
    for E, s in e_runs.items():
        h = s["history"]; accs = [x["test_acc"] for x in h]
        lines.append(f"| {E} | {accs[-1]:.4f} | {max(accs):.4f} | {_rounds_to(h, 0.90)} |")
    lines += [
        "",
        "![E-sweep](fedavg_noniid.png)",
        "",
        "## Interpretation",
        "",
        "More local epochs per round = more client compute, but under Non-IID",
        "each client's local optimum is further from the global optimum, so",
        "larger E drives the local models further apart (client drift) before",
        "each aggregation. The expected signature is a noisier / lower-plateau",
        "curve for large E despite the extra compute -- exactly the trade-off",
        "FedProx and SCAFFOLD are designed to mitigate.",
        "",
    ]
    Path("results/E_SWEEP_REPORT.md").write_text("\n".join(lines))

    # ---- mu-sweep ----
    m_runs = mu_sweep()
    fig, ax = plt.subplots(figsize=(8, 5))
    for mu in (0.001, 0.01, 0.05, 0.1):
        h = m_runs[mu]["history"]
        ax.plot([x["round"] for x in h], [x["test_acc"] for x in h], marker=".", label=f"FedProx mu={mu}")
    h = m_runs["fedavg"]["history"]
    ax.plot([x["round"] for x in h], [x["test_acc"] for x in h], "k--", label="FedAvg")
    ax.set_xlabel("round"); ax.set_ylabel("test accuracy")
    ax.set_title("FedProx mu sweep on Dir(0.1): cross-silo (0.001-0.01) vs cross-device (0.05-0.1)")
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("results/fedavg_vs_fedprox.png", dpi=120); plt.close(fig)

    lines = [
        "# mu-sweep: FedProx proximal strength (Phase 3.2)",
        "",
        f"FedProx on Dirichlet(0.1), 10 clients, E=5, {ROUNDS} rounds. "
        f"Rounds-to-{TARGET} reported.",
        "",
        "| mu | Range | Final acc | Best acc | Rounds to target |",
        "|---|---|---|---|---|",
    ]
    range_label = {0.001: "cross-silo", 0.01: "cross-silo/default", 0.05: "cross-device", 0.1: "cross-device"}
    for mu in (0.001, 0.01, 0.05, 0.1):
        h = m_runs[mu]["history"]; accs = [x["test_acc"] for x in h]
        lines.append(f"| {mu} | {range_label[mu]} | {accs[-1]:.4f} | {max(accs):.4f} | "
                     f"{_rounds_to(h, TARGET)} |")
    h = m_runs["fedavg"]["history"]; accs = [x["test_acc"] for x in h]
    lines.append(f"| - (FedAvg) | baseline | {accs[-1]:.4f} | {max(accs):.4f} | {_rounds_to(h, TARGET)} |")
    lines += [
        "",
        "![mu-sweep](fedavg_vs_fedprox.png)",
        "",
        "## Interpretation",
        "",
        "The mu-table from the notes (10.1.3): cross-silo deployments use a",
        "looser anchor (mu 0.001-0.01) because clients are stable; cross-device",
        "uses a stronger anchor (mu 0.01-0.1) to make partial-work returns from",
        "flaky devices safe. On this mild Dir(0.1)/K=10 setting the differences",
        "are small (the partition is not severe enough to separate them sharply",
        "-- the same observation as the label_skew analysis), but the ordering",
        "and the rounds-to-target trend illustrate the trade-off.",
        "",
    ]
    Path("results/MU_SWEEP_REPORT.md").write_text("\n".join(lines))
    print("\nwrote E_SWEEP_REPORT.md, MU_SWEEP_REPORT.md, fedavg_noniid.png, fedavg_vs_fedprox.png")


if __name__ == "__main__":
    main()
