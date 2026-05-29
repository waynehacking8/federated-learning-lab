"""Aggregate all experiment results into results/SUMMARY.md.

Reads every ``results/*/metrics.json`` (plus the SecAgg demo and DP
runs when present) and produces a top-level summary report. Run this
after ``run_all.py`` (and optionally ``run_dp_fedavg.py``,
``secagg_demo.py``).
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_runs(root: Path) -> list[tuple[str, dict]]:
    runs = []
    for d in sorted(root.iterdir()):
        metrics = d / "metrics.json"
        if not metrics.exists():
            continue
        try:
            data = json.loads(metrics.read_text())
        except Exception:
            continue
        # Skip non-experiment metrics (e.g. the SecAgg demo), which lack a
        # "config" block and are summarized in their own dedicated section.
        if "config" not in data:
            continue
        runs.append((d.name, data))
    return runs


def _short(run: dict) -> dict:
    cfg = run["config"]
    out = {
        "algorithm": cfg.get("algorithm", "?"),
        "partition": cfg.get("partition", "?"),
    }
    if cfg.get("partition") == "dirichlet":
        out["alpha"] = cfg.get("alpha")
    if cfg.get("algorithm") == "fedprox":
        out["mu"] = cfg.get("mu")
    if "noise_sigma" in cfg:
        out["noise_sigma"] = cfg["noise_sigma"]
        out["clip_C"] = cfg["clip_C"]
    out["rounds"] = cfg.get("rounds")
    out["E"] = cfg.get("local_epochs")
    out["clients"] = cfg.get("num_clients")
    if "history" in run:
        accs = [h["test_acc"] for h in run["history"]]
        out["final_acc"] = round(accs[-1], 4)
        out["best_acc"] = round(max(accs), 4)
        out["round_to_0.90"] = next(
            (h["round"] for h in run["history"] if h["test_acc"] >= 0.90), None
        )
        out["round_to_0.95"] = next(
            (h["round"] for h in run["history"] if h["test_acc"] >= 0.95), None
        )
    if "naive_epsilon" in run:
        out["naive_epsilon"] = round(run["naive_epsilon"], 2)
    out["wall_seconds"] = round(run.get("wall_seconds", 0), 1)
    return out


def main() -> None:
    root = Path("results")
    if not root.exists():
        raise SystemExit("no results/ directory yet")

    runs = _read_runs(root)
    if not runs:
        raise SystemExit("no metrics.json files under results/")

    lines = [
        "# Experiment summary",
        "",
        "All runs use the small MNIST CNN (Conv 1->16, Conv 16->32, FC ->64 -> 10).",
        "Numbers below come from `results/<name>/metrics.json`.",
        "",
        "## Convergence table",
        "",
        "| Run | Algo | Partition | Rounds | E | K | Final acc | Best acc | r->0.90 | r->0.95 | Wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs:
        s = _short(run)
        if s["algorithm"] == "?":
            continue
        part = s["partition"]
        if "alpha" in s:
            part += f" (a={s['alpha']})"
        algo = s["algorithm"]
        if "mu" in s:
            algo += f" (mu={s['mu']})"
        if "noise_sigma" in s:
            algo += f" (C={s['clip_C']}, sigma={s['noise_sigma']})"
        lines.append(
            f"| {name} | {algo} | {part} | {s['rounds']} | {s['E']} | {s['clients']} | "
            f"{s['final_acc']} | {s['best_acc']} | "
            f"{s.get('round_to_0.90', '-') or '-'} | {s.get('round_to_0.95', '-') or '-'} | "
            f"{s['wall_seconds']} |"
        )

    # Plot pointers.
    lines += [
        "",
        "## Plots",
        "",
    ]
    for name, _ in runs:
        if (root / name / "curve.png").exists():
            lines.append(f"- [{name}]({name}/curve.png)")
    if (root / "three_way_comparison.png").exists():
        lines += [
            "",
            "## Three-way comparison",
            "",
            "![three way](three_way_comparison.png)",
            "",
            "See [THREE_WAY_REPORT.md](THREE_WAY_REPORT.md) for the table.",
            "",
        ]

    # Privacy section, if DP runs exist.
    dp_runs = [(n, r) for n, r in runs if "naive_epsilon" in r]
    if dp_runs:
        lines += ["", "## Differential privacy", "", "| Run | sigma | C | rounds | acc | naive eps |", "|---|---|---|---|---|---|"]
        for name, r in dp_runs:
            s = _short(r)
            lines.append(
                f"| {name} | {s.get('noise_sigma')} | {s.get('clip_C')} | "
                f"{s['rounds']} | {s['final_acc']} | {s.get('naive_epsilon')} |"
            )

    # SecAgg section.
    secagg = root / "secagg_demo" / "metrics.json"
    if secagg.exists():
        data = json.loads(secagg.read_text())
        lines += [
            "",
            "## Secure aggregation skeleton",
            "",
            f"- Recovered sum matches true sum: **{data['acceptance_passes']}**",
            f"- L2 error: `{data['l2_error']:.2e}`",
            "- See [secagg_demo/REPORT.md](secagg_demo/REPORT.md).",
            "",
        ]

    # Acceptance gates.
    by_name = {n: _short(r) for n, r in runs}
    lines += [
        "",
        "## Acceptance gates (per docs/specifications.md)",
        "",
        "| Gate | Target | Observed | Pass |",
        "|---|---|---|---|",
    ]
    iid = by_name.get("fedavg_iid")
    if iid:
        target = 0.97
        ok = iid["best_acc"] >= target
        lines.append(
            f"| FedAvg IID best acc >= 0.97 | 0.97 | {iid['best_acc']} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )
    fa = by_name.get("fedavg_dirichlet_a0.1")
    fp = by_name.get("fedprox_dirichlet_a0.1_mu0.01")
    sc = by_name.get("scaffold_dirichlet_a0.1")
    if fa and fp:
        diff = fp["final_acc"] - fa["final_acc"]
        ok = diff >= 0.02
        lines.append(
            f"| FedProx beats FedAvg by >=2pp (Dir 0.1) | +0.02 | "
            f"{diff:+.4f} | {'PASS' if ok else 'FAIL'} |"
        )
    if fa and sc:
        fa90 = fa.get("round_to_0.90")
        sc90 = sc.get("round_to_0.90")
        if sc90 is not None and fa90 is not None:
            ok = sc90 < fa90
            obs = f"SCAFFOLD r{sc90} vs FedAvg r{fa90}"
        elif sc90 is not None and fa90 is None:
            ok = True
            obs = f"SCAFFOLD r{sc90}, FedAvg never reached 0.90"
        else:
            ok = False
            obs = f"FedAvg r{fa90}, SCAFFOLD never reached 0.90"
        lines.append(
            f"| SCAFFOLD reaches 0.90 faster than FedAvg (Dir 0.1) | < | {obs} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )

    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {root}/SUMMARY.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
