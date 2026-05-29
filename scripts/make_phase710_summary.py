"""Roll up Phases 7-10 + ablations into results/PHASES_7_10_SUMMARY.md.

Reads each phase's metrics.json (when present) and produces one table of
gates with PASS/FAIL and the honest notes. Safe to run at any time; skips
phases whose results are not yet on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def _load(p):
    fp = R / p / "metrics.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def main() -> None:
    lines = [
        "# Phases 7-10 + ablations -- summary",
        "",
        "Personalized FL, Byzantine robustness + gradient leakage, server-side",
        "adaptive optimizer, and federated LoRA. Every gate is reported with its",
        "honest PASS/FAIL; FAILs carry a mechanistic reason (see "
        "`docs/design-decisions.md` D14-D16 and `docs/results-validation.md`).",
        "",
        "| Phase | Experiment | Gate | Result |",
        "|---|---|---|---|",
    ]

    # Phase 7: FedPer.
    for tag, name in [("Dir(0.1)", "fedper_vs_fedavg"), ("label_skew(3)", "fedper_labelskew3")]:
        d = _load(name)
        if d:
            g = "PASS" if d["gate_pass"] else "FAIL"
            lines.append(f"| 7 FedPer ({tag}) | per-client >= FedAvg+3pp | {g} | "
                         f"FedPer {d['fedper']['final_per_client']:.4f} vs FedAvg "
                         f"{d['fedavg']['final_per_client']:.4f} "
                         f"({d['per_client_delta']*100:+.2f}pp) |")

    # Phase 8: robust + DLG.
    d = _load("robust")
    if d:
        g = "PASS" if d["gate_pass"] else "FAIL"
        ua = d["under_attack"]
        lines.append(f"| 8 Robust agg | median/Krum within 5pp; FedAvg -20pp | {g} | "
                     f"FedAvg drop {ua['fedavg']['drop_vs_baseline']*100:+.1f}pp, "
                     f"median {ua['median']['drop_vs_baseline']*100:+.1f}pp, "
                     f"krum {ua['krum']['drop_vs_baseline']*100:+.1f}pp |")
    d = _load("dlg")
    if d:
        g = "PASS" if d["leak_demonstrated"] else "FAIL"
        lines.append(f"| 8 DLG | no-DP MSE << DP MSE | {g} | "
                     f"no-DP {d['mse_no_dp']:.4f} vs DP {d['mse_dp']:.2e} |")

    # Phase 9: FedAdam.
    d = _load("fedopt_comparison")
    if d:
        g = "PASS" if d["gate_pass"] else "FAIL"
        fa = d["results"]["fedavg"]; fad = d["results"]["fedadam"]
        lines.append(f"| 9 FedAdam | fewer rounds OR +1pp | {g} | "
                     f"FedAvg {fa['final']:.4f}/r{fa['rounds_to_target']} vs "
                     f"FedAdam {fad['final']:.4f}/r{fad['rounds_to_target']} |")

    # Phase 10: FedLoRA.
    d = _load("fedlora")
    if d:
        gi = "PASS" if d["iid_gate_pass"] else "FAIL"
        gp = "PASS" if d["perso_gate_pass"] else "FAIL"
        lines.append(f"| 10 FedIT (IID) | >= 90% centralized | {gi} | "
                     f"FedIT {d['fedit_iid_final']:.4f} vs centralized {d['centralized_acc']:.4f} |")
        lines.append(f"| 10 FedSA-LoRA | per-client +1pp, half payload | {gp} | "
                     f"delta {d['personalization_delta']*100:+.2f}pp, "
                     f"adapter payload {d.get('adapter_payload_ratio_fedsa_over_fedit', 0):.2f}x |")

    lines += [
        "",
        "## Honest-FAIL notes",
        "",
        "- **FedPer (Dir / label_skew on MNIST)**: per-client accuracy",
        "  saturates (~0.99) because each client's own-distribution test slice",
        "  is trivial on MNIST -- no 3pp headroom. The FedPer global-metric",
        "  collapse confirms the head specialized (mechanism works); MNIST is",
        "  simply too easy to expose the personalization gain. See D2, D16.",
        "- **FedSA-LoRA**: correctly halves adapter payload and implements the",
        "  A-shared/B-local mechanism, but does not beat FedIT on per-client",
        "  accuracy in this toy regime (zero-init B + little data + few rounds).",
        "  Reported as FAIL rather than tuned to PASS. See D14.",
        "",
        "## Communication cost (Phase 4.2)",
        "",
        "See `results/COMM_COST_REPORT.md`: SCAFFOLD is exactly 2.00x FedAvg",
        "per round; FedProx and FedAdam are 1.00x; FedPer is 0.99x.",
        "",
    ]
    (R / "PHASES_7_10_SUMMARY.md").write_text("\n".join(lines))
    print("wrote results/PHASES_7_10_SUMMARY.md")


if __name__ == "__main__":
    main()
