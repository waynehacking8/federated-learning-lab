"""Toy SecAgg demonstration with 3 clients.

Shows that the server-side sum recovered from peer-aggregated shares
equals the true sum of client updates, while individual peer sums
look like noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from privacy.secagg import aggregate_shares, simulate_secagg_round, split_update


def main() -> None:
    torch.manual_seed(0)
    out = Path("results/secagg_demo")
    out.mkdir(parents=True, exist_ok=True)

    # Three clients, each with a small "update" vector.
    updates = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([0.5, -1.0, 2.5, 0.0]),
        torch.tensor([-2.0, 1.0, 0.5, 3.0]),
    ]
    true_sum = torch.zeros(4)
    for u in updates:
        true_sum = true_sum + u

    recovered = simulate_secagg_round(updates, seed=42)

    # Per-peer aggregated shares -- these are what the server actually sees
    # if it were eavesdropping on a single peer. They should look like noise.
    n = len(updates)
    all_shares = [split_update(u, num_peers=n, seed=42 + k) for k, u in enumerate(updates)]
    peer_sums = [
        aggregate_shares([all_shares[k][j] for k in range(n)]) for j in range(n)
    ]

    lines = [
        "# SecAgg skeleton demo",
        "",
        "Three clients, each holding a 4-d update vector. Each client",
        "splits its update into 3 additive shares and routes share `j`",
        "to peer `j`. The server then sums the 3 peer-aggregated shares.",
        "",
        "## Inputs",
        "",
        "| Client | Update |",
        "|---|---|",
        *[f"| {i} | {u.tolist()} |" for i, u in enumerate(updates)],
        "",
        f"**True sum**: `{true_sum.tolist()}`",
        "",
        "## Per-peer aggregated shares (what a single eavesdropping peer sees)",
        "",
        "These should not reveal anything about the individual updates.",
        "",
        "| Peer | Aggregated share |",
        "|---|---|",
        *[f"| {j} | {[round(x, 3) for x in s.tolist()]} |" for j, s in enumerate(peer_sums)],
        "",
        "## Server-side recovered sum",
        "",
        f"- Recovered: `{[round(x, 4) for x in recovered.tolist()]}`",
        f"- True:      `{true_sum.tolist()}`",
        f"- L2 error:  `{float((recovered - true_sum).norm()):.2e}`",
        "",
        "## Acceptance",
        "",
        f"- Server-side recovered sum equals true sum within float tolerance: "
        f"**{torch.allclose(recovered, true_sum, atol=1e-6)}**",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))

    summary = {
        "true_sum": true_sum.tolist(),
        "recovered": recovered.tolist(),
        "l2_error": float((recovered - true_sum).norm()),
        "peer_sums": [s.tolist() for s in peer_sums],
        "acceptance_passes": bool(torch.allclose(recovered, true_sum, atol=1e-6)),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"SecAgg demo: true={true_sum.tolist()} recovered={[round(x, 6) for x in recovered.tolist()]}")
    print(f"             L2 error = {float((recovered - true_sum).norm()):.2e}")
    print(f"saved {out}/")


if __name__ == "__main__":
    main()
