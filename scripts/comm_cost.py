"""Phase 4.2: exact communication cost per round per algorithm.

Counts the bytes a single client uploads per round for each algorithm,
from the actual model parameter shapes. This makes the "SCAFFOLD is ~2x
FedAvg" claim exact rather than qualitative.

    FedAvg / FedProx : upload w (the model weights).
    SCAFFOLD         : upload w AND the control variate c_k (same shape) -> 2x.
    FedAdam          : upload w (server holds m, v -- no extra client upload).
    FedPer           : upload only the shared body (w minus the head).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fl.algorithms.fedper import head_param_names
from fl.models.cnn import make_mnist_cnn


def main() -> None:
    model = make_mnist_cnn()
    sd = model.state_dict()
    float_params = {k: v for k, v in sd.items() if v.is_floating_point()}
    total_params = sum(v.numel() for v in float_params.values())
    bytes_w = total_params * 4  # float32

    head = head_param_names(model)
    head_params = sum(float_params[k].numel() for k in head if k in float_params)
    body_params = total_params - head_params

    rows = [
        ("FedAvg", bytes_w, "w"),
        ("FedProx", bytes_w, "w (proximal term is client-local, not communicated)"),
        ("SCAFFOLD", 2 * bytes_w, "w + control variate c_k (same shape) -> 2x"),
        ("FedAdam", bytes_w, "w (server-side m,v held on server, no client cost)"),
        ("FedPer", body_params * 4, "shared body only; per-client head stays local"),
    ]

    summary = {
        "total_float_params": total_params,
        "body_params": body_params,
        "head_params": head_params,
        "bytes_per_round_upload": {name: b for name, b, _ in rows},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/comm_cost.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Communication cost per round (Phase 4.2)",
        "",
        f"Model: small MNIST CNN, {total_params:,} float parameters "
        f"({bytes_w:,} bytes = {bytes_w/1024:.1f} KiB as float32 per full model).",
        "",
        "Bytes a single client uploads per round:",
        "",
        "| Algorithm | Upload/round (bytes) | Relative to FedAvg | What is sent |",
        "|---|---|---|---|",
    ]
    for name, b, what in rows:
        lines.append(f"| {name} | {b:,} | {b/bytes_w:.2f}x | {what} |")
    lines += [
        "",
        "## Takeaways",
        "",
        "- **SCAFFOLD costs exactly 2x FedAvg per round** -- it ships the",
        "  control variate `c_k` (same shape as the model) alongside the",
        "  weights. This is the price of its drift correction; it pays off",
        "  only when the reduction in rounds-to-target more than offsets the",
        "  doubled per-round payload (and, per the K=10 vs K=100 finding, only",
        "  when there are enough clients for the variates to be useful).",
        "- **FedProx is 1x** -- the proximal term is computed locally from the",
        "  broadcast weights; nothing extra is uploaded.",
        "- **FedAdam is 1x** on the client -- the optimizer state (m, v) lives",
        "  on the server.",
        f"- **FedPer is {body_params*4/bytes_w:.2f}x** -- it never uploads the",
        "  classifier head, so it is strictly cheaper than FedAvg as well as",
        "  more private (the head, which is closest to the labels, never leaves",
        "  the client).",
        "",
    ]
    Path("results/COMM_COST_REPORT.md").write_text("\n".join(lines))
    print("wrote results/COMM_COST_REPORT.md and comm_cost.json")
    for name, b, _ in rows:
        print(f"  {name:10s} {b:>10,} bytes/round  ({b/bytes_w:.2f}x)")


if __name__ == "__main__":
    main()
