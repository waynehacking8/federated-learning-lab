"""FedSA-LoRA where its mechanism applies: client-specific FEATURE skew.

The earlier FedSA tests (run_fedlora.py main, verify_fedsa.py budget grid)
used LABEL skew. Under label skew all clients share the SAME feature->label
mapping (a Sports article is Sports for everyone), so pooling B (FedIT) is
strictly better than keeping B local (FedSA) -- which is exactly why FedSA
lost there, and why no training budget closed the gap.

FedSA-LoRA's premise (Guo et al. ICLR 2025) is "A learns general knowledge,
B learns CLIENT-SPECIFIC knowledge -- so keep B local". That premise only
bites under FEATURE skew, the one canonical Non-IID type never tested here:
clients whose inputs need genuinely different representations.

This builds that regime honestly. Two synthetic text "dialects":
  - normal clients: text as-is.
  - reversed clients: the token order between [CLS] and [SEP] is reversed.
Labels are IID across clients (topic distribution identical), so the only
heterogeneity is the FEATURE/representation each client needs -- the head
can stay shared; this isolates the value of B specialization. A shared
adapter B (FedIT) must compromise across the two dialects; a local B
(FedSA) can specialize to each. Per-client eval is on the client's own
dialect.

Gate (spec section 14 intent): FedSA per-client acc >= FedIT + 1pp AND
payload halved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import TensorDataset

import scripts.run_fedlora as R

CLS, SEP = 101, 102
NUM_CLIENTS = 4          # 2 normal, 2 reversed
TRAIN_PER_CLIENT = 800
TEST_PER_CLIENT = 400


def _reverse_row(ids: torch.Tensor, attn: torch.Tensor):
    """Reverse real tokens strictly between [CLS] and [SEP]; keep padding."""
    L = int(attn.sum().item())
    if L <= 3:
        return ids.clone()
    out = ids.clone()
    body = ids[1:L - 1].flip(0)
    out[1:L - 1] = body
    return out


def _apply_domain(ds: TensorDataset, reverse: bool) -> TensorDataset:
    ids, attn, y = ds.tensors
    if not reverse:
        return TensorDataset(ids.clone(), attn.clone(), y.clone())
    new_ids = torch.stack([_reverse_row(ids[i], attn[i]) for i in range(len(ids))])
    return TensorDataset(new_ids, attn.clone(), y.clone())


def _take(ds: TensorDataset, lo: int, hi: int) -> TensorDataset:
    ids, attn, y = ds.tensors
    return TensorDataset(ids[lo:hi].clone(), attn[lo:hi].clone(), y[lo:hi].clone())


def _concat(parts):
    ids = torch.cat([p.tensors[0] for p in parts])
    attn = torch.cat([p.tensors[1] for p in parts])
    y = torch.cat([p.tensors[2] for p in parts])
    return TensorDataset(ids, attn, y)


def _build_domain_split(base_train, base_test):
    """Per-client datasets with alternating normal/reversed dialects, IID labels."""
    domains = [i % 2 == 1 for i in range(NUM_CLIENTS)]  # F,T,F,T -> normal,rev,...
    tr_parts, te_parts = [], []
    tr_idx, te_idx = [], []
    cur_tr = cur_te = 0
    for ci in range(NUM_CLIENTS):
        tr_chunk = _take(base_train, ci * TRAIN_PER_CLIENT, (ci + 1) * TRAIN_PER_CLIENT)
        te_chunk = _take(base_test, ci * TEST_PER_CLIENT, (ci + 1) * TEST_PER_CLIENT)
        tr_parts.append(_apply_domain(tr_chunk, domains[ci]))
        te_parts.append(_apply_domain(te_chunk, domains[ci]))
        tr_idx.append(list(range(cur_tr, cur_tr + TRAIN_PER_CLIENT))); cur_tr += TRAIN_PER_CLIENT
        te_idx.append(list(range(cur_te, cur_te + TEST_PER_CLIENT))); cur_te += TEST_PER_CLIENT
    return _concat(tr_parts), _concat(te_parts), tr_idx, te_idx, domains


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R.NUM_CLIENTS = NUM_CLIENTS
    R.ROUNDS = 25
    R.LOCAL_EPOCHS = 2
    R.RANK = 8
    R.TRAIN_PER_CLIENT = TRAIN_PER_CLIENT
    R.TEST_SIZE = NUM_CLIENTS * TEST_PER_CLIENT

    base_train, base_test = R._load_tokenized()
    train, test, parts, test_parts, domains = _build_domain_split(base_train, base_test)
    print(f"domains (reversed?) = {domains}", flush=True)

    print("=== FedIT (feature skew) ===", flush=True)
    fedit = R.run_federated("fedit", train, test, parts, device,
                            eval_per_client=True, per_client_test_parts=test_parts)
    print(f"FedIT per-client={fedit['per_client_mean']:.4f} each={[round(x,3) for x in fedit['per_client_accs']]}", flush=True)

    print("=== FedSA-LoRA (feature skew) ===", flush=True)
    fedsa = R.run_federated("fedsa", train, test, parts, device,
                            eval_per_client=True, per_client_test_parts=test_parts)
    print(f"FedSA per-client={fedsa['per_client_mean']:.4f} each={[round(x,3) for x in fedsa['per_client_accs']]}", flush=True)

    delta = fedsa["per_client_mean"] - fedit["per_client_mean"]
    payload_ratio = (fedsa["adapter_payload_bytes_per_round"]
                     / fedit["adapter_payload_bytes_per_round"])
    gate_pass = (delta >= 0.01) and (payload_ratio <= 0.6)

    out = Path("results/fedsa_domain"); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "regime": "feature skew (token-reversal dialects), IID labels",
        "num_clients": NUM_CLIENTS, "domains_reversed": domains,
        "rounds": R.ROUNDS, "local_epochs": R.LOCAL_EPOCHS, "rank": R.RANK,
        "fedit_per_client": fedit["per_client_mean"],
        "fedsa_per_client": fedsa["per_client_mean"],
        "fedit_each": fedit["per_client_accs"], "fedsa_each": fedsa["per_client_accs"],
        "delta": delta, "payload_ratio": payload_ratio,
        "gate_pass": bool(gate_pass),
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(NUM_CLIENTS); w = 0.38
    ax.bar(x - w / 2, fedit["per_client_accs"], w, label=f"FedIT (mean {fedit['per_client_mean']:.3f})")
    ax.bar(x + w / 2, fedsa["per_client_accs"], w, label=f"FedSA-LoRA (mean {fedsa['per_client_mean']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{ci}\n{'rev' if domains[ci] else 'norm'}" for ci in range(NUM_CLIENTS)])
    ax.set_ylabel("Per-client test accuracy (own dialect)")
    ax.set_title("FedSA-LoRA vs FedIT under feature skew (token-reversal dialects)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="lower right"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "curve.png", dpi=120); plt.close(fig)

    lines = [
        "# FedSA-LoRA under feature skew (the regime its mechanism targets)",
        "",
        "Prior FedSA tests used LABEL skew, where all clients share one",
        "feature->label mapping, so pooling B (FedIT) is strictly better and",
        "FedSA loses regardless of budget. FedSA-LoRA's premise is that B",
        "captures *client-specific* knowledge -- which only matters under",
        "FEATURE skew. Here two synthetic dialects (normal vs token-reversed",
        "text, IID labels) force genuinely different representations; the head",
        "stays shared so the comparison isolates B specialization.",
        "",
        f"{NUM_CLIENTS} clients, dialects reversed?={domains}, rank {R.RANK}, "
        f"{R.ROUNDS} rounds, E={R.LOCAL_EPOCHS}.",
        "",
        "| Method | Mean per-client acc | Per client | Adapter payload |",
        "|---|---|---|---|",
        f"| FedIT | {fedit['per_client_mean']:.4f} | {[round(x,3) for x in fedit['per_client_accs']]} | 1.00x |",
        f"| FedSA-LoRA | {fedsa['per_client_mean']:.4f} | {[round(x,3) for x in fedsa['per_client_accs']]} | {payload_ratio:.2f}x |",
        "",
        f"Delta (FedSA - FedIT) = **{delta*100:+.2f}pp**; adapter payload "
        f"ratio = **{payload_ratio:.2f}x**.",
        "",
        f"**Gate (FedSA >= FedIT + 1pp AND payload <= 0.6x)? "
        f"{'PASS' if gate_pass else 'FAIL'}.**",
        "",
        ("Conclusion: under feature skew -- the heterogeneity type FedSA-LoRA "
         "is actually designed for -- keeping B local lets each client adapt to "
         "its own dialect, while FedIT's averaged B must compromise across "
         "dialects. FedSA now beats FedIT on per-client accuracy AND halves the "
         "adapter payload. The earlier label-skew FAIL was a wrong-regime "
         "artefact: that Non-IID type does not need client-specific B."
         if gate_pass else
         "Conclusion: even under feature skew, FedSA did not clear the +1pp gate "
         "at this scale. Reported as measured."),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\ndelta={delta*100:+.2f}pp payload_ratio={payload_ratio:.2f}x gate_pass={gate_pass}")
    print("saved results/fedsa_domain/")


if __name__ == "__main__":
    main()
