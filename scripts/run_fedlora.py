"""Phase 10 experiment: FedLoRA (FedIT vs FedSA-LoRA) on AG News.

Frozen DistilBERT + LoRA adapters on attention q/v projections. Only
adapter weights (+ classifier head) are federated. Two strategies:
    - FedIT:      average A and B (+ classifier).
    - FedSA-LoRA: average only A (+ classifier); keep B local per client.

Gates:
    - IID: FedIT reaches >= 90% of centralized-LoRA accuracy within 20 rounds.
    - Dir(alpha): FedSA-LoRA mean per-client acc >= FedIT + 1pp AND payload
      per round ~ half (only A shared).

Runs OFFLINE from the HF cache (distilbert-base-uncased, ag_news).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

BASE = "distilbert-base-uncased"
NUM_LABELS = 4
RANK = 8
ROUNDS = 20
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
LOCAL_LR = 5e-4
BATCH = 16
SEED = 0
MAX_LEN = 64
TRAIN_PER_CLIENT = 600   # keep wall-clock reasonable
TEST_SIZE = 2000
RESULTS = Path("results")


def _set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenized():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = load_dataset("ag_news")

    def encode(split, n):
        rows = ds[split].shuffle(seed=SEED).select(range(n))
        enc = tok(rows["text"], truncation=True, padding="max_length",
                  max_length=MAX_LEN, return_tensors="pt")
        y = torch.tensor(rows["label"])
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train = encode("train", TRAIN_PER_CLIENT * NUM_CLIENTS)
    test = encode("test", TEST_SIZE)
    return train, test


def _make_peft_model(device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=NUM_LABELS)
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=["q_lin", "v_lin"],
                     lora_dropout=0.0, task_type="SEQ_CLS")
    model = get_peft_model(base, cfg).to(device)
    return model


def _trainable_keys(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _get_shared_state(model, keys):
    sd = dict(model.named_parameters())
    return {k: sd[k].detach().cpu().clone() for k in keys}


def _load_shared_state(model, state):
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].device))


def _label_skew_parts(labels, num_clients, classes_per_client, seed):
    from fl.datasets.mnist_partition import label_skew
    return label_skew(list(labels), num_clients, classes_per_client, seed=seed)


def _train_local(model, loader, device, epochs):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LOCAL_LR)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=ids, attention_mask=mask).logits
            loss = lossf(out, y)
            loss.backward()
            opt.step()


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        pred = model(input_ids=ids, attention_mask=mask).logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


def _payload_bytes(state, keys):
    return sum(state[k].numel() * 4 for k in keys if k in state)  # float32 bytes


def run_federated(strategy, train, test, parts, device, eval_per_client=False,
                  per_client_test_parts=None):
    """strategy in {'fedit','fedsa'}. Returns history + payload + per-client accs.

    per_client_test_parts: optional list of index lists into `test`, one per
    client, giving each client's OWN-distribution test slice. The personalized
    metric evaluates each client's model (with its own B for FedSA) on that
    client's own slice -- which is the regime where keeping B local helps.
    """
    from fl.algorithms.fedlora import FedITAggregator, FedSALoRAAggregator
    _set_seeds(SEED)

    server_model = _make_peft_model(device)
    keys = _trainable_keys(server_model)
    agg = FedITAggregator() if strategy == "fedit" else FedSALoRAAggregator()
    shared_keys = agg.shared_keys(keys)

    # One persistent model per client (holds client-local B for FedSA).
    client_models = [_make_peft_model(device) for _ in parts]
    client_loaders = [
        DataLoader(Subset(train, idx), batch_size=BATCH, shuffle=True)
        for idx in parts
    ]
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    global_shared = _get_shared_state(server_model, shared_keys)
    history = []
    payload_per_round = _payload_bytes(_get_shared_state(server_model, keys), shared_keys) * len(parts)
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        client_states = []
        sizes = []
        for cm, loader, idx in zip(client_models, client_loaders, parts):
            if not idx:
                continue
            # Load the shared params from the server; keep this client's own
            # non-shared params (e.g. B for FedSA) intact.
            _load_shared_state(cm, global_shared)
            _train_local(cm, loader, device, LOCAL_EPOCHS)
            # Report ALL trainable params; aggregator picks the shared subset.
            client_states.append(_get_shared_state(cm, keys))
            sizes.append(len(idx))
        new_shared = agg.aggregate(client_states, sizes)
        global_shared = new_shared
        # Push shared params into the server model for the global metric.
        _load_shared_state(server_model, global_shared)
        global_acc = _evaluate(server_model, test_loader, device)
        history.append({"round": r, "global_acc": global_acc})
        if r % 5 == 0 or r == ROUNDS:
            print(f"[{strategy}] round {r:2d} global_acc={global_acc:.4f}", flush=True)

    # Per-client accuracy on the DEPLOYED model: load the final aggregated
    # shared params into each client, then evaluate on that client's
    # OWN-distribution test slice. Crucially we DO reload shared params first
    # so we measure what the client actually deploys after receiving the
    # aggregation, NOT its post-local-training copy:
    #   FedSA: shared params = A only, so the client keeps its own B + head
    #          (its legitimate client-specific personalization).
    #   FedIT: shared params = A, B AND head, so every client deploys the one
    #          shared model (no personalization -- which is the point).
    # Skipping this reload is an eval leak: the client's last local step would
    # let even a FedIT client silently re-specialize its shared head, hiding
    # the architectural difference (same bug class fixed in verify_fedper.py).
    per_client = []
    if eval_per_client and per_client_test_parts is not None:
        for cm, tidx in zip(client_models, per_client_test_parts):
            if not tidx:
                continue
            _load_shared_state(cm, global_shared)
            cl_loader = DataLoader(Subset(test, tidx), batch_size=64, shuffle=False)
            per_client.append(_evaluate(cm, cl_loader, device))

    # Adapter-only payload (excludes the classifier head) -- the figure the
    # FedSA-vs-FedIT "half the payload" claim is actually about.
    adapter_keys = [k for k in shared_keys if "lora" in k]
    full_state = _get_shared_state(server_model, keys)
    adapter_payload = _payload_bytes(full_state, adapter_keys) * len(parts)

    wall = time.time() - t0
    return {"history": history, "final_global": history[-1]["global_acc"],
            "payload_bytes_per_round": payload_per_round,
            "adapter_payload_bytes_per_round": adapter_payload,
            "per_client_mean": float(np.mean(per_client)) if per_client else None,
            "per_client_accs": per_client,
            "wall_seconds": wall}


def run_centralized(train, test, device, max_epochs=30, patience=3):
    """Centralized LoRA fine-tune as the reference CEILING.

    Trained to convergence (early-stop on test-accuracy plateau), not to a
    step-count matched to the federated budget. A ceiling must be the best
    achievable centrally -- otherwise federated could spuriously appear to
    beat it (which is logically impossible). Returns the best test accuracy
    seen, so it is a genuine upper bound for the FedIT >= 90%-of-ceiling gate.
    """
    _set_seeds(SEED)
    model = _make_peft_model(device)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    best = 0.0
    stale = 0
    for ep in range(max_epochs):
        _train_local(model, loader, device, epochs=1)
        acc = _evaluate(model, test_loader, device)
        if acc > best + 1e-4:
            best = acc
            stale = 0
        else:
            stale += 1
        print(f"[centralized] epoch {ep+1} acc={acc:.4f} best={best:.4f}", flush=True)
        if stale >= patience:
            break
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = _load_tokenized()
    labels = [int(train[i][2]) for i in range(len(train))]

    # IID partition for the FedIT-vs-centralized gate.
    from fl.datasets.mnist_partition import iid
    iid_parts = iid(labels, NUM_CLIENTS, seed=SEED)

    print("=== centralized LoRA reference ===", flush=True)
    cen_acc = run_centralized(train, test, device)
    print(f"centralized LoRA acc={cen_acc:.4f}", flush=True)

    print("\n=== FedIT (IID) ===", flush=True)
    fedit_iid = run_federated("fedit", train, test, iid_parts, device)

    # Non-IID (label-skew) for the FedSA-vs-FedIT personalization gate.
    # Partition the TEST set the same way so each client has an own-distribution
    # test slice (the personalized metric is each client's model on its own slice).
    skew_parts = _label_skew_parts(labels, NUM_CLIENTS, 2, SEED)
    test_labels = [int(test[i][2]) for i in range(len(test))]
    test_parts = _label_skew_parts(test_labels, NUM_CLIENTS, 2, SEED)
    print("\n=== FedIT (label-skew) ===", flush=True)
    fedit_skew = run_federated("fedit", train, test, skew_parts, device,
                               eval_per_client=True, per_client_test_parts=test_parts)
    print("\n=== FedSA-LoRA (label-skew) ===", flush=True)
    fedsa_skew = run_federated("fedsa", train, test, skew_parts, device,
                               eval_per_client=True, per_client_test_parts=test_parts)

    # Gates.
    iid_gate = fedit_iid["final_global"] >= 0.9 * cen_acc
    # Payload ratio on the ADAPTER bytes (what FedSA actually halves), not the
    # classifier head (which both share identically).
    payload_ratio = (fedsa_skew["adapter_payload_bytes_per_round"]
                     / fedit_skew["adapter_payload_bytes_per_round"])
    perso_delta = (fedsa_skew["per_client_mean"] or 0) - (fedit_skew["per_client_mean"] or 0)
    perso_gate = perso_delta >= 0.01 and payload_ratio <= 0.6

    out = RESULTS / "fedlora"; out.mkdir(parents=True, exist_ok=True)
    summary = {
        "base": BASE, "rank": RANK, "rounds": ROUNDS, "num_clients": NUM_CLIENTS,
        "centralized_acc": cen_acc,
        "fedit_iid_final": fedit_iid["final_global"],
        "fedit_skew": {k: fedit_skew[k] for k in ("final_global", "per_client_mean", "payload_bytes_per_round", "adapter_payload_bytes_per_round")},
        "fedsa_skew": {k: fedsa_skew[k] for k in ("final_global", "per_client_mean", "payload_bytes_per_round", "adapter_payload_bytes_per_round")},
        "adapter_payload_ratio_fedsa_over_fedit": payload_ratio,
        "personalization_delta": perso_delta,
        "iid_gate_pass": bool(iid_gate), "perso_gate_pass": bool(perso_gate),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    # Communication vs accuracy: cumulative bytes on x, global acc on y.
    for name, res in [("FedIT (label-skew)", fedit_skew), ("FedSA-LoRA (label-skew)", fedsa_skew)]:
        bytes_cum = [res["payload_bytes_per_round"] * h["round"] / 1e6 for h in res["history"]]
        accs = [h["global_acc"] for h in res["history"]]
        ax.plot(bytes_cum, accs, marker=".", label=name)
    ax.set_xlabel("cumulative payload (MB, adapters shared)")
    ax.set_ylabel("global test accuracy")
    ax.set_title("FedLoRA: communication vs accuracy (AG News, DistilBERT)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("results/fedlora_communication_vs_accuracy.png", dpi=120); plt.close(fig)

    lines = [
        "# FedLoRA -- federated PEFT (Phase 10)",
        "",
        f"Frozen {BASE} + LoRA (rank {RANK}) on q/v attention; AG News "
        f"(4 classes), {NUM_CLIENTS} clients, {ROUNDS} rounds.",
        "",
        f"Centralized-LoRA reference accuracy: **{cen_acc:.4f}**.",
        "",
        "## IID gate (FedIT reaches >= 90% of centralized)",
        "",
        f"- FedIT IID final: {fedit_iid['final_global']:.4f}",
        f"- 90% of centralized: {0.9*cen_acc:.4f}",
        f"- **Gate: {'PASS' if iid_gate else 'FAIL'}**",
        "",
        "## Personalization gate (label-skew): FedSA-LoRA vs FedIT",
        "",
        "Per-client accuracy = each client's model (with its own B for FedSA)",
        "on that client's OWN-distribution test slice.",
        "",
        "| Strategy | Global acc | Mean per-client acc | Adapter payload/round (MB) |",
        "|---|---|---|---|",
        f"| FedIT | {fedit_skew['final_global']:.4f} | {fedit_skew['per_client_mean']:.4f} | "
        f"{fedit_skew['adapter_payload_bytes_per_round']/1e6:.3f} |",
        f"| FedSA-LoRA | {fedsa_skew['final_global']:.4f} | {fedsa_skew['per_client_mean']:.4f} | "
        f"{fedsa_skew['adapter_payload_bytes_per_round']/1e6:.3f} |",
        "",
        f"- Per-client delta (FedSA - FedIT): {perso_delta*100:+.2f}pp",
        f"- Adapter payload ratio (FedSA / FedIT): {payload_ratio:.2f}",
        f"- **Gate (delta >= 1pp AND adapter payload <= 0.6x): {'PASS' if perso_gate else 'FAIL'}**",
        "",
        "![comm vs acc](../fedlora_communication_vs_accuracy.png)",
        "",
        "## Interpretation",
        "",
        "FedIT averages both A and B adapter matrices; FedSA-LoRA shares only",
        "A (general structure, per Guo et al. ICLR 2025) and keeps B",
        "(client-specific) local. The adapter payload is correctly halved",
        f"({payload_ratio:.2f}x).",
        "",
        ("**Honest result:** in this deliberately small regime (DistilBERT, "
         "5 clients, ~600 samples each, rank-8 LoRA, 20 rounds) FedSA-LoRA "
         "does NOT beat FedIT on per-client accuracy -- the personalization "
         "gate does not reproduce here." if not perso_gate else
         "FedSA-LoRA's local B specializes and it beats FedIT on per-client "
         "accuracy at half the adapter payload."),
        "",
        ("Why: LoRA initializes B=0, so the adapter is inert until B trains. "
         "When only A is averaged across clients whose B matrices have "
         "diverged, the averaged A no longer matches any client's B (the "
         "B*A product is what acts on activations), and with little local "
         "data + few rounds B cannot re-align. FedSA-LoRA's published gains "
         "use larger models, more data, and more rounds. This is consistent "
         "with the algorithms.md note that FedSA-LoRA is the newest and "
         "least stress-tested of the family. The mechanism (A shared, B "
         "local) and the payload halving are demonstrated correctly; the "
         "accuracy advantage is scale-dependent and is reported as such "
         "rather than tuned until it passes (see design-decisions D14)."
         if not perso_gate else
         "This is the same shared-body / private-head idea as FedPer "
         "(Phase 7), applied inside the LoRA adapter."),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\ncentralized={cen_acc:.4f} fedit_iid={fedit_iid['final_global']:.4f} "
          f"(IID gate {'PASS' if iid_gate else 'FAIL'})")
    print(f"FedSA per-client {fedsa_skew['per_client_mean']:.4f} vs FedIT "
          f"{fedit_skew['per_client_mean']:.4f} delta {perso_delta*100:+.2f}pp "
          f"payload_ratio {payload_ratio:.2f} (perso gate {'PASS' if perso_gate else 'FAIL'})")
    print("saved results/fedlora/ and results/fedlora_communication_vs_accuracy.png")


if __name__ == "__main__":
    main()
