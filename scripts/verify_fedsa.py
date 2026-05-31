"""Isolate FedSA-LoRA's real failure root cause -- evidence, not a guess.

Earlier I claimed FedSA-LoRA fails "because B is zero-initialised + small
scale". That was a hypothesis, never tested. This script isolates it by
sweeping the two factors that the hypothesis implicates -- training budget
(rounds x local epochs) and LoRA rank -- and checking whether FedSA's
per-client accuracy approaches/beats FedIT when given more of them. If it
does, the failure was under-training (fixable, scale-dependent as claimed);
if it stays flat regardless, the failure is structural and the earlier
explanation was wrong.

Small AG News subset, DistilBERT, label-skew(2), 5 clients. Offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import scripts.run_fedlora as R
import torch
from fl.datasets.mnist_partition import label_skew


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R.TRAIN_PER_CLIENT = 500
    R.TEST_SIZE = 2000
    train, test = R._load_tokenized()
    labels = [int(train[i][2]) for i in range(len(train))]
    tlabels = [int(test[i][2]) for i in range(len(test))]

    # Factor grid: (rounds, local_epochs, rank). Baseline is the Phase-10 setting.
    configs = [
        {"rounds": 20, "epochs": 1, "rank": 8, "tag": "baseline"},
        {"rounds": 40, "epochs": 1, "rank": 8, "tag": "2x_rounds"},
        {"rounds": 20, "epochs": 3, "rank": 8, "tag": "3x_epochs"},
        {"rounds": 20, "epochs": 1, "rank": 16, "tag": "2x_rank"},
        {"rounds": 40, "epochs": 3, "rank": 16, "tag": "all_up"},
    ]
    rows = []
    for cfg in configs:
        R.ROUNDS = cfg["rounds"]; R.LOCAL_EPOCHS = cfg["epochs"]; R.RANK = cfg["rank"]
        R.NUM_CLIENTS = 5
        trp = label_skew(labels, 5, 2, seed=0)
        tep = label_skew(tlabels, 5, 2, seed=0)
        print(f"== {cfg['tag']}: rounds={cfg['rounds']} E={cfg['epochs']} r={cfg['rank']} ==", flush=True)
        fedsa = R.run_federated("fedsa", train, test, trp, device,
                                eval_per_client=True, per_client_test_parts=tep)
        fedit = R.run_federated("fedit", train, test, trp, device,
                                eval_per_client=True, per_client_test_parts=tep)
        d = (fedsa["per_client_mean"] or 0) - (fedit["per_client_mean"] or 0)
        rows.append({**cfg, "fedsa_pc": fedsa["per_client_mean"],
                     "fedit_pc": fedit["per_client_mean"], "delta": d})
        print(f"   FedSA pc={fedsa['per_client_mean']:.4f} FedIT pc={fedit['per_client_mean']:.4f} "
              f"delta={d*100:+.2f}pp", flush=True)

    # Does FedSA's gap to FedIT shrink (toward >=0) as budget/rank grows?
    base_delta = rows[0]["delta"]
    best_delta = max(r["delta"] for r in rows)
    improves = best_delta > base_delta + 0.02   # gap meaningfully closes
    fedsa_wins = best_delta >= 0.01

    out = Path("results/fedsa_verify"); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "rows": rows, "base_delta": base_delta, "best_delta": best_delta,
        "gap_closes_with_budget": bool(improves), "fedsa_ever_wins": bool(fedsa_wins),
    }, indent=2))

    lines = [
        "# FedSA-LoRA failure root-cause isolation",
        "",
        "Sweeping the factors the 'under-training / zero-init B' hypothesis",
        "implicates (rounds, local epochs, LoRA rank) to see whether FedSA's",
        "per-client gap to FedIT closes when given more budget.",
        "",
        "| Config | rounds | E | rank | FedSA pc | FedIT pc | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['rounds']} | {r['epochs']} | {r['rank']} | "
                     f"{r['fedsa_pc']:.4f} | {r['fedit_pc']:.4f} | {r['delta']*100:+.2f}pp |")
    lines += [
        "",
        f"Baseline delta = {base_delta*100:+.2f}pp; best delta over the grid = "
        f"{best_delta*100:+.2f}pp.",
        "",
        ("**Conclusion: the gap CLOSES as budget/rank grows** -- evidence that the "
         "FedSA-LoRA shortfall is under-training (zero-init B needs more steps to "
         "specialise), consistent with the claim that its advantage is "
         "scale-dependent. The mechanism is sound; the toy budget was the limiter."
         if improves else
         "**Conclusion: the gap does NOT close with more budget/rank** -- so the "
         "earlier 'under-training' explanation is NOT supported. FedSA-LoRA simply "
         "does not help at this model scale (DistilBERT/AG-News); the honest "
         "statement is 'no benefit here', and a definitive test would need the "
         "larger LLMs the FedSA-LoRA paper uses.") +
        (f" FedSA also reaches delta>=+1pp somewhere in the grid." if fedsa_wins else ""),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nbase_delta={base_delta*100:+.2f}pp best_delta={best_delta*100:+.2f}pp "
          f"gap_closes={improves} fedsa_wins={fedsa_wins}")
    print("saved results/fedsa_verify/")


if __name__ == "__main__":
    main()
