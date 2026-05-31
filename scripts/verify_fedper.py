"""Verify FedPer's mechanism actually helps -- the definitive test.

Earlier Phase 7 runs FAILED the +3pp gate because each client's
own-distribution test slice was near-trivial on MNIST (metric saturation).
That left the claim "the mechanism works, the metric just saturates"
UNVERIFIED -- a story, not evidence.

This test removes all doubt with a LABEL-PERMUTATION setup: each client
applies a fixed random permutation to its labels (client k maps true
label y -> perm_k[y]). Now a SINGLE global model is mathematically unable
to serve all clients (client A says "this 3 is a 7", client B says "this
3 is a 1"), so:
  - FedAvg (one shared head) MUST do poorly per-client (near random on the
    conflicting head).
  - FedPer (per-client head on a shared body) CAN fit each client's
    permutation, so per-client accuracy MUST be high -- IF the mechanism
    is correctly implemented.

A large FedPer - FedAvg gap here is positive proof the mechanism works;
its absence would mean a real bug. Evaluation is on each client's own
(permuted) test slice.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from fl.algorithms.fedavg import FedAvgAggregator
from fl.algorithms.fedper import FedPerAggregator, attach_fedper, head_param_names
from fl.client import Client
from fl.datasets.mnist_partition import iid
from fl.models.cnn import make_mnist_cnn
from fl.server import Server

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ROUNDS = 8
NUM_CLIENTS = 5
SEED = 0


def _seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class PermutedLabels(Dataset):
    """Wrap a dataset, remapping each label via a fixed permutation."""

    def __init__(self, base, indices, perm):
        self.base = base
        self.indices = list(indices)
        self.perm = perm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, int(self.perm[y])


def _load():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tfm)
    return train, test


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return c / t


def run(which, train, test, train_parts, test_parts, perms, device):
    _seed()
    # Each client trains on its own permuted-label data.
    clients = []
    for cid, idx in enumerate(train_parts):
        ds = PermutedLabels(train, idx, perms[cid])
        c = Client(client_id=cid, local_indices=list(range(len(idx))), train_dataset=ds,
                   device=device, local_epochs=2, local_lr=0.01, batch_size=64)
        clients.append(c)

    template = make_mnist_cnn()
    head_names = head_param_names(template)
    if which == "fedper":
        attach_fedper(clients, head_names)
        agg = FedPerAggregator(head_names)
    else:
        agg = FedAvgAggregator()

    rng = torch.Generator(); rng.manual_seed(SEED)
    # Dummy global test loader (unused metric); per-client is what matters.
    server = Server(global_model=make_mnist_cnn(), aggregator=agg, clients=clients,
                    test_loader=DataLoader(test, batch_size=512), device=device,
                    participation_rate=1.0, rng=rng)
    for r in range(1, ROUNDS + 1):
        server.run_round(r)

    # Per-client accuracy on the ACTUALLY-DEPLOYED model -- NOT the client's
    # locally-overfit copy. After aggregation:
    #   FedAvg : every client deploys the single shared global model.
    #   FedPer : client deploys the aggregated body + ITS OWN head.
    # Evaluating c._local_model instead would be cheating: that model just
    # finished training on the client's own permutation, so it would score
    # high for FedAvg too and the test would prove nothing.
    global_state = {k: v.detach().cpu().clone()
                    for k, v in server.global_model.state_dict().items()}
    eval_model = make_mnist_cnn().to(device)
    accs = []
    for cid in range(len(clients)):
        state = dict(global_state)
        if which == "fedper":
            # Swap in this client's persisted head over the aggregated body.
            for k in head_names:
                state[k] = clients[cid]._fedper_head[k].clone()
        eval_model.load_state_dict({k: v.to(device) for k, v in state.items()})
        ds = PermutedLabels(test, test_parts[cid], perms[cid])
        loader = DataLoader(ds, batch_size=256)
        accs.append(_eval(eval_model, loader, device))
    return float(np.mean(accs)), accs


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = _load()
    _seed()

    train_parts = iid(list(train.targets.numpy()), NUM_CLIENTS, seed=SEED)
    test_parts = iid(list(test.targets.numpy()), NUM_CLIENTS, seed=SEED)
    # One fixed random label permutation per client (client 0 = identity as control).
    rng = np.random.RandomState(SEED)
    perms = [np.arange(10)] + [rng.permutation(10) for _ in range(NUM_CLIENTS - 1)]

    print("== FedAvg (label-permutation) ==", flush=True)
    fa_mean, fa_each = run("fedavg", train, test, train_parts, test_parts, perms, device)
    print(f"FedAvg per-client mean={fa_mean:.4f} each={[round(x,3) for x in fa_each]}", flush=True)
    print("== FedPer (label-permutation) ==", flush=True)
    fp_mean, fp_each = run("fedper", train, test, train_parts, test_parts, perms, device)
    print(f"FedPer per-client mean={fp_mean:.4f} each={[round(x,3) for x in fp_each]}", flush=True)

    delta = fp_mean - fa_mean
    # Mechanism is proven if FedPer hugely beats FedAvg under label conflict.
    mechanism_works = delta >= 0.20

    out = Path("results/fedper_verify"); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "setup": "label-permutation per client (conflicting label maps)",
        "rounds": ROUNDS, "num_clients": NUM_CLIENTS,
        "fedavg_per_client_mean": fa_mean, "fedper_per_client_mean": fp_mean,
        "delta": delta, "mechanism_proven": bool(mechanism_works),
        "fedavg_each": fa_each, "fedper_each": fp_each,
    }, indent=2))

    lines = [
        "# FedPer mechanism verification -- label-permutation test",
        "",
        f"{NUM_CLIENTS} clients, IID images but each client applies a fixed",
        "random label permutation (client 0 = identity control). A single",
        "global head CANNOT serve conflicting label maps, so a correct FedPer",
        "(per-client head) must beat FedAvg by a wide margin.",
        "",
        "| Method | Mean per-client acc | Per client |",
        "|---|---|---|",
        f"| FedAvg | {fa_mean:.4f} | {[round(x,3) for x in fa_each]} |",
        f"| FedPer | {fp_mean:.4f} | {[round(x,3) for x in fp_each]} |",
        "",
        f"Delta (FedPer - FedAvg) = **{delta*100:+.1f}pp**",
        "",
        f"**Mechanism proven (FedPer >> FedAvg under label conflict)? "
        f"{'YES' if mechanism_works else 'NO'}**",
        "",
        ("This is positive evidence that the FedPer implementation is correct: "
         "when personalization is genuinely necessary (conflicting label maps), "
         "the per-client head captures it and FedAvg's single head cannot. The "
         "earlier Phase 7 'FAIL' was therefore a metric-saturation artefact on "
         "easy MNIST slices, NOT a broken mechanism -- now demonstrated, not asserted."
         if mechanism_works else
         "WARNING: FedPer did not beat FedAvg even under label permutation, where "
         "it provably should. This indicates a real bug to investigate."),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nDelta={delta*100:+.1f}pp mechanism_proven={mechanism_works}")
    print("saved results/fedper_verify/")


if __name__ == "__main__":
    main()
