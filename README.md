# Federated Learning Lab

> A from-scratch implementation of canonical **federated learning**
> algorithms — FedAvg, FedProx, SCAFFOLD — with optional differential-
> privacy and secure-aggregation layers, evaluated on Non-IID
> partitions of MNIST.

Built as a self-study exercise to understand the algorithmic
machinery behind privacy-preserving distributed training. The repo
prioritizes **algorithmic correctness and clear measurement of
Non-IID degradation** over scale or production hardening.

---

## What this is

- A clean PyTorch implementation of **FedAvg** (McMahan 2017) on
  MNIST with configurable client count, local epochs, and
  participation rate.
- Multiple **Non-IID partitioning schemes**: label-skew, quantity-skew,
  Dirichlet-α.
- Implementations of two **drift-mitigation variants**: FedProx
  (proximal term) and SCAFFOLD (control variates).
- An optional **differential-privacy module** wrapping local SGD
  with per-sample gradient clipping and Gaussian noise (DP-SGD).
- A secure-aggregation skeleton (additive-secret-sharing primitive,
  not full SecAgg protocol).
- Convergence-comparison plots: exploitability vs Non-IID severity
  across all three algorithms.

## What this is NOT

- **Not a production framework.** No remote-procedure-call layer, no
  client authentication, no real network transport. All "clients"
  run in-process as separate state dictionaries.
- **Not a privacy audit.** The DP module follows DP-SGD's basic
  recipe but is not a certified privacy accountant. Production needs
  Opacus or TF-Privacy with rigorous ε accounting.
- **Not large-scale.** Single-machine simulation up to ~100 clients;
  beyond that the in-process model becomes the bottleneck.
- **Not aimed at LLM-scale fine-tuning.** A FedLoRA prototype on a
  small model is listed in the roadmap but is not the headline
  deliverable.

---

## Project layout

```
federated-learning-lab/
├── fl/
│   ├── server.py                  # Aggregator: collects updates, averages, broadcasts
│   ├── client.py                  # Local trainer: receives model, trains, returns delta
│   ├── algorithms/
│   │   ├── fedavg.py              # Vanilla FedAvg
│   │   ├── fedprox.py             # Proximal-term hook (client-side grad)
│   │   └── scaffold.py            # Control variates + server-side aggregator
│   ├── datasets/
│   │   └── mnist_partition.py     # IID, label-skew, Dirichlet-alpha
│   └── models/
│       └── cnn.py                 # Small CNN for MNIST (~46k params)
├── privacy/
│   ├── dp.py                      # DP-SGD primitives + DPSGDClient (vmap)
│   └── secagg.py                  # Additive secret sharing skeleton
├── scripts/
│   ├── experiment.py              # Shared experiment scaffolding
│   ├── run_fedavg_mnist.py        # FedAvg runner
│   ├── run_fedprox_mnist.py       # FedProx runner
│   ├── run_scaffold_mnist.py      # SCAFFOLD runner
│   ├── run_dp_fedavg.py           # DP-FedAvg runner
│   ├── secagg_demo.py             # Toy 3-client SecAgg demo
│   ├── run_all.py                 # Phase 1-4 in one process
│   ├── make_comparison_plots.py   # Cross-algorithm plot
│   └── make_summary.py            # SUMMARY.md from all metrics.json
├── docs/                          # algorithms, specs, design decisions, roadmap, refs
└── results/                       # Per-run artifacts + SUMMARY.md
```

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

# Run a single algorithm.
python -m scripts.run_fedavg_mnist  --partition iid       --rounds 15
python -m scripts.run_fedavg_mnist  --partition dirichlet --alpha 0.1 --rounds 25
python -m scripts.run_fedprox_mnist --partition dirichlet --alpha 0.1 --mu 0.01 --rounds 25
python -m scripts.run_scaffold_mnist --partition dirichlet --alpha 0.1 --rounds 25
python -m scripts.run_dp_fedavg     --partition iid       --rounds 20 --noise-sigma 1.0
python -m scripts.secagg_demo

# Or run the full phase 1-4 sweep in one go (sequential, ~45 min on a single GPU).
python -m scripts.run_all

# Build the cross-experiment summary after any combination of runs.
python -m scripts.make_summary
```

Each experiment writes ``results/<name>/{metrics.json, curve.png, REPORT.md}``.
``run_all`` additionally produces ``results/three_way_comparison.png`` and
``results/THREE_WAY_REPORT.md``; ``make_summary`` produces ``results/SUMMARY.md``.

---

## Algorithms at a glance

| Algorithm | Idea | Trade-off |
|---|---|---|
| **FedAvg** (2017) | Weighted average of local model updates | Drifts under Non-IID; cheap |
| **FedProx** (2018) | Add `(μ/2)·\|w - w_global\|²` to local loss | Tolerates straggler clients; needs μ tuning |
| **SCAFFOLD** (2019) | Subtract drift via per-client control variates | Faster convergence; ~2× communication |

See [`docs/algorithms.md`](docs/algorithms.md) for derivations and
intuitions.

---

## Why these algorithms?

FedAvg is the baseline against which all federated work is measured.
FedProx and SCAFFOLD attack the same problem (client drift under
Non-IID data) from opposite directions — proximal regularization
versus direct gradient correction. Comparing them on the same
problem isolates the algorithmic trade-off cleanly. See
`docs/design-decisions.md` for the longer reasoning.

---

## References

See [`docs/references.md`](docs/references.md). Core papers:

1. McMahan et al. (2017) — *Communication-Efficient Learning of Deep
   Networks from Decentralized Data* (FedAvg).
2. Li et al. (2018) — *Federated Optimization in Heterogeneous
   Networks* (FedProx).
3. Karimireddy et al. (2019) — *SCAFFOLD: Stochastic Controlled
   Averaging for Federated Learning*.
4. Abadi et al. (2016) — *Deep Learning with Differential Privacy*
   (DP-SGD).
5. Bonawitz et al. (2017) — *Practical Secure Aggregation for
   Privacy-Preserving Machine Learning*.

---

## Author

Wei Cheng (Wayne) Chiu · [GitHub](https://github.com/waynehacking8) ·
M.S. Computer Science, NTUST (April 2026).
