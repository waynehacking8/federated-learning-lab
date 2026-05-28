# Roadmap

Living document tracking what's done, what's in progress, and what's
planned.

---

## Phase 1 — FedAvg core

### 1.1 Data and model
- [ ] MNIST loader using `torchvision.datasets.MNIST`.
- [ ] Small CNN: 2 conv layers + 2 fully-connected; ~21k parameters.
- [ ] IID partition (random equal split across clients).

### 1.2 Server / Client API
- [ ] `Server` class: maintains global model, runs `round()` method.
- [ ] `Client` class: holds local data and current model copy, runs
      `local_update(global_model)` returning model delta or weights.

### 1.3 FedAvg algorithm
- [ ] Weighted-average aggregation by client sample count.
- [ ] Client subsampling each round (configurable participation rate).
- [ ] Per-round evaluation on a held-out global test set.

### 1.4 Convergence experiment
- [ ] Run 50 rounds on IID MNIST; verify accuracy approaches centralized
      baseline (~99% with the chosen CNN).
- [ ] Save convergence plot to `results/fedavg_iid.png`.

---

## Phase 2 — Non-IID partitioning

### 2.1 Partition schemes
- [ ] Label-skew: each client sees `k` of 10 classes, `k ∈ {1, 2, 3}`.
- [ ] Quantity-skew: client sizes follow a power law.
- [ ] Dirichlet-α: class proportions per client `~ Dir(α)`, with
      `α ∈ {0.1, 0.5, 1.0, 10.0}` representing very-skewed to near-IID.

### 2.2 Non-IID sensitivity experiment
- [ ] Re-run FedAvg under each partition.
- [ ] Tabulate rounds-to-target-accuracy vs partition severity.
- [ ] Save plot to `results/fedavg_noniid.png`.

---

## Phase 3 — FedProx

### 3.1 Algorithm implementation
- [ ] Proximal-term gradient: `g_prox = g + μ * (w_local - w_global)`.
- [ ] Configurable μ; sweep `μ ∈ {0, 0.001, 0.01, 0.1, 1.0}`.

### 3.2 Comparison experiment
- [ ] Side-by-side FedAvg vs FedProx on the most-skewed partition.
- [ ] Save plot to `results/fedavg_vs_fedprox.png`.

---

## Phase 4 — SCAFFOLD

### 4.1 Algorithm implementation
- [ ] Per-client control variate `c_k` and server control variate `c`.
- [ ] Corrected gradient: `g̃ = g - c_k + c`.
- [ ] Control-variate update at end of round.

### 4.2 Comparison experiment
- [ ] Three-way comparison: FedAvg / FedProx / SCAFFOLD on the same
      Non-IID partition.
- [ ] Save plot to `results/three_way_comparison.png`.
- [ ] Communication cost analysis: bytes per round per algorithm.

---

## Phase 5 — Differential privacy

### 5.1 DP-SGD primitive
- [ ] Per-sample gradient clipping with norm `C`.
- [ ] Gaussian noise with scale `σ * C / batch_size`.

### 5.2 DP-FedAvg experiment
- [ ] Sweep `(C, σ)` to understand the privacy/accuracy trade-off.
- [ ] Plot accuracy vs noise scale; report `ε` approximations.

---

## Phase 6 — Secure aggregation skeleton

### 6.1 Additive secret sharing
- [ ] Each client splits its update into `K-1` random pieces + 1 piece
      that makes the sum correct; sends pieces to peers.
- [ ] Server only sees the final sum across clients, not individuals.

### 6.2 Demonstration
- [ ] Toy example with 3 clients showing the server cannot reconstruct
      individual contributions.

---

## Phase 7 — Stretch goals

- [ ] **FedLoRA**: federated fine-tuning of a small language model
      (e.g., DistilBERT) using LoRA adapters. Adapter weights are tiny,
      so communication cost is manageable.
- [ ] **FedDyn / FedNova**: newer drift-mitigation variants worth
      benchmarking against the trio.
- [ ] **Opacus integration**: replace the custom DP module with
      Opacus and report rigorous (ε, δ) bounds.
- [ ] **Asynchronous FL**: relax synchronous-round assumption; clients
      submit updates whenever ready.
- [ ] **Real network deployment**: package server and client as
      separate processes communicating over gRPC; demonstrate on a
      single machine, scalable to multi-host.

---

## What's *not* on the roadmap

- **Production crypto.** Real-world Bonawitz SecAgg, threshold key
  recovery, etc. — out of scope.
- **Federated unlearning.** Important but a separate research thread.
- **Cross-silo orchestration platforms** (e.g., NVIDIA FLARE,
  OpenFL). The prototype demonstrates algorithms, not deployment.

---

*Last updated on initialization. To be revised as phases complete.*
