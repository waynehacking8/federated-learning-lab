# Roadmap

Living document tracking what's done, what's in progress, and what's
planned.

---

## Phase 1 — FedAvg core

### 1.1 Data and model
- [x] MNIST loader using `torchvision.datasets.MNIST`.
- [x] Small CNN: 2 conv layers + 2 fully-connected. Actual param count
      is 46,706, not ~21k (spec miscalculation; see design-decisions D9).
- [x] IID partition (random equal split across clients).

### 1.2 Server / Client API
- [x] `Server` class: maintains global model, runs `run_round()` method.
- [x] `Client` class: holds local data and current model copy, runs
      `local_update(global_state)` returning weights + sample count.

### 1.3 FedAvg algorithm
- [x] Weighted-average aggregation by client sample count.
- [x] Client subsampling each round (configurable participation rate).
- [x] Per-round evaluation on a held-out global test set.

### 1.4 Convergence experiment
- [x] Ran IID MNIST; FedAvg reaches 0.9863 (>= 0.97 acceptance gate),
      approaching the centralized baseline. (15 rounds sufficed.)
- [x] Convergence plot saved to `results/fedavg_iid/curve.png`.

---

## Phase 2 — Non-IID partitioning

### 2.1 Partition schemes
- [x] Label-skew: each client sees `k` of 10 classes (configurable;
      experiments used `k = 2`).
- [ ] Quantity-skew: client sizes follow a power law. (Not implemented;
      label-skew and Dirichlet were sufficient to separate algorithms.)
- [x] Dirichlet-α: class proportions per client `~ Dir(α)`; experiments
      used `α = 0.1`.

### 2.2 Non-IID sensitivity experiment
- [x] Re-ran FedAvg under IID, Dirichlet(0.1), and label_skew(2).
- [x] Finding: Dirichlet(0.1) on 10 clients is too mild to separate the
      algorithms (all within 1 pp); label_skew(2) is the discriminating
      regime. Client count is first-order (see design-decisions D10).
- [x] Plots saved per run under `results/<run>/curve.png`.

---

## Phase 3 — FedProx

### 3.1 Algorithm implementation
- [x] Proximal-term gradient: `g_prox = g + μ * (w_local - w_global)`,
      applied as a grad hook (see `fl/algorithms/fedprox.py`).
- [x] Configurable μ; experiments swept `μ ∈ {0.01, 0.1}`.

### 3.2 Comparison experiment
- [x] Side-by-side FedAvg vs FedProx on label_skew(2), K=10 and K=100.
- [x] Finding: μ=0.1 hurts at K=10 (over-anchoring, -2 pp) but is
      neutral at K=100. Plots in `results/labelskew_comparison.png` and
      `results/K100_labelskew_comparison.png`.

---

## Phase 4 — SCAFFOLD

### 4.1 Algorithm implementation
- [x] Per-client control variate `c_k` and server control variate `c`.
- [x] Corrected gradient: `g̃ = g - c_k + c`.
- [x] Control-variate update at end of round (Option II; see
      `fl/algorithms/scaffold.py`).

### 4.2 Comparison experiment
- [x] Three-way comparison on Dirichlet(0.1) (`three_way_comparison.png`,
      `THREE_WAY_REPORT.md`) and label_skew(2) at K=10 and K=100.
- [x] Key finding: SCAFFOLD is worst at K=10 (0.686) but best at K=100
      (0.918) on label_skew(2) -- ranking is client-count dependent
      (design-decisions D10).
- [ ] Communication cost analysis: bytes per round per algorithm.
      (Noted qualitatively -- SCAFFOLD is ~2x FedAvg per round -- but not
      yet measured exactly.)

---

## Phase 5 — Differential privacy

### 5.1 DP-SGD primitive
- [x] Per-sample gradient clipping with norm `C` (via `torch.func.vmap`
      + per-sample grad; see `privacy/dp.py`).
- [x] Gaussian noise calibrated to `σ * C`.

### 5.2 DP-FedAvg experiment
- [x] Swept σ at fixed C=1: σ=1 -> 0.906 acc, σ=5 -> 0.777 acc
      (monotonic degradation). Runs under `results/dp_fedavg_iid_C1_s*`.
- [x] Curves saved per run; naive ε reported (RDP accountant deferred to
      the Opacus stretch goal -- see design-decisions D6).

---

## Phase 6 — Secure aggregation skeleton

### 6.1 Additive secret sharing
- [x] Each client splits its update into random pieces summing to the
      true update; pieces distributed across peers (see
      `privacy/secagg.py`).
- [x] Server only sees the final sum across clients, not individuals.

### 6.2 Demonstration
- [x] Toy example with 3 clients; recovered sum matches true sum
      exactly (L2 error 0). See `results/secagg_demo/REPORT.md`.

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

*Last updated 2026-05-29: Phases 1-6 implemented and experimentally
validated (8/8 unit tests green; results under `results/`, summarized in
`results/SUMMARY.md`). Remaining open items: quantity-skew partition,
exact communication-cost measurement, and the Phase 7 stretch goals.*
