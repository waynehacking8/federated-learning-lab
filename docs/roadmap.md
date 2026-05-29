# Roadmap

Living document tracking what's done, what's in progress, and what's
planned.

The list of phases below is also the explicit answer to "which
interview talking points have empirical evidence in this repo".
Cross-reference with `docs/interview-map.md`.

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
- [x] E-sweep `{1, 5, 10}` on Dir(α=0.1) — demonstrates the
      "larger E → more drift" claim from notes §1.6. See
      `results/E_SWEEP_REPORT.md` and `results/fedavg_noniid.png`.
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
- [x] Rounds-to-target for cross-silo μ range (0.001–0.01) versus
      cross-device μ range (0.01–0.1) — demonstrates the μ-table from
      notes §10.1.3. See `results/MU_SWEEP_REPORT.md` and
      `results/fedavg_vs_fedprox.png`.

---

## Phase 4 — SCAFFOLD

### 4.1 Algorithm implementation
- [x] Per-client control variate `c_k` and server control variate `c`.
- [x] Corrected gradient: `g̃ = g - c_k + c`.
- [x] Control-variate update at end of round (Option II — the
      no-extra-gradient form; see `docs/specifications.md` §6 and
      `fl/algorithms/scaffold.py`).

### 4.2 Comparison experiment
- [x] Three-way comparison on Dirichlet(0.1) (`three_way_comparison.png`,
      `THREE_WAY_REPORT.md`) and label_skew(2) at K=10 and K=100.
- [x] Key finding: SCAFFOLD is worst at K=10 (0.686) but best at K=100
      (0.918) on label_skew(2) -- ranking is client-count dependent
      (design-decisions D10).
- [x] Communication cost analysis: bytes per round per algorithm
      (SCAFFOLD payload is ~2× FedAvg — see notes §10.1.4). Measured in
      `results/COMM_COST_REPORT.md`.

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

## Phase 7 — Personalized federated learning

Notes §1.8 argues that "one global model for all clients" is the wrong
default when client distributions differ. This phase implements the
simplest credible alternative — partial sharing — and measures whether
it helps under severe heterogeneity.

### 7.1 FedPer (partial sharing) — `fl/algorithms/fedper.py`
- [ ] Split the CNN into shared feature extractor (conv layers) and
      per-client classifier head (final `Linear(64, 10)`).
- [ ] Aggregate only the feature-extractor parameters; keep each
      client's head local across rounds.
- [ ] Per-client evaluation against its own test partition (the
      personalized metric) plus a global metric on the union.

### 7.2 Personalization experiment
- [ ] Compare FedPer vs FedAvg on Dir(α=0.1).
- [ ] Acceptance: FedPer's per-client mean accuracy ≥ FedAvg's by ≥ 3
      points at round 50, even if global metric is similar.
- [ ] Save plot to `results/fedper_vs_fedavg.png`.

---

## Phase 8 — Byzantine robustness and gradient-leakage demo

Notes §2.5 and §10.1.7 argue that FL needs both robust aggregation
(against poisoning) and per-update privacy (against DLG-style
inversion). This phase makes both empirical.

### 8.1 Robust aggregators — `fl/algorithms/robust.py`
- [ ] Coordinate-wise median.
- [ ] Krum (and Multi-Krum with `m = n - f`).
- [ ] Trimmed mean.
- [ ] (Stretch) Bulyan = Krum candidate pool + trimmed-mean over candidates.

### 8.2 Poisoning attack
- [ ] Simulate `f` Byzantine clients that flip gradient sign
      (sign-flip attack) or scale gradients by a large constant.
- [ ] Acceptance: median / Krum keep accuracy within 5 points of the
      no-attack baseline at `f ≤ (n-2)/2`; FedAvg degrades visibly.
- [ ] Document the Liu et al. (ICML 2023) Non-IID caveat — most
      classical aggregators silently fail under heterogeneous data.

### 8.3 Deep Leakage from Gradients demo
- [ ] Toy reconstruction of a single MNIST image from its gradient
      (Zhu et al. 2019).
- [ ] Re-run with DP-SGD-noised gradients; show that bounded ε breaks
      the reconstruction.
- [ ] Save figure to `results/dlg_with_and_without_dp.png`.

---

## Phase 9 — Server-side adaptive optimizer (FedAdam / FedYogi)

Notes §1.6 and Reddi et al. 2020 reframe the server-side aggregation
as a single optimizer step — opening Adam / Yogi / Adagrad as drop-in
replacements for plain weighted averaging.

### 9.1 FedOpt protocol — `fl/algorithms/fedopt.py`
- [ ] Treat the average client delta as a "pseudo-gradient".
- [ ] Apply Adam / Yogi update on the server side with separate
      server-LR and momentum buffers.
- [ ] Make the optimizer pluggable — usable on top of FedAvg /
      FedProx / SCAFFOLD inputs.

### 9.2 Comparison experiment
- [ ] On Dir(α=0.1), compare FedAvg, FedAvg + server Adam, FedProx +
      server Adam.
- [ ] Acceptance: server-side adaptive optimizer improves either
      convergence speed (rounds-to-target) or final accuracy under
      heterogeneity.
- [ ] Save plot to `results/fedopt_comparison.png`.

---

## Phase 10 — FedLoRA prototype (federated PEFT)

Notes §4.3 and §10.1.5 argue that FL on full LLM weights is
infeasible, and the field has converged on PEFT — specifically LoRA
adapters. This phase ports the existing infrastructure to a tiny
text-classification model so the LoRA leg is reproducible end-to-end.

### 10.1 FedIT baseline — `fl/algorithms/fedlora.py`
- [ ] Choose a small language model (DistilBERT-class) and a small
      text-classification dataset (AG News or SST-2).
- [ ] Add LoRA adapters on attention projections (rank `r ∈ {4, 8}`).
- [ ] Federate ONLY the adapter weights `(A, B)` — the base model is
      frozen and identical across clients.
- [ ] Acceptance: FedIT reaches ≥ 90% of centralized-LoRA-fine-tune
      accuracy within 20 rounds on IID partition.

### 10.2 FedSA-LoRA (selective aggregation)
- [ ] Share only the `A` matrix; keep `B` local.
- [ ] Compare communication (FedIT vs FedSA-LoRA) and per-client
      personalization on a Dirichlet-α partition.
- [ ] Demonstrates the notes §10.1.5 "A learns common knowledge,
      B learns client-specific" insight empirically.
- [ ] Save plot to `results/fedlora_communication_vs_accuracy.png`.

### 10.3 (Stretch) FlexLoRA — heterogeneous rank
- [ ] Different clients train different ranks `r_k`.
- [ ] Server-side aggregation: reconstruct each client's `B_k A_k` to
      full-size delta, average, SVD back to per-client target rank.
- [ ] Only worth implementing if 10.2 produces clean baselines.

---

## Phase 11 — Stretch goals

- [ ] **FedDyn / FedNova**: newer drift-mitigation variants worth
      benchmarking against the trio.
- [ ] **Opacus integration**: replace the custom DP module with
      Opacus and report rigorous (ε, δ) bounds using the RDP
      accountant.
- [ ] **Asynchronous FL**: relax synchronous-round assumption; clients
      submit updates whenever ready.
- [ ] **Real network deployment**: package server and client as
      separate processes communicating over gRPC; demonstrate on a
      single machine, scalable to multi-host.
- [ ] **One-shot FL**: reproduce FedDISC / DENSE on a toy dataset;
      compare against multi-round FedAvg on rounds-to-target.

---

## What's *not* on the roadmap

- **Production crypto.** Real-world Bonawitz SecAgg, threshold key
  recovery, etc. — out of scope.
- **Federated unlearning.** Important but a separate research thread.
- **Cross-silo orchestration platforms** (e.g., NVIDIA FLARE,
  OpenFL). The prototype demonstrates algorithms, not deployment.
- **MWEM under SMPC.** Covered by the NICS PETS competition write-up;
  duplicating it here would dilute the FL focus.
- **Secure LLM inference (PUMA / THOR).** Multi-month crypto
  engineering; out of scope.
- **LLM-side serving optimizations (vLLM / FlashAttention).** Covered
  by the separate GPU Inference Benchmarks repo.

---

*Last updated 2026-05-29 to align with the Taiwan AI Labs interview-prep
notes (May 2026). Phases 1–6 are implemented and experimentally validated
(results under `results/`, summarized in `results/SUMMARY.md`); Phases
7–10 (personalized FL, robust aggregation + DLG, server-side optimizer,
FedLoRA) were added to give the notes' arguments empirical evidence in
this repo.*
