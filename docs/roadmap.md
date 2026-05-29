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
- [x] Split the CNN into shared feature extractor + per-client head
      (final Linear). Head discovered programmatically via
      `head_param_names` / `is_shared` (not hard-coded).
- [x] Aggregate only the body; each client keeps its head across rounds.
- [x] Per-client evaluation on each client's own slice + global metric.

### 7.2 Personalization experiment
- [x] Compared FedPer vs FedAvg on Dir(α=0.1) and label_skew(3).
- [~] Acceptance (per-client ≥ FedAvg + 3pp): **FAIL on MNIST** —
      FedAvg's per-client accuracy already saturates at ~0.995 (each
      client's own slice is near-trivial), leaving no 3pp headroom. The
      FedPer global-metric collapse (0.39 vs 0.98) confirms the head
      specialized — the mechanism works; MNIST is too easy to reward it.
      Reported honestly (design-decisions D16), not tuned to a PASS.
- [x] Plots: `results/fedper_vs_fedavg.png`, `results/fedper_labelskew3.png`.

---

## Phase 8 — Byzantine robustness and gradient-leakage demo

Notes §2.5 and §10.1.7 argue that FL needs both robust aggregation
(against poisoning) and per-update privacy (against DLG-style
inversion). This phase makes both empirical.

### 8.1 Robust aggregators — `fl/algorithms/robust.py`
- [x] Coordinate-wise median.
- [x] Krum (and Multi-Krum with `m = n - f`).
- [x] Trimmed mean.
- [x] Bulyan = Krum candidate pool + trimmed-mean over candidates.

### 8.2 Poisoning attack
- [x] Amplified sign-flip Byzantine clients (`w_global - 10·(w_local -
      w_global)`); calibrated so the undefended baseline actually breaks
      (design-decisions D17).
- [x] **PASS**: at n=10, f=2 (IID), FedAvg collapses to 0.098 (−89pp);
      median/Krum/Multi-Krum/trimmed/Bulyan all stay within 0.9pp of the
      0.991 baseline. See `results/robust/REPORT.md`.
- [x] Documented the Liu et al. (ICML 2023) Non-IID caveat.

### 8.3 Deep Leakage from Gradients demo
- [x] Single MNIST image reconstructed from its gradient (Zhu 2019).
- [x] **PASS**: no-DP reconstruction is pixel-perfect (MSE ~0); with
      DP-SGD noise (C=1, σ=1) it collapses to noise.
- [x] Figure saved to `results/dlg_with_and_without_dp.png`.

---

## Phase 9 — Server-side adaptive optimizer (FedAdam / FedYogi)

Notes §1.6 and Reddi et al. 2020 reframe the server-side aggregation
as a single optimizer step — opening Adam / Yogi / Adagrad as drop-in
replacements for plain weighted averaging.

### 9.1 FedOpt protocol — `fl/algorithms/fedopt.py`
- [x] Average client delta treated as a pseudo-gradient.
- [x] Server-side Adam / Yogi / Adagrad with τ=1e-3 adaptivity floor
      (Reddi 2020) and its own momentum buffers.
- [x] Optimizer pluggable; usable on FedAvg / FedProx client updates.

### 9.2 Comparison experiment
- [x] On Dir(α=0.1): FedAvg, FedAdam, FedProx+server-Adam.
- [~] Acceptance (faster OR +1pp): **FAIL** — FedAdam 0.980 vs FedAvg
      0.982, slower to target (r13 vs r7). Plain FedAvg already converges
      in 7 rounds on this well-conditioned task, so adaptive server steps
      add nothing. Mechanism verified; benefit is problem-dependent
      (design-decisions D18). `results/fedopt_comparison.png`.

---

## Phase 10 — FedLoRA prototype (federated PEFT)

Notes §4.3 and §10.1.5 argue that FL on full LLM weights is
infeasible, and the field has converged on PEFT — specifically LoRA
adapters. This phase ports the existing infrastructure to a tiny
text-classification model so the LoRA leg is reproducible end-to-end.

### 10.1 FedIT baseline — `fl/algorithms/fedlora.py`
- [x] Frozen DistilBERT + LoRA (rank 8) on q/v attention; AG News.
- [x] Federate only adapter weights (+ task head); base model frozen.
- [x] **PASS**: FedIT IID reaches 0.8985, ≥ 90% of the centralized-LoRA
      reference (0.887), within 20 rounds. `results/fedlora/REPORT.md`.

### 10.2 FedSA-LoRA (selective aggregation)
- [x] Share only `A` (+ keep `B` and head local).
- [x] Adapter payload correctly halved (FedSA 0.50× FedIT).
- [~] Per-client personalization gate: **FAIL** in this toy regime
      (zero-init B + little data + few rounds → averaged-A/local-B
      coupling instability). Mechanism + payload verified; the accuracy
      advantage is scale-dependent (design-decisions D14). Plot:
      `results/fedlora_communication_vs_accuracy.png`.

### 10.3 (Stretch) FlexLoRA — heterogeneous rank
- [ ] Not implemented — 10.2 did not produce a clean FedSA baseline at
      this scale, so the heterogeneous-rank stretch is deferred (the
      spec gated it on 10.2 succeeding).

---

## Phase 11 — Stretch goals

- [ ] **FedDyn / FedNova**: newer drift-mitigation variants worth
      benchmarking against the trio.
- [x] **Rigorous (ε, δ) accounting**: done via Google `dp_accounting`'s
      subsampled-Gaussian RDP accountant, cross-checked against PLD (the
      same engine Opacus wraps). σ=1 → ε≈1.99, σ=5 → ε≈0.24 at δ=1e-5.
      See `scripts/dp_accounting_report.py` and design-decisions D13.
      (Full Opacus *training* integration remains future work.)
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
