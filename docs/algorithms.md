# Algorithms

Distillation of the three federated-learning algorithms implemented in
this repo, with derivations and intuitions. Intended as a quick-
reference for myself, not a tutorial.

---

## 1. The setting

A set of clients ``{1, ..., K}`` each holds a private local dataset
``D_k``. A central server orchestrates training. We want to learn a
single global model ``w`` that performs well on the union of all
``D_k``, **without ever centralizing the data**.

Two cardinal constraints differentiate FL from standard distributed
training:

| Constraint | Implication |
|---|---|
| Data is **Non-IID** across clients | Each client's local optimum differs; naive averaging drifts |
| Communication is **expensive** | Few rounds; each round must move maximum information per bit |

---

## 2. FedAvg

### Algorithm

For ``t = 0, 1, ..., T``:

1. Server broadcasts current ``w_t`` to a sampled subset of clients ``S_t``.
2. Each client ``k ∈ S_t`` performs ``E`` local SGD epochs on ``D_k``
   starting from ``w_t``, producing ``w_t^k``.
3. Server aggregates:

       w_{t+1} = Σ_{k ∈ S_t}  (n_k / n) · w_t^k

   where ``n_k = |D_k|`` and ``n = Σ_k n_k``.

### Why the weighting

Without ``n_k/n`` weighting, a clinic with 100 patients influences the
global model as much as a hospital with 1 million — clearly wrong.
Weighting by sample count makes FedAvg approximately equivalent to
centralized SGD with the same total dataset, which is the right
reference point.

### When it breaks

- **Large E** + **Non-IID data**: clients drift toward their local
  optima, which differ. The averaged direction can be near-zero
  ("destructive interference") and convergence stalls.
- **Heterogeneous participation**: if low-data clients participate
  more often, their model bias dominates.

### 2.1 The two convergence-bound papers (read both)

Two papers carve up the FedAvg convergence story. New practitioners
conflate them; the distinction matters for which assumption set
applies to which deployment.

| Paper | Setting | Bound | Key takeaway |
|---|---|---|---|
| **Li et al. (ICLR 2020)** | Strongly convex + L-smooth | `O(1/T)` | Learning rate **must decay**; otherwise even full-gradient FedAvg leaves a residual gap of order `η`. Heterogeneity enters as `Γ = F* − Σ p_k F_k*` and inflates the constant. |
| **Khaled et al. (AISTATS 2020)** | Non-convex, bounded-variance + bounded-heterogeneity (no bounded-gradient assumption) | `O(1/√(NKT))` for local SGD | Theoretical foundation for "more local steps `K` reduces communication without hurting the asymptotic rate". Every communication-efficient FL paper since builds on this. |

The two papers together are the answer to "*why* does FedAvg
converge under Non-IID, and *when* does the rate degrade?" — Li
gives the strongly-convex picture and the residual gap; Khaled
gives the non-convex picture and the local-step trade-off.

---

## 3. FedProx

### Idea

Discourage local models from drifting too far from the broadcast
global model by adding a proximal term to the local objective:

    L_k(w) = F_k(w) + (μ/2) · ‖w − w_t‖²

where ``F_k`` is the local loss and ``μ`` controls the strength of the
anchor. Larger ``μ`` → more conservative local updates → less drift,
but slower local progress.

### When it helps

- **Heterogeneous compute budgets**: faster clients can do more local
  steps without drifting; slower clients return early without
  poisoning the average. The proximal anchor protects the global
  model from over-fit local updates.
- **Moderately Non-IID data**: ``μ`` smoothly trades drift for
  conservatism.

### When it fails

- **Severely Non-IID** (e.g., each client sees only one class): even
  with large ``μ``, the optimal local model is far from the global
  model, so the proximal term forces underfitting.

### Choosing μ in practice

`μ` interacts with the deployment topology, not just the data
heterogeneity:

| Deployment | Typical `μ` range | Rationale |
|---|---|---|
| Cross-silo (hospitals, banks) | `0.001 – 0.01` | Stable clients with stronger compute can tolerate looser anchoring; the proximal term is mostly insurance against straggler partial-work. |
| Cross-device (mobile) | `0.01 – 0.1` | Higher dropout and noisier client populations need a stronger anchor; large `μ` also makes "partial work" returns from slow devices safe to aggregate. |
| Default starting point | `0.01` | The Flower FedProx baseline; sweep on log-scale by validation accuracy. |

---

## 4. SCAFFOLD

### Idea

Instead of penalizing drift, *correct* the local gradient direction by
subtracting an estimated drift. Maintain control variates:

- ``c`` — a global control variate (server-side).
- ``c_k`` — a per-client control variate.

During local update, the corrected gradient is:

    g̃ = g − c_k + c

At the end of a round, ``c_k`` and ``c`` are updated to better
estimate the drift.

### Intuition

``c_k`` represents the systematic deviation of client ``k``'s local
gradients from the global average. Subtracting ``c_k`` and adding ``c``
"steers" the local update toward where the server-side optimum lies.

### When it helps

Convergence is provably faster than FedAvg under Non-IID data and
matches the rate of centralized SGD. The catch: communication cost
roughly doubles, since ``c_k`` must be transmitted alongside the model
update each round.

### SCAFFOLD is variance-reduction, not regularization

The geometry: `c_k - c` estimates the offset between client `k`'s
gradient direction and the global average gradient direction. The
local update subtracts that estimate, so each local step is — in
expectation — aligned with the global gradient. This is the same
machinery as **SVRG** (stochastic variance-reduced gradient)
transplanted to the federated setting. Once you see it this way, the
proof structure of Karimireddy 2020 becomes a direct adaptation of
variance-reduction analyses to local SGD.

### Two update options for the control variates

The Karimireddy paper offers two ways to refresh `c_k` at the end of
a round:

- **Option I (expensive):** `c_k⁺ = ∇F_k(w)` — one extra full
  forward/backward pass per round per client.
- **Option II (default):** `c_k⁺ = c_k - c + (1 / (K η)) (w_global - w_local)`
  — no extra gradient computation; uses quantities already on hand
  at end of round.

Option II is what `fl/algorithms/scaffold.py` implements. Option I is
almost never used in practice; we mention it only because it shows up
on whiteboard derivations.

### 2024 caveat — backdoor amplification

*Mind the Cost of Scaffold* (arXiv 2411.16167) shows that the same
`c_k` that fixes drift also **carries** malicious gradient directions
across rounds. A Byzantine client's poison signal lives inside its
own control variate and gets reused round after round, so the
backdoor attack signal is *amplified* relative to FedAvg.

The practical implication: SCAFFOLD is a good default for *trusted*
cross-silo deployments (hospitals, banks) but is a worse default
than FedProx when the threat model includes a fraction of
adversarial clients. This is why Phase 8 of the roadmap pairs
robust aggregators with the algorithmic trio rather than treating
SCAFFOLD as universally superior.

---

## 5. Decision criteria

Algorithm choice is not just "which converges fastest in theory" —
it depends on (a) the deployment topology (cross-silo vs cross-device),
(b) the threat model, and (c) the communication budget.

| Dimension | FedAvg | FedProx | SCAFFOLD |
|---|---|---|---|
| Cross-silo (few stateful clients) | OK; drifts under severe Non-IID | **Strong default** — small `μ`, partial-work tolerant | **Best under heterogeneity** if comms budget allows 2× |
| Cross-device (many transient clients) | **Default** — stateless | Acceptable | **Avoid** — stateful `c_k` goes stale when device participation is low |
| Communication budget | 1× baseline | 1× | 2× (`w` + `c_k`) |
| System heterogeneity | Sensitive to stragglers | Robust (`μ` protects partial work) | Medium |
| Adversarial clients (poisoning) | Standard exposure | Standard exposure | **Worse** — control variates amplify backdoors (arXiv 2411.16167) |

Practical rule of thumb for the Taiwan AI Labs target deployment
(cross-silo, medical or financial): start with FedProx at `μ = 0.01`;
escalate to SCAFFOLD only when heterogeneity is severe enough to
justify the 2× communication cost AND the threat model rules out
backdoor attacks.

---

## 6. Non-IID partitioning schemes

Used in `fl/datasets/mnist_partition.py` to systematically vary
heterogeneity:

| Scheme | Description |
|---|---|
| **IID** | Random assignment; baseline |
| **Label-skew** | Each client sees a subset of class labels (e.g., 2 of 10 classes) |
| **Quantity-skew** | Client dataset sizes follow a power law |
| **Dirichlet-α** | Class proportions sampled from `Dir(α)`; smaller α → more skew |

Reported metrics should explicitly state the partition scheme used.

---

## 7. Common pitfalls

- **Reporting only test accuracy.** Two algorithms can hit the same
  accuracy after many rounds yet differ by 5× in rounds-to-target.
  Always plot accuracy *vs* communication rounds.
- **Forgetting the IID baseline.** Without comparing against
  IID FedAvg, you can't tell whether Non-IID hurts or your algorithm
  is just bad.
- **Hyperparameter cross-contamination.** FedProx's μ and FedAvg's
  learning rate interact strongly with E (local epochs). Always
  re-tune when changing E or partition severity.
- **Single global model assumption.** When per-client distributions
  diverge sharply, the right answer may not be a single global model
  at all — see §8 on personalized FL.

---

## 8. Personalized federated learning

When client distributions diverge sharply (one hospital sees mostly
oncology, another mostly cardiology), forcing a single global model
hurts everyone. The personalized FL family relaxes that assumption.
Four representative strategies, sorted by how much they share:

| Strategy | What is shared | What is local | Example |
|---|---|---|---|
| Federate-then-finetune | Whole model | Final fine-tune | Run FedAvg, then each client trains a few epochs on its own data |
| Partial sharing | Lower layers / feature extractor | Classifier head | **FedPer** (Arivazhagan 2019), **FedRep** (Collins ICML 2021) |
| Meta-learning style | A good initialization | A few personalization steps | **Per-FedAvg** (Fallah NeurIPS 2020) — MAML inside FedAvg |
| Hypernetwork | A network that generates client weights from a client embedding | Nothing per client | **pFedHN** (Shamsian ICML 2021) — natural support for unseen clients |

This repo implements partial sharing (FedPer) in Phase 7 — it is the
simplest credible alternative that still exposes the personalization
vs global-utility trade-off cleanly. The same A/B-asymmetry idea
appears again in §9: FedSA-LoRA shares only the `A` matrix and keeps
`B` local, which is partial sharing transplanted to the LoRA setting.

---

## 9. FedLoRA family — federated parameter-efficient fine-tuning

Full-model FL on a 70B language model is not affordable per round.
The field has converged on **LoRA adapters**: freeze the base model
`W`, train a low-rank update `ΔW = B A` (`B: d×r`, `A: r×k`, `r ≪ min(d,k)`).
For `d=k=4096, r=8` this drops the per-client payload from ~16M
parameters to ~65k — a >250× reduction. The remaining design space
is how to aggregate `(A, B)` across heterogeneous clients.

| Method | Core mechanism | Strength | Weakness |
|---|---|---|---|
| **FedIT** | Apply FedAvg to `(A, B)` directly | Simplest; same-rank only | Averaging `A` and `B` independently introduces non-linearity error (`B̄·Ā ≠ B·A`-averaged) |
| **FlexLoRA** (2024) | Reconstruct each client's `B_k A_k` server-side, average to full-size delta, SVD back to per-client rank | Supports heterogeneous rank | Server-side SVD cost |
| **HetLoRA** | Zero-pad to common shape; sparsity-weighted aggregation | Cheap | Outlier-sensitive |
| **FLoRA** | Stack rather than average; avoid the `B̄·Ā` non-linearity | Mathematically cleaner | Client-side merge overhead |
| **SLoRA** | Two-stage sparse-pretraining initialization | Helps under data heterogeneity | Two-stage workflow complexity |
| **FedSA-LoRA** (ICLR 2025) | Share only `A`; keep `B` local | Lower comms, harder model inversion, free personalization | Newest; less stress-tested |

The FedSA-LoRA insight is the most important one for the Taiwan AI
Labs interview: `A` encodes shared structure across clients, `B`
encodes client-specific structure. Treating them asymmetrically is
the same intuition as FedPer (shared body + private head) — only
applied inside the adapter rather than across model layers.

Roadmap Phase 10 implements FedIT first (the baseline) and then
FedSA-LoRA (the SOTA-of-2025 selective-aggregation variant) to make
this insight reproducible on a small text-classification task.
