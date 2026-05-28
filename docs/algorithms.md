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

---

## 5. Decision criteria

| If you have | Use |
|---|---|
| IID data, plentiful communication | FedAvg |
| Non-IID data, heterogeneous clients, tight communication | FedProx |
| Non-IID data, want faster convergence, communication is cheap | SCAFFOLD |
| Non-IID data, need provable convergence | SCAFFOLD |

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
