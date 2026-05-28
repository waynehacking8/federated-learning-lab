# Design Decisions

Non-obvious choices made while scoping this prototype, with the
reasoning behind each.

---

## D1. Why FedAvg / FedProx / SCAFFOLD as the core trio?

**Decision:** Implement these three algorithms before any newer
variants (FedNova, FedDyn, FedOpt, FedLoRA).

**Why:** Together they span the design space of the Non-IID drift
problem:

| Approach | Representative algorithm |
|---|---|
| Naive averaging | FedAvg |
| Regularization-based drift control | FedProx |
| Variance-reduction-based drift control | SCAFFOLD |

Newer algorithms generally combine or refine these axes. Understanding
the trio gives the conceptual foundation needed to read any 2024–2026
FL paper.

---

## D2. Why MNIST, not a real medical or financial dataset?

**Decision:** MNIST as the default benchmark.

**Why:**
- **Cheap and reproducible** — anyone can run experiments without
  data-access negotiation. Useful for sanity checks during
  algorithm development.
- **Easy to manipulate Non-IID partitioning** — MNIST's 10 classes
  give clean label-skew partitions; the visual structure makes the
  effect of skew interpretable.
- **Comparable to the literature** — every FL paper since 2017
  reports MNIST results. Numbers can be cross-validated against
  published baselines.

**What's lost:** real-world Non-IID patterns (medical imaging
heterogeneity, financial fraud distributions) are absent. The
roadmap lists CIFAR-10 and Tiny-ImageNet upgrades; medical or
financial datasets would require organizational data access.

---

## D3. Why in-process client simulation, not real RPC?

**Decision:** Clients are Python objects holding state; the "round"
loop iterates them in-process.

**Why:**
- **Algorithmic clarity** — the round-by-round dynamics are the
  thing being studied. Network code obscures the algorithm.
- **Reproducibility** — deterministic seeding works across all
  clients in one process; impossible to guarantee with distributed
  network state.
- **Speed of iteration** — running 100 simulated rounds in seconds
  enables hyperparameter ablations that would take days over real
  network.

**What's lost:** all the systems engineering of real FL — RPC
serialization, fault tolerance, secure channels, client
authentication. A `flower` or `fedml` integration would be the next
step if those concerns dominate.

---

## D4. Why PyTorch, not JAX / TensorFlow?

**Decision:** PyTorch end-to-end.

**Why:**
- **Mental model alignment** — I use PyTorch in current work; one
  less translation cost.
- **Manual state management** — federated learning requires explicit
  per-client model copies and weight averaging. PyTorch's eager mode
  and explicit ``state_dict`` make this straightforward; JAX's
  pytrees + ``jit`` would force me to think about transformations
  that aren't the point of this prototype.
- **Opacus compatibility** — the DP module would integrate with
  Opacus when the time comes; Opacus is PyTorch-native.

---

## D5. Why a separate `privacy/` module rather than baking DP into FedAvg?

**Decision:** Differential privacy is a wrapper around the local
training step, not a property of the federation algorithm.

**Why:** DP-SGD's gradient clipping + Gaussian noise are independent
of whether the outer loop is FedAvg, FedProx, or SCAFFOLD. Keeping
the concerns separated means the DP module can be turned on or off
for any of the three core algorithms — useful for ablations.

**What's lost:** a tighter integration could exploit algorithm-
specific structure (e.g., SCAFFOLD's control variates might amplify
privacy noise; this is a research question, not a default choice).

---

## D6. Why a custom DP-SGD instead of Opacus?

**Decision:** Implement DP-SGD's two primitives (per-sample gradient
clipping, Gaussian noise) directly.

**Why:** DP-SGD is two lines of math (clip per-sample gradients,
add calibrated noise). Writing it explicitly forces engagement with
the privacy mechanism. Opacus is the right call for production.

**What's lost:** rigorous (ε, δ) accounting via the moments-accountant
or RDP composition. This is on the roadmap as an upgrade once the
core mechanism is validated.

---

## D7. Why a SecAgg skeleton rather than the full protocol?

**Decision:** Implement additive secret sharing as a pedagogical
primitive; do not implement Bonawitz 2017's full protocol with
threshold key recovery.

**Why:** The full SecAgg protocol is a multi-month engineering
project on its own (Diffie-Hellman key exchange, Shamir secret
sharing for fault tolerance, dropout handling). A skeleton that
demonstrates "additive secret sharing means the server only sees
the sum, not individual contributions" carries the conceptual
weight without the engineering burden.

**What's lost:** real privacy guarantees from secure aggregation are
not validated. The roadmap lists "integrate `tensorflow-federated`'s
SecAgg" as the production upgrade path.

---

## D8. Why include this `design-decisions.md`?

Same reasoning as the sibling `iot-pdm-pipeline` repo: forcing the
rationale into writing reduces the chance of building something that
sounds defensible in conversation but isn't actually defensible.
