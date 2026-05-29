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

---

## D9. Why prioritize cross-silo defaults over cross-device?

**Decision:** Pick defaults (μ ranges, SCAFFOLD use, stateful
control variates) that are tuned for cross-silo deployment first;
cross-device behavior is documented but not optimized.

**Why:**
- The target deployment story for this prototype is **medical and
  financial federations** — hospital and bank consortia. Both are
  archetypal cross-silo: small number of clients (5–50), high
  uptime, strong compute, persistent state allowed.
- The cross-silo / cross-device split changes which algorithm is
  rational. SCAFFOLD's per-client control variate `c_k` is
  practical when clients are always reachable; on a phone
  population with 5% per-round participation, `c_k` goes stale
  faster than it can be refreshed. The notes' three-way table
  (§10.1.4) bakes this in.
- FedProx's μ range is also topology-dependent: 0.001–0.01 for
  cross-silo (stable clients can tolerate looser anchoring), 0.01–0.1
  for cross-device (stronger anchor needed to make partial-work
  returns safe). The defaults in `docs/algorithms.md` §3 are the
  cross-silo ones.

**What's lost:** This repo is not a drop-in study of Gboard-style
FL. A cross-device chapter would need participation-rate sampling,
client-state aging, and noisier client populations — all out of
scope here.

---

## D10. Why add Phases 7–10 (personalized, robust, FedOpt, FedLoRA)?

**Decision:** The original roadmap stopped at Phase 6 (SecAgg).
After mapping the interview-prep notes against the repo, four
arguments from the notes had no empirical evidence to point at:

1. **Notes §1.8** argues that "one global model for all clients" is
   often the wrong objective in heterogeneous deployments. There
   was no personalized-FL implementation. → **Phase 7 (FedPer).**
2. **Notes §2.5 and §10.1.7** argue that FL needs both DP and robust
   aggregation, and that DLG-style attacks make the "we only share
   gradients" defense insufficient. There was no robust-aggregator
   code and no DLG demo. → **Phase 8 (Krum/Median/Bulyan + DLG).**
3. **Notes §1.6** mentions the server-side adaptive-optimizer
   perspective (Reddi 2020) as a power-up worth knowing.
   → **Phase 9 (FedAdam).**
4. **Notes §4.3 and §10.1.5** argue FedLoRA is the natural endgame
   for federated LLM fine-tuning. The original roadmap had FedLoRA
   only as a one-line stretch goal. → **Phase 10 (FedIT + FedSA-LoRA).**

**Why these four specifically:**
- All four can be implemented on the same in-process simulation
  scaffold without taking on production complexity.
- Each one is a 2–4-hour implementation against the existing
  aggregator protocol — no architectural rewrite required.
- Each one directly closes a "what about X?" follow-up that the
  interview notes flag as likely.

**What's NOT in:** EPEAgents-style federated multi-agent systems,
PUMA-style secure inference, and zkML. These are multi-week crypto
or systems projects; cite them from `docs/references.md` and the
notes, do not promise an implementation.

---

## D11. Why a separate `docs/interview-map.md`?

**Decision:** Maintain an explicit cross-reference from each
interview-prep notes section to the file or experiment in this
repo that demonstrates it.

**Why:** The interview is a verbal exam, not a code review. During
the conversation you need to be able to say "the empirical answer
to that is at `fl/algorithms/scaffold.py` plus
`results/three_way_comparison.png`" within two seconds. The map
makes that lookup pre-computed instead of improvised.

**What's lost:** A small maintenance cost — when the repo or the
notes change, the map needs to be re-synced. The map is short by
design to keep that cost low.
