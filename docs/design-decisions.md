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

## D12. CNN parameter count is 46,706, not 21,706

**Decision:** Implement the architecture as stated in
`specifications.md` section 1 (Conv 1->16, Conv 16->32, FC 512->64,
FC 64->10) and accept the actual parameter count of 46,706. The
"21,706" figure in the spec is a miscalculation; the architecture is
the contract.

**Why:** The architecture is the canonical LeNet-style CNN used in
McMahan 2017 and almost every subsequent FL paper on MNIST. Changing
the channel widths or FC dimensions to hit 21,706 would diverge from
the reference architecture for no methodological benefit.

**What changed:** `make_mnist_cnn()` no longer asserts a specific
parameter count; callers can use `parameter_count(model)` for
introspection.

---

## D10. SCAFFOLD's ranking is client-count dependent (label_skew finding)

**Observation, not a design choice:** On label_skew(2 of 10 classes),
SCAFFOLD goes from *worst* of the trio at 10 clients (final 0.686 vs
FedAvg 0.822) to *best* at 100 clients (0.918 vs 0.885). See
`results/LABELSKEW_REPORT.md` and `results/K100_LABELSKEW_REPORT.md`.

**Why this happens:** SCAFFOLD's correction `g - c_local + c_global`
relies on the control variates being good estimates of gradient
direction. The server variate `c_global` is the mean of all client
variates. With 10 severely-skewed clients (2 classes each, 5 local
epochs) every `c_local` is estimated from a tiny biased slice, so their
mean is noisy and the correction degrades convergence. At 100 clients
the mean of 100 noisy variates is a far better estimate -- the
variance-reduction regime SCAFFOLD targets. FedProx mu=0.1 shows a
milder version: it costs 2 pp at K=10 (over-anchoring to a drifting
global model) but is neutral at K=100 (the global model is stable).

**Takeaway for the lab:** never rank drift-correction algorithms from a
single small-scale experiment. Client count is a first-order variable,
not a footnote. This is also why the spec's Dirichlet(0.1) on 10
clients was too mild to separate the algorithms (all within 1 pp) --
documented earlier as the reason the label_skew sweep was added.

**Convergence correction (added after the 120-round re-run, see D15):**
the original K=10 "0.686" was measured at 25 rounds and was badly
non-converged. Run to 120 rounds, SCAFFOLD K=10 climbs to ~0.83 (best
0.857) -- nearly level with FedAvg's 0.844. So the dramatic "worst at
K=10" gap is *mostly a truncation artifact*: at convergence SCAFFOLD is
roughly tied with FedAvg at K=10 and clearly best at K=100. The
client-count *direction* (SCAFFOLD improves more with more clients) holds;
the *magnitude* of the K=10 deficit shrinks from 14pp to ~1-3pp once both
are run to a plateau. Always check convergence before comparing finals.

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

---

## D13. RDP epsilon, not naive composition, for the DP runs

**Decision:** Report the (epsilon, delta) of the DP-FedAvg runs with the
subsampled-Gaussian RDP accountant (Google `dp_accounting`, the engine
Opacus wraps), cross-checked against an independent PLD accountant. Keep
the old naive-composition number only as a foil.

**Why:** The naive strong-composition estimate ignored privacy
amplification by subsampling (q ~= 0.005) and overshot the true epsilon
by ~700x (sigma=1: naive 1425 vs RDP 1.99). A number that large is
indistinguishable from "no privacy" and is actively misleading. The RDP
value (sigma=1 -> eps 1.99; sigma=5 -> eps 0.24, delta=1e-5) sits at the
canonical operating point from Abadi 2016 (eps~2 -> ~95% on MNIST,
centralized) and is the meaningful figure. RDP and PLD agree within ~10%
(RDP slightly looser, as theory predicts), which is evidence the bound
is real and not a single-library artefact. The DP-SGD mechanism itself
was verified: clip-to-C exact, noise std = sigma*C exact.

**What's lost:** nothing -- this strictly improves the privacy claim.
Full Opacus integration remains a stretch goal (roadmap Phase 11).

---

## D14. FedSA-LoRA's personalization advantage is scale-dependent (not reproduced in the toy regime)

**Observation, not a design choice:** In the Phase 10 setup (frozen
DistilBERT, 5 clients, ~600 samples each, rank-8 LoRA, 20 rounds),
FedSA-LoRA does NOT beat FedIT on per-client accuracy -- the
personalization gate fails. We report this honestly rather than tuning
until it passes.

**Why FedSA-LoRA struggles here:** LoRA initializes B=0, so the adapter
is inert until B trains. FedSA-LoRA averages only A across clients while
each keeps its own B. When the per-client B matrices diverge (which they
do under label skew, and which is exactly the regime the paper targets),
the averaged A no longer matches any single client's B -- and since the
adapter acts through the product B*A, the mismatch degrades the model.
With little local data and few rounds, B cannot re-align to the freshly
averaged A. Guo et al. (ICLR 2025) demonstrate the gains on larger
models, more data, and more rounds.

**What this still demonstrates (correctly):** (a) the FedIT IID gate
passes -- federated LoRA on adapters reaches the centralized-LoRA
reference; (b) the A-shared / B-local mechanism is implemented per the
paper; (c) the adapter payload is correctly halved (FedSA shares ~0.50x
the adapter bytes of FedIT). The accuracy advantage is the one piece
that is scale-dependent, and the repo says so.

**Why not just shrink the gate until it passes:** CLAUDE.md section 1 --
surface tradeoffs, do not hide confusion. A FAIL with a correct
mechanism and a cited reason is more honest (and more useful in an
interview) than a PASS manufactured by cherry-picking the seed or metric.

---

## D15. SCAFFOLD K=10 label-skew: report the converged value (120 rounds), not the 25-round snapshot

**Decision:** The earlier SCAFFOLD K=10 label_skew(2) number (0.686 at
25 rounds) was a non-converged snapshot -- the curve was still rising.
Re-run to 120 rounds to reach a genuine plateau and report that value.

**Why:** Under extreme few-client skew, SCAFFOLD's control variates take
many rounds to become useful estimates (they start at zero and each is
built from a 2-class slice). Truncating at 25 rounds understated its
converged accuracy. The unified-budget sweep (50 rounds) already showed
it had climbed to ~0.72; the 120-round run confirms the plateau. The
client-count finding (D10/SCAFFOLD worse at K=10, best at K=100) still
holds at convergence -- but the *magnitude* of the K=10 gap shrinks once
both regimes are run to a plateau, so we report converged values and a
plateau check (mean |delta| over last 5 rounds < 0.5pp) for every run.

**What's lost:** more GPU time per run. Worth it: a truncated number that
ranks an algorithm backwards is exactly the failure mode D10 warns about.

---

## D16. FedPer's gain is invisible on MNIST per-client metrics (report it honestly)

**Observation:** On MNIST, FedPer's per-client accuracy ties FedAvg
(~0.99 both) on Dir(0.1) AND on label_skew(3) -- the +3pp gate fails on
both -- because the per-client test slice is dominated by a few classes
and is trivially classified by any reasonable model. FedPer's global
(union) accuracy collapses (0.39 vs FedAvg 0.98), which proves the head
specialized; the metric simply has no headroom to reward it.

**Why we do not chase a PASS:** the honest fix would be a harder dataset
(CIFAR/FEMNIST) where per-client slices are non-trivial, which is a
roadmap item (D2), not a tuning knob. Manufacturing a PASS by shrinking
the gate or cherry-picking clients would misrepresent the result.

**What still stands:** the FedPer mechanism (shared body, per-client head,
body-only aggregation) is implemented and unit-tested; the
global-collapse / per-client-stable split is exactly the personalization
signature. The gain magnitude is dataset-dependent and reported as such.

---

## D17. Byzantine attack strength must be calibrated, or the defense looks unnecessary

**Observation:** A pure sign-flip (reflect the honest update through the
broadcast point, scale 1) barely dented FedAvg (-0.5pp) because the
honest update after 2 local epochs is tiny, so its reflection is tiny
too. With that weak attack, FedAvg "passed" and the robust aggregators
looked pointless.

**Decision:** Use the standard amplified sign-flip,
`w_byz = w_global - scale * (w_local - w_global)` with scale=10, which is
the attack strength robust-aggregation papers actually evaluate against.
Under it FedAvg collapses to ~0.10 (-89pp) while coordinate-median stays
at 0.989 (-0.2pp) -- the intended contrast.

**Lesson:** a robustness result is only meaningful if the attack is
strong enough to break the undefended baseline. Always verify the
baseline degrades before claiming the defense helps.

---

## D18. FedAdam shows no gain on Dir(0.1)/MNIST -- the task is too easy (report honestly)

**Observation:** On Dir(0.1), K=10, server-side FedAdam (server_lr=0.05,
tau=1e-3) reaches 0.980 vs FedAvg's 0.982 and is *slower* to the 0.95
target (round 13 vs 7). The gate (fewer rounds OR +1pp) FAILS.

**Why:** Reddi et al. 2020's adaptive server optimizers help when the
aggregated pseudo-gradient is ill-scaled across coordinates and the
problem is poorly conditioned. Plain FedAvg already hits 0.95 in 7 rounds
here -- MNIST + a mild Dirichlet split is too well-behaved for Adam's
per-coordinate normalization to add anything, and Adam's early
second-moment warmup costs a few rounds. The mechanism is implemented and
unit-tested; the benefit is problem-dependent.

**Consistent theme (D14, D16, D18):** FedPer, FedSA-LoRA, and FedAdam all
fail their gates for the SAME root reason -- MNIST / small AG News with
mild heterogeneity is too easy to expose the advantage these methods are
built for. Each is correctly implemented (mechanism verified); the
benchmark, not the code, is the limiter. The honest fix is a harder
dataset (CIFAR/FEMNIST) + ill-conditioned objective, which is roadmap
work, not a tuning knob. We report the FAILs with this reason rather than
tuning hyperparameters until a PASS appears.
