# Results validation against the literature

Every headline number in this lab is cross-checked here against the
original papers and an independent privacy accountant. The goal is that
no reported value is "surprising" without a cited, mechanistic reason.

Generated 2026-05-29. Sources are listed at the bottom.

---

## 1. FedAvg IID -- 0.9863

**Claim:** FedAvg on IID MNIST reaches ~0.986, approaching the
centralized baseline.

**Literature anchor:** McMahan et al. 2017 (the FedAvg paper) report
FedAvg matching the centralized CNN baseline (~99%) on IID MNIST within
a modest number of rounds. The small 2-conv LeNet-style CNN reaches
~99% centrally.

**Verdict: reasonable.** 0.986 is just under the centralized ceiling,
exactly as expected for IID FedAvg. Acceptance gate (>=0.97) passes.

---

## 2. Dirichlet(0.1), K=10 -- all three algorithms within ~1pp (0.97-0.98)

**Claim:** FedAvg 0.977, FedProx(mu=0.01) 0.977, SCAFFOLD 0.967 -- not
separable.

**Mechanistic reason:** With only 10 clients at full participation,
every aggregation round still mixes all 10 label distributions, and
Dir(0.1) -- though "very skewed" by the alpha value -- still leaves each
client with a long tail of minority-class samples. On a task as easy as
MNIST the global model recovers regardless of the drift-correction
mechanism. SCAFFOLD sitting ~1pp *below* FedAvg here is the known
near-convergence behaviour: once drift is small, the control-variate
correction injects variance rather than removing it.

**Verdict: reasonable, and it is why the label_skew sweep was added.**
The two "FAIL" acceptance gates (FedProx beats FedAvg by >=2pp; SCAFFOLD
reaches 0.90 faster) fail because the *premise* (Dir(0.1)/K=10 is hard
enough to separate algorithms) is false, not because of an
implementation error.

---

## 3. label_skew(2): SCAFFOLD worst at K=10, best at K=100

**Claim:** On 2-classes-per-client skew, SCAFFOLD goes from worst at
K=10 to best at K=100.

**Mechanistic reason (verified against the SCAFFOLD paper):**
- SCAFFOLD's correction is `g - c_local + c_global`. The server control
  variate `c_global` is the *mean* of client control variates.
- The paper's guarantee is that SCAFFOLD is "not affected by data
  heterogeneity or client sampling" *at convergence* -- it says nothing
  about a small-client, few-round, many-local-epoch transient.
- Control variates initialise at zero and only become useful estimates
  of the global direction after several rounds. With K=10 severely
  skewed clients and E=5 local epochs, each `c_local` is a noisy
  estimate from a 2-class slice, and averaging only 10 of them gives a
  poor `c_global` early on. With K=100 the mean of 100 variates is a far
  better global-direction estimate -- the variance-reduction regime the
  algorithm targets.

**Important correction from the convergence re-run (50 rounds, unified
budget):** the original K=10 SCAFFOLD number (0.686 at 25 rounds) was a
*non-converged* value -- the curve was still rising. See
`UNIFIED_LABELSKEW_REPORT.md` for the converged comparison; the
client-count effect is real but its magnitude shrinks once both regimes
are run to a plateau. We therefore report SCAFFOLD's value as a
plateau, not a truncated snapshot.

**Verdict: reasonable.** The literature does not claim SCAFFOLD wins in
*every* regime; it claims asymptotic robustness. A small-K transient
disadvantage is consistent with how control variates are built.

---

## 4. FedProx mu=0.1 hurts at K=10 (0.802 vs FedAvg 0.822)

**Literature anchor (FedProx authors, direct):** "large mu can restrict
the trajectory of the iterates by constraining the iterates to be closer
to that of the global model, potentially slowing convergence" and "a
large mu may cause the local models to be too close to the global model,
leading to slow convergence."

**Verdict: reasonable, textbook.** Under a fixed round budget, an
over-strong proximal anchor to a global model that is itself drifting
slows progress. mu=0.01 (milder) is neutral; mu=0.1 costs ~2pp at K=10
and is neutral again at K=100 where the global model is more stable.

---

## 5. DP-FedAvg privacy/accuracy -- rigorous epsilon

**Claim (corrected):** sigma=1 -> acc 0.906 at epsilon ~= 2.0;
sigma=5 -> acc 0.777 at epsilon ~= 0.24 (delta=1e-5).

**Accounting method:** subsampled-Gaussian RDP accountant (Google
`dp_accounting`, the engine Opacus wraps), Poisson sampling rate
q = 32/6000 = 0.00533, T = 3760 local steps per client. Cross-checked
against an independent PLD (privacy-loss-distribution) accountant.

| sigma | RDP eps | PLD eps | naive (old) |
|------:|--------:|--------:|------------:|
| 1.0   | 1.99    | 1.79    | 1425        |
| 5.0   | 0.24    | 0.22    | 285         |

The two independent accountants agree within ~10% (RDP is the looser,
safer upper bound). The old "naive" composition number was ~700x too
pessimistic because it ignored privacy amplification by subsampling.

**Literature anchor (Abadi et al. 2016, the DP-SGD paper):** on MNIST
they report (eps=2, delta=1e-5) -> ~95% and (eps=0.5, delta=1e-5) ->
~90%, using a centralized 1000-unit MLP trained for many epochs.

**Verdict: reasonable.** Our sigma=1 run sits at eps~=2.0 with 0.906 acc.
It is a few points below Abadi's 95% at the same epsilon because ours is
*federated* (10 clients, only E=1 local epoch x 20 rounds = far fewer
total gradient steps and no central shuffling), and uses a CNN trained
to far fewer steps. Same privacy operating point, slightly lower utility
-- exactly the direction expected from the federated + low-step-count
setting. The monotonic drop to 0.777 at sigma=5 (eps~=0.24, much
stronger privacy) is the standard privacy/utility tradeoff.

---

## 6. SecAgg -- exact reconstruction (L2 = 0)

Additive secret sharing is exact integer/float arithmetic: paired random
shares cancel, so the recovered sum equals the true sum to floating-point
zero. Nothing to validate against the literature -- it is arithmetic, and
the demo confirms the server sees only the sum, not individuals.

---

## Sources

- McMahan et al. 2017, *Communication-Efficient Learning of Deep
  Networks from Decentralized Data* (FedAvg).
- Li et al. 2018/2020, *Federated Optimization in Heterogeneous
  Networks* (FedProx) --
  <https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf>
- Karimireddy et al. 2020, *SCAFFOLD: Stochastic Controlled Averaging
  for Federated Learning* -- <https://arxiv.org/abs/1910.06378>
- Abadi et al. 2016, *Deep Learning with Differential Privacy* (DP-SGD).
- Google `dp_accounting` library (RDP + PLD accountants).

---

## 7. FedPer (Phase 7) -- personalization metric saturates on Dir(0.1)

**Observed:** On Dir(alpha=0.1), K=10, FedPer mean per-client accuracy
(~0.993) is essentially tied with FedAvg (~0.995) -- the +3pp gate is
NOT met.

**Why (mechanistic, not a bug):** Under Dir(0.1) each client's local
test distribution is dominated by 1-2 classes, so even the shared global
model classifies a client's own slice at ~0.99. There is no headroom for
a personalized head to add 3pp. FedPer's advantage only appears when the
single global model is genuinely *compromised* by conflicting clients;
on an easy per-client metric it cannot show. The FedPer global-metric
collapse (~0.5 vs FedAvg ~0.976) confirms the head specialized -- the
mechanism works, the metric just saturates. See `results/fedper_labelskew`
for the harder label_skew(3) regime where the effect is visible, and
design-decisions D16.

**Literature anchor:** Arivazhagan et al. 2019 (FedPer) and Collins et
al. 2021 (FedRep) report personalization gains on sharply heterogeneous
benchmarks; the gain size is a function of how much the global model is
forced to compromise, which a near-separable per-client metric hides.

---

## 8. Robust aggregation (Phase 8) -- median/Krum resist sign-flip

**Observed:** Under an amplified sign-flip attack (f=2 of n=10, IID),
coordinate-median / Krum / Multi-Krum / trimmed-mean / Bulyan stay near
the no-attack baseline while FedAvg degrades sharply.

**Literature anchor:** Blanchard et al. 2017 (Krum) require n >= 2f+3 for
the Byzantine-resilience guarantee; here n=10, f=2 -> 2f+3 = 7 <= 10, so
the guarantee applies. Krum selects the candidate minimizing the sum of
squared distances to its n-f-2 closest peers -- exactly the
implementation. Yin et al. 2018 give the median/trimmed-mean breakdown
analysis.

**Caveat documented:** Liu et al. (ICML 2023) -- under strong Non-IID,
honest-client divergence approaches attacker divergence and these
distance-based defenses silently fail. The demo is on IID, where the
guarantee holds; the caveat is stated in the Phase 8 report.

---

## 9. DLG (Phase 8.3) -- gradient leakage, broken by DP

**Observed:** A single MNIST image is reconstructed near-perfectly from
its gradient (MSE ~ 0.000); with DP-SGD noise (C=1, sigma=1) on the
gradient the reconstruction fails (pure noise).

**Literature anchor:** Zhu et al. 2019 (Deep Leakage from Gradients) --
the gradient-matching attack; uses smooth activations for the LBFGS inner
loop (we use a sigmoid CNN for the same reason). The DP break is the
expected consequence of the post-processing + noise: clipping + Gaussian
noise destroy the per-example signal the attack inverts.

---

## 10. FedAdam (Phase 9) -- server-side adaptive optimizer

**Observed:** On Dir(0.1), treating the averaged client delta as a
pseudo-gradient and applying server-side Adam (tau=1e-3) changes
convergence speed / final accuracy versus plain FedAvg.

**Literature anchor:** Reddi et al. 2020 (Adaptive Federated
Optimization) -- the FedOpt framework; tau=1e-3 is their adaptivity floor
(larger than vanilla Adam's epsilon). The gate checks the paper's claimed
benefit (fewer rounds to target OR higher final accuracy) holds on this
prototype.

---

## 11. FedLoRA (Phase 10) -- FedIT works, FedSA-LoRA is scale-dependent

**Observed:** FedIT (federate all adapters) reaches the centralized-LoRA
reference on IID AG News (gate PASS). FedSA-LoRA (share only A, keep B
local) correctly halves the adapter payload but does NOT beat FedIT on
per-client accuracy in this toy regime (gate FAIL).

**Literature anchor:** Sun et al. 2024 (FedIT) and Guo et al. ICLR 2025
(FedSA-LoRA: "A learns general knowledge, B learns client-specific;
share only A"). The mechanism and the payload halving match the paper;
the accuracy advantage is reported by the paper on larger models / more
data / more rounds than this prototype runs. See design-decisions D14 for
why we report the FAIL honestly rather than tuning to a PASS.

---

## Additional sources (Phases 7-10)

- Arivazhagan et al. 2019, *Federated Learning with Personalization
  Layers* (FedPer).
- Collins et al. 2021, *Exploiting Shared Representations for
  Personalized FL* (FedRep).
- Blanchard et al. 2017, *Machine Learning with Adversaries: Byzantine
  Tolerant Gradient Descent* (Krum) -- <https://arxiv.org/abs/1703.02757>
- Yin et al. 2018, *Byzantine-Robust Distributed Learning* (median,
  trimmed mean).
- Liu et al. 2023 (ICML), Non-IID caveat for robust aggregation.
- Zhu et al. 2019, *Deep Leakage from Gradients* (DLG).
- Reddi et al. 2020, *Adaptive Federated Optimization* (FedAdam/FedYogi).
- Sun et al. 2024 (FedIT); Guo et al. 2025 (FedSA-LoRA) --
  <https://arxiv.org/abs/2410.01463>

---

## 12. E-sweep (Phase 2.2) -- larger E helps on *mild* Non-IID

**Observed:** FedAvg on Dir(0.1), K=10: E=1 -> 0.966, E=5 -> 0.980,
E=10 -> 0.982 (rounds-to-0.90: 8 / 4 / 3). Larger E converges faster and
slightly higher -- the opposite of the naive "larger E -> more drift"
claim.

**Why (and why it is not a contradiction):** the "larger E increases
client drift" result (Khaled 2020; FedProx motivation) bites under
*severe* heterogeneity, where each client's local optimum is far from the
global one. Dir(0.1) on 10 fully-participating clients is only mildly
Non-IID (every client still sees a tail of all classes), so the extra
local compute from large E dominates the small drift it adds. The drift
penalty appears under sharp skew (label_skew(2), where the unified sweep
shows FedAvg plateauing well below IID). The E trade-off is
partition-severity-dependent -- same lesson as the mu-sweep, the SCAFFOLD
client-count finding, and the FedPer/FedAdam saturation: a single mild
benchmark hides effects that only a harder regime exposes.

**Literature anchor:** Khaled et al. 2020 (local-SGD bound with the local
steps trade-off); the drift cost scales with the heterogeneity constant,
which is small for mild Dirichlet.
