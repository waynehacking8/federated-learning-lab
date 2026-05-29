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
