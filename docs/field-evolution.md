# Field Evolution: Federated Learning + Private LLM Deployment, 2017–2026

> Why this file exists: a FedAvg / FedProx / SCAFFOLD prototype on MNIST
> is easy to read as "I implemented a 2017 paper". This document
> re-frames the same work inside the 2026 sovereign-AI landscape, so a
> reviewer can see the prototype as the foundation of a contemporary
> private-LLM deployment story, not a museum piece.

---

## 1. The nine-year arc, in one sentence

> FedAvg (2017) was the field's Transformer-paper moment; FedProx and
> SCAFFOLD (2018–2020) made it work under Non-IID; Differential Privacy
> and Secure Aggregation (2016–2017) supplied the privacy primitives;
> LoRA-era FedLoRA variants (2023–2025) re-anchored the field around
> LLM fine-tuning; Apple Private Cloud Compute (2024) is the *consumer*
> InstructGPT moment for private AI; and the sovereign-LLM deployment
> landscape (Naver-BOK, Mistral, Sakana, Taiwan AI Labs FedGPT) is
> where the next paradigm shift is being negotiated in 2025–2026.

---

## 2. Five core mental models — the world view behind the algorithms

These five frame everything in `fl/` and `privacy/`. Reading them before
the code makes the design decisions obvious.

### M1. Privacy is a statistical property, not an engineering attribute

Differential privacy guarantees *indistinguishability* — two
neighboring datasets produce nearly the same output distribution. DP
does **not** prevent information leakage; it bounds a single
individual's influence on the output. Once the privacy budget is
spent, it is gone. Composition is additive. This is why DP-SGD looks
like SGD with two ugly extras (clip, then noise) and why the rest of
the system has to carry the accountant forward.

### M2. Federated learning is communication-bound, not compute-bound

This single fact flips every ML-engineering instinct. GPUs do not
help. Wall-clock time is dominated by round-trips, stragglers, and
client dropouts. *Every* serious FL improvement — FedAvg over
distributed SGD, FedProx over FedAvg, SCAFFOLD over FedProx, FedLoRA
over full-model federated training — is in the end an answer to a
single question: **how do we use fewer bytes per round, or fewer
rounds, to reach the same utility**?

### M3. Threat model precedes mechanism

Semi-honest versus malicious. Honest-majority versus
dishonest-majority. Curious-server versus colluding-clients. Each
assumption changes which primitive is correct. New practitioners
choose a tool (HE / SMPC / TEE / DP); senior practitioners draw the
attacker graph first and the tool falls out of it.

### M4. Non-IID is objective drift, not data drift

The textbook framing — "clients have different distributions" — is
correct but inert. The actionable framing is: each client's local
optimum is offset from the global optimum, so $E$ local steps drag
the model toward $\sum p_k w_k^* \neq w^*$. SCAFFOLD subtracts that
drift; FedProx adds a quadratic anchor that resists it. The
distribution is downstream of an *optimization geometry* problem.

### M5. Sovereign AI is a supply-chain and trust-topology question

Banks and hospitals do not choose against cloud LLMs because the
cloud model is technically weaker; they choose against it because
data-residency rules and audit obligations make the cloud
non-deployable. The right architecture follows from the trust
topology, not from the model leaderboard. This is why FedGPT-style
on-prem stacks exist and why Apple PCC, however elegant, does not
solve sovereign-AI for B2G / B2B.

---

## 3. Three live disagreements

These are the questions an interviewer will have an opinion on.

### D1. DP versus SMPC — weak-fast versus strong-slow

- **DP camp** (Google, Apple, Abadi 2016): a privacy guarantee you can
  run is the only privacy guarantee that matters. DP-SGD scales to
  arbitrary models.
- **SMPC camp** (CrypTen, MP-SPDZ, Aly et al.): an $\epsilon=8$ DP
  guarantee is a compliance number, not a privacy number. Real
  guarantees are cryptographic.
- **Where the field is**: industry runs DP; academia and certain
  high-assurance deployments run SMPC. The pragmatic median is
  DP-on-the-input + SMPC-on-the-aggregator.

### D2. FedAvg simplicity versus SCAFFOLD / FedProx correction

- **McMahan camp**: with careful client sampling and learning-rate
  scheduling, FedAvg is enough.
- **Karimireddy camp**: FedAvg has a Non-IID bias that algorithmic
  patches must remove.
- **Where the field is**: cross-device deployments stay with FedAvg
  (stateful per-client control variates are impractical when clients
  drop out). Cross-silo deployments — hospital and bank federations —
  pick FedProx (most common) or SCAFFOLD (when the data heterogeneity
  is severe enough to justify 2× communication overhead).

### D3. TEE versus HE+SMPC for confidential inference

- **TEE camp** (Apple PCC, NVIDIA H100/H200/B200 confidential mode):
  hardware root-of-trust, <3% overhead at 70B scale, $0$ overhead on
  B200. Production-ready today.
- **Crypto camp** (Gentry, Halevi, zama.ai): TEE has been broken at
  least every two years (TEE.Fail, Battering RAM, GPUBreach). The
  only durable answer is mathematical.
- **Where the field is**: production has converged on TEE. The crypto
  approach is restricted to offline audit (zkML) and very specific
  niches (encrypted embedding lookup). FHE inference for full LLMs
  is still minutes-per-token.

---

## 4. Where this prototype sits

`fl/` implements **FedAvg, FedProx, SCAFFOLD** on MNIST under
configurable Non-IID partitions. `privacy/` provides DP-SGD and an
additive-secret-sharing skeleton.

This is the **unit test** for the FL stack that sits underneath modern
on-prem LLM deployments:

- The **FedAvg loop** in `fl/server.py` is the same control flow used
  by NVIDIA FLARE, Flower, FedML, OpenFL, and the Google Federated
  Computing Platform — the differences are in transport, security
  filters, and orchestration, not the algorithmic core.
- The **SCAFFOLD control variates** in `fl/algorithms/scaffold.py`
  are the same mechanism that, scaled up and combined with PEFT,
  drives FedSA-LoRA (ICLR 2025).
- The **DP-SGD module** in `privacy/dp.py` is the primitive that, in
  production, becomes Opacus / TF-Privacy with a proper RDP
  accountant.

The prototype is **not** an FedGPT reproduction; it is the layer of
the stack that has to be right before any production federated
LLM deployment can be reasoned about.

---

## 5. Has this field had its "GPT moment"?

> My answer: not yet, but **Apple Private Cloud Compute (2024) is its
> InstructGPT moment**.

**Why not GPT yet**:
1. **No scale → emergence axis.** FL improvements come from algorithm
   design, not from "more compute". There is no scaling law for FL the
   way there is for transformer pretraining.
2. **No transformer-equivalent substrate.** Cross-device FL still
   runs on the same FedAvg loop McMahan wrote in 2017; everything
   since is a patch.
3. **No public deployment.** Gboard's FL is invisible to users; PySyft
   and Flower are research tools.

**Why Apple PCC is the InstructGPT-equivalent**:
- First system to package confidential computing + verifiable
  inference + privacy-by-design into a *product*.
- Transparency-log attestation that third-party researchers can
  audit — a deployable trust story, not a paper.
- Apple has not solved B2G / B2B sovereignty (the model still runs
  on Apple-owned silicon in Apple-owned data centers, subject to US
  CLOUD Act), but the architecture is the template every subsequent
  sovereign system will be measured against.

**What the GPT moment will probably look like** (best guess,
defensible in interview):
- A **federated foundation model** trained across regulated silos
  (banks, hospitals, ministries) where retention of utility under
  Non-IID + privacy + Byzantine tolerance is finally
  demonstrated at frontier scale.
- Or: a verifiable-inference primitive (FHE / zkML) breaks the
  3–4-order-of-magnitude wall and lets a publicly-auditable cloud
  service run frontier-scale LLMs at production latency.
- The first one is plausible by 2027–2028; the second is plausible
  by 2030 absent a hardware breakthrough.

---

## 6. How this maps to interview talking points

If a reviewer asks "what does this prototype demonstrate":

1. **Algorithmic literacy.** FedAvg + FedProx + SCAFFOLD + DP-SGD is
   the reading-comprehension floor for any FL paper from 2017 to
   2026.
2. **Cross-silo versus cross-device awareness.** The choice between
   the three algorithms depends on the deployment topology. Stating
   "I'd default to FedProx for hospital federation, SCAFFOLD only if
   heterogeneity is extreme, never SCAFFOLD for cross-device" is a
   stronger signal than reciting their derivations.
3. **Threat-model literacy.** DP is for output release; SMPC is for
   aggregation; TEE is for inference; they compose. Naming the right
   one for the right adversary is what production architecture
   actually demands.

If a reviewer asks "why isn't this on LLMs yet":

- "Because the FL stack has to be correct before federated LoRA on
  Llama can be debugged. The papers I find interesting next — FedSA-
  LoRA from ICLR 2025, the one-shot FL line with diffusion-synthesized
  pseudo-data, the 2024 SCAFFOLD backdoor amplification result — all
  build on top of the primitive I implemented here."

---

## 7. Further reading

- `docs/references.md` — the full paper list, now extended with the
  2022–2026 lineage (FedSA-LoRA, one-shot FL, Byzantine-robust 2025,
  Apple PCC, NVIDIA H100/H200/B200 CC, TEE.Fail, zkML, FHE on GPU,
  sovereign-LLM deployments).
- `docs/algorithms.md` — derivations of FedAvg, FedProx, SCAFFOLD.
- `docs/design-decisions.md` — why this scope, why these choices.
- `docs/roadmap.md` — what's done, what's planned, what's stretch.
