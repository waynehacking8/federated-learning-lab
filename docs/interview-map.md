# Interview Map — Notes → Repo

> Cross-reference from the Taiwan AI Labs interview-prep notes (`面試深度
> 筆記_TaiwanAILabs.md`) to the file or experiment in this repo that
> demonstrates each point. Use this during the interview to point at
> concrete code rather than just citing the algorithm.

The notes are organised in 10 chapters; this map follows their order.

---

## Ch 0.5 — Five mental models, three disagreements

| Notes section | Repo evidence |
|---|---|
| M1 Privacy is a statistical property | `privacy/dp.py` — DP-SGD clip + Gaussian noise; `docs/algorithms.md` §3 (DP) shows budget accounting |
| M2 FL is communication-bound | `docs/field-evolution.md` §2 frames every algorithm in `fl/algorithms/` as a "fewer bytes / fewer rounds" answer; roadmap Phase 9 (FedAdam) + Phase 10 (FedLoRA) are the next two compressions |
| M3 Threat model precedes mechanism | `docs/design-decisions.md` D7 (semi-honest SecAgg skeleton, not malicious); `privacy/secagg.py` documents the assumed adversary |
| M4 Non-IID = objective drift | `fl/datasets/mnist_partition.py` (Dirichlet partition); `fl/algorithms/scaffold.py` (SCAFFOLD subtracts drift); roadmap Phase 2.2 makes the drift measurable |
| M5 Sovereign AI = trust topology | `docs/field-evolution.md` §5 (Apple PCC is InstructGPT-moment, not GPT-moment); references §"Sovereign / private LLM deployment" |
| D1 DP vs SMPC | `privacy/dp.py` (DP path) + `privacy/secagg.py` (SMPC primitive); the two are complementary, not exclusive |
| D2 FedAvg vs SCAFFOLD/FedProx | Three-way comparison in roadmap Phase 4.2 (results saved to `results/three_way_comparison.png`) |
| D3 TEE vs HE+SMPC | `docs/references.md` §"Confidential computing" — TEE side; `privacy/secagg.py` — crypto-side primitive |

---

## Ch 1 — Federated learning theory

| Notes | Repo |
|---|---|
| 1.1 Cross-silo vs cross-device | `docs/design-decisions.md` D2 + new D9 — this lab targets cross-silo, which is why FedProx/SCAFFOLD are first-class |
| 1.2 FedAvg algorithm | `fl/algorithms/fedavg.py` + `docs/algorithms.md` §2 |
| 1.3 Non-IID — four types, client drift | `fl/datasets/mnist_partition.py` (label / quantity / Dirichlet); `docs/algorithms.md` §6 lists the taxonomy; `results/fedavg_noniid.png` (Phase 2.2) makes drift visible |
| 1.4 FedProx (proximal term) | `fl/algorithms/fedprox.py` + `docs/specifications.md` §5 + new μ table in `docs/algorithms.md` |
| 1.5 SCAFFOLD (control variates) | `fl/algorithms/scaffold.py` + `docs/specifications.md` §6 — Option II is the default; Option I exists in derivation only |
| 1.6 Three convergence-degradation factors | `docs/algorithms.md` §3 — Non-IID, E, sampling each named explicitly; Phase 2.2 ablation tabulates rounds-to-target |
| 1.7 System challenges (stragglers, comm, system heterogeneity) | `docs/design-decisions.md` D3 (in-process simulation deliberately skips network); `fl/client.py` allows variable `local_epochs` per client |
| 1.8 Personalized FL (FedPer / FedRep / Per-FedAvg / pFedHN) | New roadmap Phase 7 — FedPer prototype (shared body + private head) |

---

## Ch 2 — Privacy cryptography (your home territory)

| Notes | Repo |
|---|---|
| 2.1 DP (Laplace, exponential, sensitivity, composition) | `privacy/dp.py` (Gaussian variant for SGD); `docs/algorithms.md` §3 derivation |
| 2.2 SMPC (additive secret sharing, three safety properties, Beaver triples) | `privacy/secagg.py` — additive-share skeleton; `docs/design-decisions.md` D7 explains the scope cut |
| 2.3 MP-SPDZ (sfix, sint) | Not implemented (out of scope per D7); cite NICS competition for hands-on experience |
| 2.4 MWEM-under-SMPC (NICS) | Not in repo — your NICS competition is the canonical evidence; this lab demonstrates the FL leg of the stack |
| 2.5 FL attacks (DLG, MI, poisoning) and defences | New roadmap Phase 8 — DLG reconstruction demo + Krum/Median/Bulyan robust aggregators |
| 2.6 DP and SMPC are complementary | `docs/algorithms.md` §3-4 — DP wraps local SGD, SecAgg wraps the aggregation; they compose |

---

## Ch 4 — Three frontier directions

| Notes section | Repo evidence |
|---|---|
| 4.2 Federated MAS / EPEAgents | `docs/references.md` lists EPEAgents (ICML 2025); out of scope as code, but cited as future direction |
| 4.3 FedLoRA family (FlexLoRA / HetLoRA / FLoRA / SLoRA / FedSA-LoRA) | New roadmap Phase 10 — FedLoRA prototype; `docs/algorithms.md` §7 has the family table + FedSA-LoRA A/B-asymmetry insight |
| 4.4 Secure / private LLM inference (PUMA, THOR, verifiable inference) | `docs/references.md` §"Confidential computing"; no implementation (PUMA-class systems are weeks of crypto engineering) |

---

## Ch 5 — Whiteboard / oral derivations

| Notes question | Practice with |
|---|---|
| "Draw one FedAvg round" | `fl/server.py::Server.run_round` and `fl/client.py::Client.local_update` are the literal whiteboard |
| "Explain client drift, and how FedProx and SCAFFOLD fix it" | `docs/algorithms.md` §2-4; the three-way plot at `results/three_way_comparison.png` is the empirical proof |
| "Why is E a trade-off?" | Ablation in Phase 2.2 — E-sweep `{1, 5, 10}` on Dir(α=0.1) |
| "ε accounting in MWEM (ε/2T per step)" | `privacy/dp.py` docstring on the accounting model; full derivation in NICS write-up |
| "Additive secret sharing: split 7 across three parties" | `privacy/secagg.py` demonstration notebook |

---

## Ch 6 — System design

| Notes question | Repo evidence to cite |
|---|---|
| 6.1 Cross-silo medical FL platform | `docs/design-decisions.md` D9 (cross-silo first); roadmap Phases 1–4 are the prototype of the algorithmic core |
| 6.2 FedGPT AgentTeam on-prem | Roadmap Phase 10 (FedLoRA); references §"Sovereign LLM deployment" |
| 6.3 LLM inference is slow — how to diagnose | Out of repo scope; cite your GPU Inference Benchmarks repo |

---

## Ch 7 — Resume deep-dives

| CV item | Repo evidence |
|---|---|
| Federated-learning self-study | This whole repo |
| NICS PETS (MWEM under SMPC) | Not in this repo — separate write-up; this repo is the FL leg of the same privacy story |
| Comfortable with PyTorch and ML fundamentals | Roadmap Phases 1.1 (CNN), 1.4 (centralised baseline), 5 (DP-SGD per-sample gradients) |

---

## Ch 10 — 2026 SOTA enhancements

| Notes section | Repo evidence |
|---|---|
| 10.1.1 Li 2020 vs Khaled 2020 convergence bounds | `docs/algorithms.md` §2.1 (new) — names both, distinguishes assumptions |
| 10.1.2 SCAFFOLD Option I vs Option II | `docs/specifications.md` §6 + `docs/algorithms.md` §4 — Option II is the default; Option I is derivation-only |
| 10.1.3 FedProx μ ranges (cross-silo 0.001–0.01, cross-device 0.01–0.1) | `docs/algorithms.md` §3 (new μ table) |
| 10.1.4 Three-way cross-silo vs cross-device table | `docs/design-decisions.md` D9 (new) |
| 10.1.5 FedLoRA family | Roadmap Phase 10 + `docs/algorithms.md` §7 (new) + `docs/references.md` §"Heterogeneous LoRA" |
| 10.1.6 One-shot FL | `docs/references.md` §"One-shot FL"; not implemented |
| 10.1.7 Byzantine-robust aggregators (Krum → Multi-Krum → Median → Bulyan → Liu 2023) | Roadmap Phase 8 + `docs/specifications.md` §11 (new) |
| 10.1.7 SecAgg full protocol | `privacy/secagg.py` skeleton; full Bonawitz protocol explicitly out of scope per D7 |
| 10.2.x LLM inference SOTA (vLLM V1, EAGLE-3, DistServe, SpinQuant, KIVI, MoE) | Out of repo; this is the LLM-systems track, not the FL track |
| 10.3 Confidential computing + zkML | `docs/references.md` §"Confidential computing" |
| 10.4 Competition / customer / regulation | `docs/field-evolution.md` §5 |

---

## How to use this map in the interview

If asked an algorithmic question:
1. Cite the notes section to show structured prep.
2. Open the relevant repo file from this map to show concrete code.
3. Cite the matching `results/*.png` if the question is empirical.

If asked something this repo does NOT cover (e.g. PUMA, EPEAgents,
zkML), say so — pointing at the explicit "not implemented" notes
in `docs/design-decisions.md` is stronger than improvising.
