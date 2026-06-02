# Concept Map — Theory → Code

> Cross-reference from federated-learning concepts and algorithms to the
> file or experiment in this repo that implements or demonstrates each
> one. Use it to jump from an idea to the concrete code rather than just
> citing the algorithm.

The map is organised thematically, from FL foundations through privacy
cryptography to frontier directions and system design.

---

## Foundational mental models and core trade-offs

| Concept | Repo evidence |
|---|---|
| M1 Privacy is a statistical property | `privacy/dp.py` — DP-SGD clip + Gaussian noise; `docs/algorithms.md` §3 (DP) shows budget accounting |
| M2 FL is communication-bound | `docs/field-evolution.md` §2 frames every algorithm in `fl/algorithms/` as a "fewer bytes / fewer rounds" answer; roadmap Phase 9 (FedAdam) + Phase 10 (FedLoRA) are the next two compressions |
| M3 Threat model precedes mechanism | `docs/design-decisions.md` D7 (semi-honest SecAgg skeleton, not malicious); `privacy/secagg.py` documents the assumed adversary |
| M4 Non-IID = objective drift | `fl/datasets/mnist_partition.py` (Dirichlet partition); `fl/algorithms/scaffold.py` (SCAFFOLD subtracts drift); roadmap Phase 2.2 makes the drift measurable |
| M5 Sovereign AI = trust topology | `docs/field-evolution.md` §5 (Apple PCC is InstructGPT-moment, not GPT-moment); references §"Sovereign / private LLM deployment" |
| DP vs SMPC | `privacy/dp.py` (DP path) + `privacy/secagg.py` (SMPC primitive); the two are complementary, not exclusive |
| FedAvg vs SCAFFOLD/FedProx | Three-way comparison in roadmap Phase 4.2 (results saved to `results/three_way_comparison.png`) |
| TEE vs HE+SMPC | `docs/references.md` §"Confidential computing" — TEE side; `privacy/secagg.py` — crypto-side primitive |

---

## Federated learning theory

| Concept | Repo |
|---|---|
| Cross-silo vs cross-device | `docs/design-decisions.md` D2 + D9 — this lab targets cross-silo, which is why FedProx/SCAFFOLD are first-class |
| FedAvg algorithm | `fl/algorithms/fedavg.py` + `docs/algorithms.md` §2 |
| Non-IID — four types, client drift | `fl/datasets/mnist_partition.py` (label / quantity / Dirichlet); `docs/algorithms.md` §6 lists the taxonomy; `results/fedavg_noniid.png` (Phase 2.2) makes drift visible |
| FedProx (proximal term) | `fl/algorithms/fedprox.py` + `docs/specifications.md` §5 + the μ table in `docs/algorithms.md` |
| SCAFFOLD (control variates) | `fl/algorithms/scaffold.py` + `docs/specifications.md` §6 — Option II is the default; Option I exists in derivation only |
| Three convergence-degradation factors | `docs/algorithms.md` §3 — Non-IID, E, sampling each named explicitly; Phase 2.2 ablation tabulates rounds-to-target |
| System challenges (stragglers, comm, system heterogeneity) | `docs/design-decisions.md` D3 (in-process simulation deliberately skips network); `fl/client.py` allows variable `local_epochs` per client |
| Personalized FL (FedPer / FedRep / Per-FedAvg / pFedHN) | Roadmap Phase 7 — FedPer prototype (shared body + private head) |

---

## Privacy cryptography

| Concept | Repo |
|---|---|
| DP (Laplace, exponential, sensitivity, composition) | `privacy/dp.py` (Gaussian variant for SGD); `docs/algorithms.md` §3 derivation |
| SMPC (additive secret sharing, three safety properties, Beaver triples) | `privacy/secagg.py` — additive-share skeleton; `docs/design-decisions.md` D7 explains the scope cut |
| MP-SPDZ (sfix, sint) | Not implemented (out of scope per D7) |
| MWEM-under-SMPC | Not in repo — this lab demonstrates the FL leg of the privacy stack |
| FL attacks (DLG, MI, poisoning) and defences | Roadmap Phase 8 — DLG reconstruction demo + Krum/Median/Bulyan robust aggregators |
| DP and SMPC are complementary | `docs/algorithms.md` §3-4 — DP wraps local SGD, SecAgg wraps the aggregation; they compose |

---

## Frontier directions

| Concept | Repo evidence |
|---|---|
| Federated MAS / EPEAgents | `docs/references.md` lists EPEAgents (ICML 2025); out of scope as code, but cited as future direction |
| FedLoRA family (FlexLoRA / HetLoRA / FLoRA / SLoRA / FedSA-LoRA) | Roadmap Phase 10 — FedLoRA prototype; `docs/algorithms.md` §7 has the family table + FedSA-LoRA A/B-asymmetry insight |
| Secure / private LLM inference (PUMA, THOR, verifiable inference) | `docs/references.md` §"Confidential computing"; no implementation (PUMA-class systems are weeks of crypto engineering) |

---

## Derivations worth working through

| Question | Code to work through it with |
|---|---|
| "Draw one FedAvg round" | `fl/server.py::Server.run_round` and `fl/client.py::Client.local_update` are the literal walkthrough |
| "Explain client drift, and how FedProx and SCAFFOLD fix it" | `docs/algorithms.md` §2-4; the three-way plot at `results/three_way_comparison.png` is the empirical proof |
| "Why is E a trade-off?" | Ablation in Phase 2.2 — E-sweep `{1, 5, 10}` on Dir(α=0.1) |
| "ε accounting (ε/2T per step)" | `privacy/dp.py` docstring on the accounting model |
| "Additive secret sharing: split 7 across three parties" | `privacy/secagg.py` demonstration notebook |

---

## System design

| Question | Repo evidence |
|---|---|
| Cross-silo medical FL platform | `docs/design-decisions.md` D9 (cross-silo first); roadmap Phases 1–4 are the prototype of the algorithmic core |
| Federated LLM fine-tuning on-prem | Roadmap Phase 10 (FedLoRA); references §"Sovereign LLM deployment" |
| Diagnosing slow LLM inference | Out of repo scope (this is the LLM-systems track, not the FL track) |

---

## PyTorch and ML fundamentals coverage

| Capability | Repo evidence |
|---|---|
| Federated-learning implementation | This whole repo |
| Privacy primitives (DP under composition) | `privacy/dp.py` — the FL leg of the privacy story |
| Comfortable with PyTorch and ML fundamentals | Roadmap Phases 1.1 (CNN), 1.4 (centralised baseline), 5 (DP-SGD per-sample gradients) |

---

## 2026 SOTA enhancements

| Concept | Repo evidence |
|---|---|
| Li 2020 vs Khaled 2020 convergence bounds | `docs/algorithms.md` §2.1 — names both, distinguishes assumptions |
| SCAFFOLD Option I vs Option II | `docs/specifications.md` §6 + `docs/algorithms.md` §4 — Option II is the default; Option I is derivation-only |
| FedProx μ ranges (cross-silo 0.001–0.01, cross-device 0.01–0.1) | `docs/algorithms.md` §3 (μ table) |
| Three-way cross-silo vs cross-device table | `docs/design-decisions.md` D9 |
| FedLoRA family | Roadmap Phase 10 + `docs/algorithms.md` §7 + `docs/references.md` §"Heterogeneous LoRA" |
| One-shot FL | `docs/references.md` §"One-shot FL"; not implemented |
| Byzantine-robust aggregators (Krum → Multi-Krum → Median → Bulyan → Liu 2023) | Roadmap Phase 8 + `docs/specifications.md` §11 |
| SecAgg full protocol | `privacy/secagg.py` skeleton; full Bonawitz protocol explicitly out of scope per D7 |
| LLM inference SOTA (vLLM V1, EAGLE-3, DistServe, SpinQuant, KIVI, MoE) | Out of repo; this is the LLM-systems track, not the FL track |
| Confidential computing + zkML | `docs/references.md` §"Confidential computing" |
| Competition / customer / regulation landscape | `docs/field-evolution.md` §5 |

---

## How to navigate this map

When exploring an algorithmic concept:
1. Find the row above to locate the implementing file.
2. Open the relevant repo file to read the concrete code.
3. Open the matching `results/*.png` if the concept is empirical.

When a concept is something this repo does NOT cover (e.g. PUMA,
EPEAgents, zkML), the explicit "not implemented" notes in
`docs/design-decisions.md` document the scope boundary and why it was
drawn there.
