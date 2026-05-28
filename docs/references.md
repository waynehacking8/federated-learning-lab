# References

Curated reading list, ordered from "required" to "background".

---

## Required papers

### FL core

1. **McMahan et al. (2017)**, *Communication-Efficient Learning of
   Deep Networks from Decentralized Data.* AISTATS. — Introduces
   FedAvg.
2. **Li et al. (2018)**, *Federated Optimization in Heterogeneous
   Networks.* MLSys 2020. — Introduces FedProx.
3. **Karimireddy et al. (2019)**, *SCAFFOLD: Stochastic Controlled
   Averaging for Federated Learning.* ICML 2020. — Introduces
   SCAFFOLD.
4. **Kairouz et al. (2019)**, *Advances and Open Problems in
   Federated Learning.* — Standard survey; section 5 (Non-IID) is
   particularly worth reading.

### Privacy

5. **Abadi et al. (2016)**, *Deep Learning with Differential
   Privacy.* CCS. — DP-SGD; the moments accountant.
6. **Bonawitz et al. (2017)**, *Practical Secure Aggregation for
   Privacy-Preserving Machine Learning.* CCS. — The full SecAgg
   protocol.
7. **Mironov (2017)**, *Rényi Differential Privacy.* CSF. — RDP
   composition; modern privacy accounting.

### Non-IID partitioning methodology

8. **Hsu, Qi, Brown (2019)**, *Measuring the Effects of Non-Identical
   Data Distribution for Federated Visual Classification.* — Defines
   the Dirichlet-α partition scheme used widely since.
9. **Li et al. (2022)**, *Federated Learning on Non-IID Data Silos:
   An Experimental Study.* — Empirical comparison of FedAvg /
   FedProx / SCAFFOLD across many Non-IID regimes.

---

## Modern extensions

10. **Reddi et al. (2020)**, *Adaptive Federated Optimization.* — Treats
    the server-side aggregation as a single optimizer step, allowing
    Adam / Yogi / Adagrad to replace simple averaging. The FedOpt /
    FedAdam family.
11. **Acar et al. (2021)**, *Federated Learning Based on Dynamic
    Regularization.* — FedDyn; adaptive regularizer.
12. **Wang et al. (2020)**, *Tackling the Objective Inconsistency
    Problem in Heterogeneous Federated Optimization.* — FedNova;
    normalization for heterogeneous local steps.

---

## LLM + FL

13. **Babakniya et al. (2023)**, *SLoRA: Federated Parameter
    Efficient Fine-Tuning of Language Models.* — Federated LoRA
    fine-tuning.
14. **Yi et al. (2024)**, *FedLoRA: Model-Heterogeneous Personalized
    Federated Learning for Open-Source LLMs.* — Multiple LoRA ranks
    across heterogeneous clients.

### Heterogeneous LoRA — the rank-mismatch problem family

The fundamental issue: across clients, neither rank nor adapter
content is homogeneous, and naïvely averaging $A_i$ and $B_i$
separately incurs aggregation noise because $\overline{B}\,\overline{A}
\neq \overline{BA}$. Four lines of attack:

- **FlexLoRA** — reconstruct each client's $B_iA_i$ at the server,
  perform full-size SVD, redistribute to each client at its own rank.
  Cost: server-side SVD.
- **HetLoRA** — zero-pad each client's LoRA to a common shape, then
  sparsity-weight the aggregation. Cost: brittle to outlier clients.
- **FLoRA** — stack rather than average, eliminating the
  $\overline{BA}$ approximation error. Cost: client-side merge.
- **SLoRA** — two-stage sparse pretraining initialization to absorb
  data heterogeneity before LoRA tuning.
- **FedSA-LoRA** (ICLR 2025, arXiv:2410.01463) — share only $A$,
  keep $B$ local. Rationale: $A$ encodes shared structure, $B$
  encodes client-specific structure. This single change reduces
  communication, makes model-inversion attacks harder, and provides
  personalization for free.

### One-shot FL (2025 frontier)

Multi-round FL is expensive and amplifies privacy leakage. **One-shot
FL** compresses the protocol to a single communication round.

- **FedDISC**, **DENSE**, **Co-Boosting** — server-side diffusion or
  rectified-flow models synthesize pseudo-data; clients upload
  distilled samples; the server distills the global model from the
  ensemble. Outperforms multi-round FedAvg by up to 21.73% on medical
  imaging tasks (see survey arXiv:2505.02426).

### Byzantine-robust FL and its modern caveats

- **Blanchard et al. (NeurIPS 2017)** — **Krum** / **Multi-Krum**:
  pick the update with the smallest sum of distances to its closest
  $n-f-2$ neighbors. Tolerates $f < (n-2)/2$ adversaries.
- **Yin et al. (ICML 2018)** — **Coordinate-wise Median**: cheap,
  effective against outliers, weak against colluding attacks.
- **Mhamdi et al. (ICML 2018)** — **Bulyan**: two-stage Krum +
  trimmed-mean; strongest theoretical guarantee, $O(n^2 d)$ cost.
- **Liu et al. (ICML 2023)** — robust aggregation under heterogeneous
  data via gradient splitting. **Most prior Byzantine-robust
  aggregators silently fail under Non-IID data** because honest
  clients themselves diverge.
- **NDSS 2025** — systematic re-examination of prior Byzantine-robust
  aggregators; finds attack surface that earlier evaluations missed.

### A 2024 caveat for SCAFFOLD

- **arXiv:2411.16167** — *Mind the Cost of Scaffold for Federated
  Learning.* Shows that SCAFFOLD's control variates **amplify backdoor
  attack contagion** by carrying malicious gradient direction in
  $c_i$. A non-trivial robustness regression versus FedAvg in
  adversarial deployments.

---

## Production frameworks (for context, not reimplemented)

15. **Flower** (`flwr.dev`) — Pythonic FL framework with real RPC. As
    of v1.23 (2025-11), introduces **SuperLink + SuperNode** as
    long-lived processes, dynamic supernode registration, and
    per-run virtualenv isolation. Plus **FlowerTune** (2025-06): an
    LLM-fine-tuning federated leaderboard across NLP / finance /
    medical / coding.
16. **NVIDIA FLARE** — Enterprise FL platform; SCAFFOLD and FedProx
    implementations. As of 2025, integrated with Meta **ExecuTorch**
    for on-device training (cross-device), and ships **Confidential
    Federated AI** that runs aggregation inside attested GPU TEEs.
17. **FedML** — Research-oriented; supports cross-silo and cross-
    device modes.
18. **Google Federated Computing Platform** (open-sourced at
    `google-parfait/federated-compute`) — production cross-device
    stack behind Gboard / Android. TensorFlow Federated computations
    compiled to Android artifacts and scheduled via
    `FederatedComputeScheduler`.
19. **OpenFL** (Intel, now Linux Foundation) — TaskRunner with mTLS +
    Intel SGX / TDX confidential computing; Workflow API supports
    non-traditional patterns including vertical FL with private set
    intersection.

---

## Confidential computing and verifiable inference (2024–2026)

The privacy story for production federated systems and on-prem LLM
deployment now extends well beyond DP and SecAgg.

### Apple Private Cloud Compute (2024)

- **Apple Security Engineering and Architecture (2024)**, *Private
  Cloud Compute: A new frontier for AI privacy in the cloud.* —
  Custom Apple silicon, append-only transparency log of binary
  measurements, third-party-verifiable attestation, no remote shell,
  ephemeral storage keys. Apple-controlled supply chain is the
  argument for why generic TEE (SGX / SEV) cannot match this; in
  sovereign-AI contexts, this is also Apple's structural disadvantage
  versus on-prem deployments.

### NVIDIA confidential GPUs

- **NVIDIA (2023)**, *Confidential Computing on H100 GPUs* (whitepaper).
  Hardware root-of-trust, encrypted GPU memory, performance counters
  disabled, SPDM channel to CPU TEE (AMD SEV-SNP / Intel TDX).
- **Liu et al. (2024)**, arXiv:2409.03992 — measured H100 / H200
  confidential-mode overhead on LLM inference. **70B model overhead
  approaches zero**; short-prompt TTFT pays 19–26% due to PCIe
  bounce-buffer encryption.
- **NVIDIA Blackwell B200** — first TEE-I/O capable GPU; NVLink
  traffic encrypted in-band, removing bounce-buffer overhead.

### TEE attacks (2024–2026) — read these to understand the threat
ceiling

- **TEE.Fail (2025-10)** — sub-$1k DDR5 bus interposer extracts
  attestation keys from SGX / TDX / SEV-SNP simultaneously.
- **Battering RAM (2025-10)** — $50 DDR4 interposer breaks SGX and
  SEV-SNP.
- **Heracles (CCS 2025)** — chosen-plaintext attack on SEV-SNP via
  AES-XEX ciphertext side-channel.
- **GPUBreach (2026-04)** — GPU rowhammer corrupts GPU page tables,
  chains to NVIDIA driver bug for full host compromise. IOMMU does
  not stop it.

Conclusion: TEE is necessary but not sufficient for sovereign-AI
deployment. Physical security and ephemeral key rotation must
accompany hardware TEE.

### zkML and verifiable inference (current state)

- **Lagrange DeepProve-1 (2025-08)** — first zk-proof for full GPT-2
  inference; 54×–158× faster than EZKL, extends to Gemma3.
- **Sun et al. (NeurIPS 2024)** — **zkLLM**: 13B-parameter zk-proof in
  ~15 minutes, <200 KB proof size.
- **zkPyTorch (2025-03)** — direct PyTorch → ZK circuit compilation;
  VGG-16 in 2.2 s, but Llama-3 still ~150 s per token.
- **EZKL** — strongest on tabular ML; transformer LLM not yet
  supported.

Practical takeaway for 2026 sovereign-AI deployment: zkML is **viable
for offline audit / spot-checking**, not for inline LLM serving.

### FHE on GPU

- **Zama Concrete-ML 1.9 (2025-04)** — TFHE-rs with GPU acceleration
  ~30× over CPU; encrypted LoRA fine-tuning supported.
- **EncryptedLLM (ICML 2025)** — GPT-2 forward pass ~200× over CPU
  baseline.
- **HEngine (ACM TACO 2025)** — Microsoft SEAL homomorphic
  multiplication ~218× over CPU; warp-shuffle NTT optimization.

Practical takeaway: FHE LLM is still minutes-per-token for 7B
models; production niche is restricted to encrypted embedding lookup
or small classifiers.

---

## Sovereign / private LLM deployment landscape (2025–2026)

Useful both as interview context and as the broader thesis behind why
this lab exists.

- **Apple PCC (2024)** — consumer-side privacy benchmark.
- **Naver HyperCLOVA X (2026-01)** — first central-bank sovereign AI
  (Bank of Korea); fully on-prem.
- **Mistral (EU)** — €1.7B Series B (2025-09); deep public-sector
  alignment in France, Germany, Luxembourg.
- **Sakana AI (Japan)** — $135M Series B at $2.65B (2025-11), MUFG /
  Khosla / Lux / **In-Q-Tel** investors.
- **Taiwan AI Labs FedGPT AgentTeam (2025-06)** — on-prem deployment
  on a single NVIDIA H200; Taiwan-cognition benchmark 81.4 versus
  38–44 for general-purpose models.
- **Phison aiDAPTIV+ (2025)** — SSD-tier VRAM extension; Llama 3.1
  405B fine-tune on 2 GPUs at the SC25 demo.
- **DeepSeek V3 / R1 / V4 (2025)** — open-weight + Huawei Ascend
  stack; the open-source rebuttal to closed sovereign AI.

---

## Suggested reading order

---

## Suggested reading order

If approaching FL fresh:

1. McMahan 2017 sections 1–3 — see the algorithm in its original
   form.
2. Implement FedAvg on IID MNIST yourself — the code is short and
   illuminating.
3. Kairouz 2019 section 5 — understand Non-IID terminology.
4. Hsu 2019 — see how to systematically vary Non-IID severity.
5. Li 2018 (FedProx) and Karimireddy 2019 (SCAFFOLD) — compare the
   two drift-control philosophies.
6. Abadi 2016 — the privacy machinery.
7. Reddi 2020 — once the trio is internalized, learn the server-side
   optimizer perspective.

This matches the implementation phases in `roadmap.md`.
