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

---

## Production frameworks (for context, not reimplemented)

15. **Flower** (`flwr.dev`) — Pythonic FL framework with real RPC.
16. **NVIDIA FLARE** — Enterprise FL platform; SCAFFOLD and FedProx
    implementations.
17. **FedML** — Research-oriented; supports cross-silo and cross-
    device modes.

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
