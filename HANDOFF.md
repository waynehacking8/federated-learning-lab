# Handoff — Read This First

You are an AI agent continuing development on this repository in a
new environment. This file is the single source of truth for what
to do next. Read it end-to-end before acting.

---

## What this repo is

`federated-learning-lab` is a from-scratch implementation of canonical
federated-learning algorithms (FedAvg, FedProx, SCAFFOLD) with
optional differential-privacy and secure-aggregation layers,
evaluated on Non-IID partitions of MNIST.

GitHub: https://github.com/waynehacking8/federated-learning-lab

---

## Current status

- All documentation is in place: `README.md`, `docs/algorithms.md`,
  `docs/specifications.md`, `docs/design-decisions.md`,
  `docs/roadmap.md`, `docs/references.md`.
- All Python modules under `fl/`, `privacy/`, `scripts/` raise
  `NotImplementedError` with TODO blocks in their docstrings.
- `tests/` contains failing pytest tests for partition utilities,
  the FedAvg aggregator, and DP-SGD primitives.

---

## This environment's advantage

You have a GPU available. Use it for:

- Training the MNIST CNN at every phase (centralized baseline,
  FedAvg, FedProx, SCAFFOLD, DP-FedAvg).
- (Stretch) FedLoRA — federated fine-tuning of a small language
  model. Adapter weights are tiny so the federation cost is
  manageable; this is the natural home for the LLM-side experiments.

PyTorch device discipline (per `AGENTS.md`):
- Build the model on the GPU once; serialize state dicts as CPU
  tensors.
- Each client moves model + batch to GPU at the start of
  `local_update`, leaves there until return.
- Aggregation runs on CPU.

---

## Working order for this session

Follow `docs/roadmap.md`. Minimum end-state: Phase 4 done
(three-way comparison FedAvg / FedProx / SCAFFOLD on Dir(α=0.1)
MNIST). Concretely:

1. **Setup**:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   # Reinstall torch + torchvision matching the local CUDA version.
   pip install -r requirements.txt
   pip install pytest
   ```

2. **Phase 1.1 — data and model**:
   - Read `docs/specifications.md` section 1 (CNN) and 2 (IID
     partitioning).
   - Implement `fl/models/cnn.py::make_mnist_cnn` and
     `fl/datasets/mnist_partition.py::iid`.
   - Verify: `pytest tests/test_partition.py::test_iid_no_loss`
     green; CNN parameter count is 21,706.

3. **Phase 1.2 — server and client API**:
   - Read specifications section 3.
   - Implement `fl/server.py` and `fl/client.py`.
   - Smoke test: one client × one local epoch reduces global test
     loss.

4. **Phase 1.3 — FedAvg**:
   - Read specifications section 4.
   - Implement `fl/algorithms/fedavg.py::aggregate`.
   - Make `pytest tests/test_fedavg.py` green.

5. **Phase 1.4 — IID baseline experiment**:
   - Read specifications section 9 for target metrics.
   - Implement `scripts/run_fedavg_mnist.py`.
   - Run: `python -m scripts.run_fedavg_mnist --num-clients 10
     --partition iid --rounds 50 --local-epochs 5`.
   - Acceptance: test accuracy ≥ 0.97; save curve to
     `results/fedavg_iid.png`.

6. **Phase 2 — Non-IID partitions**:
   - Implement `label_skew` and `dirichlet` in
     `fl/datasets/mnist_partition.py`. Make
     `pytest tests/test_partition.py` fully green.
   - Run the same FedAvg experiment on Dir(α=0.1); confirm
     accuracy degrades vs IID.

7. **Phase 3 — FedProx**:
   - Read specifications section 5.
   - Implement proximal-term gradient correction in
     `fl/algorithms/fedprox.py`.
   - Acceptance: on Dir(α=0.1), beat FedAvg by ≥ 2 percentage
     points at round 50 with μ=0.01.

8. **Phase 4 — SCAFFOLD**:
   - Read specifications section 6.
   - Implement control-variate machinery in
     `fl/algorithms/scaffold.py`.
   - Acceptance: reach 0.90 accuracy in fewer rounds than FedAvg
     on Dir(α=0.1).
   - Save three-way comparison plot to
     `results/three_way_comparison.png`.

9. **(Stretch) Phase 5 — DP-SGD**:
   - Implement primitives in `privacy/dp.py`.
   - Make `pytest tests/test_dp.py` green.
   - Run DP-FedAvg with `(C=1, σ=1)`; expect accuracy ≥ 0.90 at
     100 rounds on IID.

10. **(Stretch) FedLoRA** if time remains:
    - Federated fine-tuning of DistilBERT-class model on a small
      text-classification task. Out of current spec; add a
      `D{n}` design-decision entry if you start.

---

## Hard constraints (do NOT violate)

- **No emoji** in any file.
- **Do not virtualize the author's experience.** Wei Cheng (Wayne)
  Chiu has comfortable PyTorch and ML fundamentals but **no
  production FL, DP-SGD, or SMPC experience**. Docs and comments
  must not claim otherwise.
- **PyTorch only**, not JAX or TF (per `design-decisions.md` D4).
- **In-process clients**, not RPC (D3).
- **Custom DP-SGD before Opacus** (D6).
- **Follow `AGENTS.md`** for conventions.

---

## Verification gate before merging any phase

```
pytest tests/ -k "not slow"
git status
```

Then flip checkboxes in `docs/roadmap.md` to `[x]` and commit.

---

## When in doubt

1. `docs/specifications.md` — concrete formulas, hyperparameters,
   reference metrics.
2. `docs/algorithms.md` — algorithmic intent (FedAvg / FedProx /
   SCAFFOLD derivations).
3. `docs/design-decisions.md` — "why this, not that".
4. `AGENTS.md` — conventions.
5. Add a new `D{n}` entry rather than silently contradicting
   an existing decision.
