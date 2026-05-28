# Agent Execution Guide

This document tells an AI agent (e.g., Claude, Cursor, Codex) how to
work this repository productively. Read this **before** opening any
source file.

---

## Repository contract

- Every Python module under `fl/`, `privacy/`, `scripts/` currently
  raises `NotImplementedError` with a TODO block in its docstring.
  The TODO is the **work order**.
- The TODO points to a phase in `docs/roadmap.md`. Each phase has an
  **acceptance criterion** at the bottom of that phase's section.
- Concrete formulas, hyperparameters, expected metric values, and
  reference benchmarks live in `docs/specifications.md`. The roadmap
  says *what* to do; the specifications doc says *how*.

If a TODO is ambiguous, the order of resolution is:

1. Check `docs/specifications.md` for the function-level spec.
2. Check `docs/algorithms.md` for the algorithmic intent (formulas).
3. Check `docs/design-decisions.md` for the "why this and not that".
4. Default to a minimal, defensible choice and record it as a new
   `## D{n}` entry in `docs/design-decisions.md`.

Never silently invent a choice that contradicts an existing doc.

---

## Working order

Implement phases in roadmap order. Each phase should be a separate
commit so the history reads as the actual development sequence.

| Phase | Branch (recommended) | Acceptance gate |
|---|---|---|
| 1.1 Data + model | `feat/data-model` | `pytest tests/test_partition.py::test_iid` green |
| 1.2 Server/Client | `feat/server-client` | client+server can complete one round on tiny synthetic data |
| 1.3 FedAvg aggregator | `feat/fedavg` | `pytest tests/test_fedavg.py` green |
| 1.4 IID baseline | `feat/iid-baseline` | global test acc ≥ 0.97 within 30 rounds on MNIST |
| 2.x Non-IID partitions | `feat/noniid` | Dir(α=0.1) curve diverges from Dir(α=10) by ≥ 5 acc points |
| 3.x FedProx | `feat/fedprox` | beats FedAvg by ≥ 2 acc points on Dir(α=0.1) at 50 rounds |
| 4.x SCAFFOLD | `feat/scaffold` | converges in fewer rounds than FedAvg on Dir(α=0.1) |
| 5.x DP-SGD | `feat/dp-sgd` | acc still ≥ 0.90 at noise scale σ=1.0, clip C=1.0 |
| 6.x SecAgg skeleton | `feat/secagg` | notebook shows server-side sum reconstructs without seeing individual updates |

Do not work on Phase N+1 before Phase N's tests pass.

---

## GPU usage

This lab is designed to run on a single GPU (one host, in-process
client simulation). Important conventions:

- Build the model on the GPU once, then `state_dict()` snapshots are
  CPU tensors (cheaper to serialize).
- Each client's `local_update` moves model + batch to GPU at the
  start, leaves on GPU until return, then moves the returned state
  dict to CPU.
- Aggregation happens on CPU (state dicts are small enough; avoids
  contention with concurrent training).
- For reproducibility, set `torch.manual_seed`,
  `torch.cuda.manual_seed_all`, and `torch.use_deterministic_algorithms(True)`
  at the entry point of any script.

---

## Conventions

- **Python style**: PEP 8, `black` formatting, `ruff` linting,
  type-hinted public functions.
- **Tests**: pytest. New code without a test is incomplete.
- **No emoji in code or docs.**
- **Commit messages**: `<type>: <short description>` (feat / fix /
  refactor / docs / test / chore). Phase number in body.
- **Doc updates**: when you finish a phase, flip its checkboxes in
  `docs/roadmap.md` to `[x]` and add a one-line note about what
  changed since the spec.

---

## What you do NOT need to ask

The following are pre-decided and documented. Do not re-litigate:

- PyTorch (not JAX/TensorFlow). See `design-decisions.md` D4.
- MNIST (not CIFAR-10 initially). See D2.
- In-process client simulation (not RPC). See D3.
- Custom DP-SGD before Opacus. See D6.
- SecAgg skeleton (not full Bonawitz protocol). See D7.

If you have a strong reason to change any of these, write a new
`D{n}` entry first; do not commit code that contradicts a current
decision.

---

## Author background (for grounding tone in docs)

Wei Cheng (Wayne) Chiu — NTUST CS Master's (April 2026 graduate),
LLM / multi-agent systems background. Comfortable with PyTorch and
ML fundamentals. **No production experience in federated learning,
differential privacy implementations, or secure multi-party
computation.** This is a self-study prototype, not a re-implementation
of a product.

When you write docs or comments, do not claim experience the author
does not have.
