# Specifications

Function-level technical specifications, hyperparameters, and
reference numbers. Agents implementing the code should treat the
formulas, shapes, and benchmarks here as contracts.

---

## 1. Model — `fl/models/cnn.py`

### Architecture

```
Conv2d(1, 16, kernel_size=5, padding=0)   →   ReLU   →   MaxPool2d(2)
Conv2d(16, 32, kernel_size=5, padding=0)  →   ReLU   →   MaxPool2d(2)
Flatten
Linear(32 * 4 * 4, 64)                    →   ReLU
Linear(64, 10)
```

Input: `(batch, 1, 28, 28)`. Output: `(batch, 10)` logits.
Parameter count: 21,706 (verify with `sum(p.numel() for p in model.parameters())`).

### Acceptance criteria

- Centralized SGD baseline: ≥ 0.99 test accuracy after 10 epochs.

---

## 2. Data partitioning — `fl/datasets/mnist_partition.py`

### IID

```python
def iid(targets, num_clients, seed) -> list[list[int]]:
    # Shuffle indices, split equally.
    # Each client gets len(targets) // num_clients samples.
```

### Label-skew

```python
def label_skew(targets, num_clients, classes_per_client, seed):
    # 1. Group indices by class.
    # 2. For each client, sample `classes_per_client` of 10 classes.
    # 3. Split each class's index pool equally among clients holding
    #    that class.
```

### Dirichlet-α (Hsu, Qi, Brown 2019)

```python
def dirichlet(targets, num_clients, alpha, seed):
    # For each class c:
    #   p_c ~ Dir(alpha * ones(num_clients))  # proportions per client
    #   Assign that class's samples to clients in proportion to p_c.
    # Smaller alpha => more skew; alpha=100 ≈ IID.
```

### Acceptance criteria

- Total samples across all clients = `len(targets)` (no overlap or loss).
- For `dirichlet(alpha=0.1)`: at least 50% of clients have a
  dominant class > 50% of their dataset.
- For `dirichlet(alpha=100)`: max class share per client ≤ 20%.

---

## 3. Server / Client API — `fl/server.py`, `fl/client.py`

### `Server`

```python
@dataclass
class Server:
    global_model: nn.Module        # current global state
    aggregator: Aggregator         # plug-in aggregation strategy
    clients: list[Client]
    test_loader: DataLoader

    def run_round(self, round_index: int) -> dict:
        # 1. Sample clients (uniform subsample at participation_rate).
        # 2. Broadcast current state_dict (CPU copy).
        # 3. Each selected client returns (new_state_dict, n_samples).
        # 4. Aggregate.
        # 5. Evaluate on test_loader; return {round, test_loss, test_acc}.
```

### `Client`

```python
@dataclass
class Client:
    client_id: int
    local_indices: list[int]    # indices into the shared training set
    device: torch.device
    local_epochs: int
    local_lr: float
    batch_size: int

    def local_update(self, global_state: dict) -> tuple[dict, int]:
        # Load global_state into a local model copy.
        # Train for local_epochs over local_indices with SGD.
        # Return updated state_dict (CPU) and n_samples used.
```

### Aggregator protocol

```python
class Aggregator(Protocol):
    def aggregate(
        self,
        client_states: list[dict],
        sample_sizes: list[int],
    ) -> dict: ...
```

### Acceptance criteria

- After 1 round with one client doing 1 epoch on MNIST, the global
  model's test loss strictly decreases.

---

## 4. FedAvg — `fl/algorithms/fedavg.py`

### Algorithm

```
w_{t+1} = Σ_k (n_k / n) · w_t^k
```

implemented as weighted average over `state_dict()` keys.

### Notes

- Average **only** floating-point tensors. Skip integer tensors
  (e.g., BatchNorm `num_batches_tracked`) — they're meaningless to
  average.
- The state dict structure must be identical across clients
  (assert this; helpful debugging).

### Acceptance criteria

- For 10 IID clients on MNIST, 50 rounds × 5 local epochs:
  test accuracy ≥ 0.97.

---

## 5. FedProx — `fl/algorithms/fedprox.py`

### Local objective

```
L_k(w) = F_k(w) + (μ/2) · ‖w − w_global‖²
```

### Implementation

- Snapshot `w_global` at the start of each round (do not gradient-track it).
- During local SGD: compute proximal gradient = `μ · (w_local - w_global)`
  and add to the standard loss gradient before the optimizer step.
- Aggregation is identical to FedAvg.

### Acceptance criteria

- On Dirichlet(α=0.1), FedProx with `μ=0.01` achieves ≥ 2 percentage
  points higher final accuracy than FedAvg over 50 rounds, both with
  5 local epochs.

---

## 6. SCAFFOLD — `fl/algorithms/scaffold.py`

### State

- Server keeps `c_global` — a state-dict-shaped tensor pytree
  initialized to zeros.
- Each client keeps `c_local` — same shape, also zeros initially.

### Local update

For each batch:

```
g = ∇ L(w_local; batch)
g̃ = g - c_local + c_global
w_local ← w_local - lr · g̃
```

### End-of-round updates (client side)

```
Option II (the default, simpler form):
    c_local_new = c_local - c_global + (1 / (K * lr)) · (w_global - w_local)
    Δc = c_local_new - c_local
    return (delta_w, delta_c, n_samples)
```

Where K = total local steps taken this round.

### End-of-round (server side)

```
w_global ← w_global + (1/N) · Σ delta_w_k
c_global ← c_global + (S/N) · Σ delta_c_k
```

Where N = total clients, S = number of clients participating.

### Acceptance criteria

- Communication payload per round is approximately 2× FedAvg's
  (both `delta_w` and `delta_c` of the same shape).
- On Dirichlet(α=0.1), SCAFFOLD reaches a given accuracy threshold
  (e.g., 0.90) in fewer rounds than FedAvg.

---

## 7. DP-SGD — `privacy/dp.py`

### Algorithm

For each minibatch:

```
1. Compute per-sample gradients g_i for each example i.
2. Clip each g_i to L2 norm C:
       g_i ← g_i · min(1, C / ‖g_i‖_2)
3. Average and add noise:
       ĝ = (1 / B) · (Σ_i g_i + N(0, σ² C² · I))
4. Optimizer step with ĝ.
```

### Implementation notes

- Per-sample gradients require either `torch.func.vmap` or
  `opacus.GradSampleModule`. For pedagogical clarity in this
  prototype, use a manual loop over the batch (slow but obvious).
- Recommended hyperparameters (DP-FedAvg starting point):
  - `C = 1.0`
  - `σ = 1.0`
  - batch size 32, local epochs 1

### Acceptance criteria

- With `(C, σ) = (1.0, 1.0)` on IID MNIST, test accuracy ≥ 0.90
  after 100 rounds.
- With `(C, σ) = (1.0, 5.0)` (more noise), accuracy degrades
  monotonically.

---

## 8. SecAgg skeleton — `privacy/secagg.py`

### Additive secret sharing

For a vector `x_k` from client k and N total clients:

```
1. Generate N-1 random vectors r_1, ..., r_{N-1} of the same shape as x_k.
2. Compute the final share: r_N = x_k - Σ_{i=1}^{N-1} r_i
3. Send each r_j to peer j (i.e., k holds nothing of its own share).
4. Each peer sums all shares it receives.
5. Server sums the peers' sums to recover Σ_k x_k WITHOUT seeing
   any individual x_k.
```

### Acceptance criteria

- A demonstration notebook with 3 clients and toy vectors shows:
  - Each per-peer aggregate is uninformative about individual x_k.
  - The server-side sum equals (within float tolerance) the true
    sum of x_k.

---

## 9. Reference benchmarks

Expected metrics on this prototype (single-host GPU, MNIST):

| Configuration | Rounds | Test acc | Notes |
|---|---|---|---|
| Centralized SGD (baseline) | 10 epochs | ≥ 0.99 | Ground truth |
| FedAvg, IID, K=10, E=5 | 50 | ≥ 0.97 | Phase 1.4 gate |
| FedAvg, Dir(α=0.1), K=10, E=5 | 50 | ~0.85 | Drift visible |
| FedProx, Dir(α=0.1), μ=0.01 | 50 | ≥ 0.87 | Phase 3 gate |
| SCAFFOLD, Dir(α=0.1) | 50 | ≥ 0.90 | Phase 4 gate |
| DP-FedAvg, IID, (C=1, σ=1) | 100 | ≥ 0.90 | Phase 5 gate |
| FedPer, Dir(α=0.1) — mean per-client acc | 50 | ≥ FedAvg + 3 | Phase 7 gate |
| Median, IID, n=10, f=2 sign-flip | 50 | ≥ baseline − 5 | Phase 8 gate |
| FedAdam, Dir(α=0.1) | 50 | rounds-to-target ≤ 0.8 × FedAvg's | Phase 9 gate |
| FedIT, IID, AG News | 20 | ≥ 0.9 × centralized LoRA | Phase 10 gate |

If your numbers are more than 3 percentage points below these,
suspect a bug rather than the design.

### Training time budget (single GPU)

| Run | Approx wall clock |
|---|---|
| 50 rounds × 10 clients × 5 epochs (MNIST CNN) | 5–10 min |
| 50 rounds × 100 clients × 5 epochs | 30–60 min |
| DP-SGD 100 rounds × 10 clients × 1 epoch | 15–30 min |

If you see these times triple, the per-sample gradient loop is
probably running on CPU.

---

## 10. FedPer (personalized FL) — `fl/algorithms/fedper.py`

### Split

Reuse `make_mnist_cnn`. Partition the parameter set into:

- **Shared body** = both `Conv2d` layers + the first `Linear(32*4*4, 64)`
  (and ReLUs / pools — they have no parameters).
- **Per-client head** = the final `Linear(64, 10)`.

Names in `state_dict()` should be inspected programmatically — do
not hard-code parameter names; instead define a helper
`is_shared(param_name) -> bool` that the aggregator consults.

### Aggregation

```
shared_t+1 = Σ (n_k / n) · shared_t^k        # FedAvg over body only
head_t+1^k = head_t^k                        # each client keeps its own
```

The head never moves through the server. Document this in the
client's `local_update` docstring.

### Acceptance criteria

- On Dirichlet(α=0.1) with 10 clients, FedPer's **mean per-client
  test accuracy** (each client evaluated on its own held-out
  partition) is ≥ 3 percentage points higher than FedAvg's at
  round 50.
- The global metric (one classifier head evaluated on the union test
  set) is allowed to be lower than FedAvg's — that's the point.

---

## 11. Robust aggregators — `fl/algorithms/robust.py`

Drop-in `Aggregator` implementations following the same protocol as
`fedavg`.

### Coordinate-wise median

```
w_{t+1}[i] = median_k(w_t^k[i])
```

Element-wise over flattened parameters.

### Krum / Multi-Krum

For each candidate `w_t^k`, compute the sum of squared distances to
its `n - f - 2` closest peers. Pick the candidate with the smallest
score (Krum) or average the top-`m` smallest (Multi-Krum).

### Trimmed mean

Sort each coordinate across clients, drop the top and bottom `β` (e.g.
`β = f`), average the rest.

### Bulyan (stretch)

Run Multi-Krum to pick a candidate pool of size `≥ 2f + 3`, then
coordinate-wise trimmed mean over that pool.

### Acceptance criteria

- Define a sign-flip Byzantine client that returns `-w_t^k` instead
  of `w_t^k` (clipped to reasonable norm to look credible).
- With `n = 10`, `f = 2`: median and Krum keep IID accuracy within
  5 points of the no-attack baseline; FedAvg degrades by ≥ 20
  points.
- Document the **Liu 2023 caveat**: these aggregators silently
  degrade under Dir(α=0.1) because honest-client divergence is
  comparable to attacker divergence.

---

## 12. Deep Leakage from Gradients demo — `privacy/dlg.py`

A pedagogical reproduction, not a research-grade attack.

### Setup

- Single MNIST image `x`, single CNN forward pass, capture gradient
  `g* = ∇L(model(x), y)`.
- Initialise dummy `x' ~ N(0, 1)` and dummy `y' ~ uniform softmax`.
- Iteratively minimise `‖∇L(model(x'), y') - g*‖²` over `(x', y')`.

### Acceptance criteria

- Without DP: `x'` becomes visually recognisable as the original
  digit within `≤ 500` LBFGS steps. Save a side-by-side figure.
- With DP-SGD `(C=1.0, σ=1.0)` applied to `g*` before sharing: the
  reconstruction fails to converge to anything recognisable.
- Output figure: `results/dlg_with_and_without_dp.png` — two rows,
  before/after DP.

This demo is the empirical answer to "why FL still needs DP and
SecAgg even though it only shares gradients".

---

## 13. FedOpt — `fl/algorithms/fedopt.py`

Treat the round's aggregated delta `Δ̄ = Σ (n_k / n)(w_local_k - w_global)`
as a pseudo-gradient and apply a server-side optimizer.

### FedAdam update

```
g̃_t = -Δ̄                                   # treat negative average delta as a gradient
m_t = β1 · m_{t-1} + (1 - β1) · g̃_t
v_t = β2 · v_{t-1} + (1 - β2) · g̃_t²
w_{t+1} = w_t - lr_server · m_t / (sqrt(v_t) + τ)
```

`τ = 1e-3` is the "adaptivity floor" from Reddi et al. 2020 and is
larger than in standard Adam.

### Implementation notes

- Maintain `(m_t, v_t)` as state on the server, shapes matching the
  flat parameter vector.
- Make the optimizer choice (Adam / Yogi / Adagrad) a constructor
  argument.
- Acceptance: on Dir(α=0.1) with K=10 clients, E=5, FedAvg + FedAdam
  reaches the FedAvg target accuracy in ≥ 20% fewer rounds, OR
  exceeds the final accuracy by ≥ 1 percentage point.

---

## 14. FedLoRA prototype — `fl/algorithms/fedlora.py`

### Base model

A small encoder-only transformer (DistilBERT or similar) loaded
frozen. LoRA adapters attached to the attention `query` and `value`
projections at rank `r ∈ {4, 8}`. Use the `peft` library to avoid
re-implementing LoRA from scratch.

### Federation

Each client owns adapter weights `(A_k, B_k)` per attached layer.
Only adapters cross the network — the base model is shared but
never aggregated.

### FedIT (baseline)

Aggregate `A` and `B` with weighted averaging (same recipe as
FedAvg, restricted to adapter parameters).

### FedSA-LoRA (selective aggregation)

Aggregate `A` only. Each client keeps `B_k` across rounds. The
server delivers updated `Ā` each round; clients reconstruct their
local model as `W + B_k Ā`.

### Acceptance criteria

- IID partition: FedIT reaches ≥ 90% of the centralised-LoRA
  fine-tune accuracy within 20 rounds.
- Dir(α=0.1) partition: FedSA-LoRA's mean per-client accuracy is ≥
  FedIT's by ≥ 1 percentage point AND payload-per-round is roughly
  half (only `A` is shared).
- Save `results/fedlora_communication_vs_accuracy.png`: x-axis =
  total bytes communicated, y-axis = mean per-client accuracy.
