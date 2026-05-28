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
