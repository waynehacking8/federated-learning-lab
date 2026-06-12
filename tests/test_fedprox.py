"""FedProx proximal gradient correctness.

The grad-hook implementation adds mu * (w - w_global) to p.grad after
backward. This must be exactly equivalent to autograd through the
explicit local objective F(w) + (mu/2) * ||w - w_global||^2.
"""

from __future__ import annotations

import torch
from torch import nn

from fl.algorithms.fedprox import attach_proximal_hook


def _make_batches(seed: int = 0, n_batches: int = 3):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(8, 4, generator=g), torch.randint(0, 3, (8,), generator=g))
        for _ in range(n_batches)
    ]


class _FakeClient:
    """Minimal stand-in for fl.client.Client's hook contract."""

    def __init__(self, model: nn.Module, batches, lr: float):
        self.model = model
        self.batches = batches
        self.lr = lr
        self.grad_hook = None
        self.device = torch.device("cpu")

    def local_update(self, model: nn.Module, global_state: dict):
        self.model.load_state_dict(global_state)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        for xb, yb in self.batches:
            opt.zero_grad(set_to_none=True)
            loss_fn(self.model(xb), yb).backward()
            if self.grad_hook is not None:
                self.grad_hook(self.model)
            opt.step()
        return {k: v.clone() for k, v in self.model.state_dict().items()}, 8


def test_proximal_hook_matches_explicit_loss_autograd() -> None:
    mu, lr = 0.5, 0.1
    batches = _make_batches()
    torch.manual_seed(42)
    init = nn.Linear(4, 3)
    global_state = {k: v.clone() for k, v in init.state_dict().items()}

    # Path A: hook-based FedProx.
    model_a = nn.Linear(4, 3)
    client = _FakeClient(model_a, batches, lr)
    attach_proximal_hook(client, mu)
    state_a, _ = client.local_update(model_a, global_state)

    # Path B: explicit proximal term through autograd.
    model_b = nn.Linear(4, 3)
    model_b.load_state_dict(global_state)
    w_global = [p.detach().clone() for p in model_b.parameters()]
    opt = torch.optim.SGD(model_b.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in batches:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model_b(xb), yb)
        prox = sum(((p - gp) ** 2).sum() for p, gp in zip(model_b.parameters(), w_global))
        (loss + 0.5 * mu * prox).backward()
        opt.step()

    for k in state_a:
        torch.testing.assert_close(
            state_a[k], model_b.state_dict()[k], atol=1e-6, rtol=1e-5
        )


def test_mu_zero_is_plain_sgd() -> None:
    batches = _make_batches(seed=1)
    torch.manual_seed(7)
    init = nn.Linear(4, 3)
    global_state = {k: v.clone() for k, v in init.state_dict().items()}

    model_a = nn.Linear(4, 3)
    client = _FakeClient(model_a, batches, lr=0.1)
    attach_proximal_hook(client, mu=0.0)  # no-op by contract
    state_a, _ = client.local_update(model_a, global_state)

    model_b = nn.Linear(4, 3)
    plain = _FakeClient(model_b, batches, lr=0.1)
    state_b, _ = plain.local_update(model_b, global_state)

    for k in state_a:
        torch.testing.assert_close(state_a[k], state_b[k])
