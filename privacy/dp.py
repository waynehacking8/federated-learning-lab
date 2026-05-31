"""Differential Privacy via DP-SGD (Abadi et al., 2016).

Two primitives:
    1. Per-sample gradient clipping with norm C.
    2. Gaussian noise with scale sigma * C added to the sum of clipped
       per-sample gradients (then averaged by batch size).

Plus a thin trainer ``DPSGDClient`` and a naive composition-based
(epsilon, delta) estimator. For production-grade accounting use
Opacus's RDP accountant -- this estimator is intentionally simple.

Implementation note: per-sample gradients are computed via
``torch.func.grad + vmap`` rather than the manual Python loop
suggested in docs/specifications.md section 7. Both approaches were
called out by the spec; vmap is identical mathematically but ~10-50x
faster, which keeps the DP-FedAvg demo within a reasonable wall-clock
budget. The two primitives below remain explicit and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def clip_per_sample_gradients(per_sample_grads: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip each row of ``per_sample_grads`` to L2 norm ``max_norm``.

    ``per_sample_grads`` has shape (batch, ...). Per-sample norms are
    computed across all dims except the first.
    """
    flat = per_sample_grads.view(per_sample_grads.shape[0], -1)
    norms = flat.norm(dim=1)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    shape = (-1,) + (1,) * (per_sample_grads.dim() - 1)
    return per_sample_grads * scale.view(*shape)


def add_gaussian_noise(grad: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma^2 * I) to ``grad``.

    Returns a new tensor on the same device/dtype as ``grad``.
    """
    if noise_scale <= 0:
        return grad.clone()
    noise = torch.randn_like(grad) * noise_scale
    return grad + noise


def naive_epsilon(noise_scale: float, sample_rate: float, steps: int, delta: float = 1e-5) -> float:
    """Naive composition (Abadi 2016 eq. 3) epsilon estimator.

    This is a strict upper bound that ignores subsampling amplification
    and RDP composition, so it overshoots the true epsilon by 2-3 orders
    of magnitude. Prefer ``rdp_epsilon`` for a meaningful number; this is
    kept only to show how loose naive composition is.
    """
    # Per-step epsilon under (epsilon, delta)-DP for Gaussian mechanism:
    #   epsilon_step approx sqrt(2 * ln(1.25 / delta)) / sigma
    eps_step = math.sqrt(2 * math.log(1.25 / delta)) / max(noise_scale, 1e-12)
    # Strong composition for T steps:
    #   epsilon_total approx sqrt(2 * T * ln(1 / delta)) * eps_step
    return math.sqrt(2 * steps * math.log(1 / delta)) * eps_step


# Standard RDP orders used by Opacus / TF-Privacy.
_RDP_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def rdp_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float = 1e-5
) -> float:
    """Tight (epsilon, delta) via the subsampled-Gaussian RDP accountant.

    Uses Google's ``dp_accounting`` (the engine Opacus wraps). The noise
    multiplier is ``sigma`` (noise std = sigma * C on the C-clipped grad
    sum), the sensitivity is ``C``, and Poisson subsampling at
    ``sample_rate`` provides privacy amplification. ``steps`` is the total
    number of local SGD steps per client.

    Raises ImportError if dp_accounting is unavailable so callers can fall
    back to ``naive_epsilon`` explicitly rather than silently.
    """
    from dp_accounting import dp_event, rdp

    accountant = rdp.RdpAccountant(_RDP_ORDERS)
    event = dp_event.PoissonSampledDpEvent(
        sample_rate, dp_event.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, steps)
    return accountant.get_epsilon(delta)


@dataclass
class DPConfig:
    clip_C: float = 1.0
    noise_sigma: float = 1.0
    delta: float = 1e-5


class DPSGDClient:
    """Client wrapper that applies DP-SGD on local updates.

    Per-sample gradients are vectorised with ``torch.func.vmap`` (see the
    module docstring); both DP primitives -- per-sample L2 clipping to C
    and Gaussian noise with std sigma * C -- are applied explicitly in
    ``local_update``. Sized for the small MNIST CNN.
    """

    def __init__(
        self,
        client_id: int,
        local_indices: list[int],
        train_dataset,
        device: torch.device,
        dp: DPConfig,
        local_epochs: int = 1,
        local_lr: float = 0.05,
        batch_size: int = 32,
    ) -> None:
        self.client_id = client_id
        self.local_indices = local_indices
        self.train_dataset = train_dataset
        self.device = device
        self.dp = dp
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self._loader: Optional[DataLoader] = None
        self.grad_hook = None  # compat with Client protocol

    def _make_loader(self) -> DataLoader:
        if self._loader is None:
            subset = Subset(self.train_dataset, self.local_indices)
            self._loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return self._loader

    def n_samples(self) -> int:
        return len(self.local_indices)

    def local_update(self, model: nn.Module, global_state: dict) -> tuple[dict, int]:
        if not hasattr(self, "_local_model") or self._local_model is None:
            from fl.models.cnn import make_mnist_cnn

            self._local_model = make_mnist_cnn().to(self.device)
        local_model = self._local_model
        local_model.load_state_dict(
            {k: v.to(self.device, non_blocking=True) for k, v in global_state.items()}
        )
        local_model.train()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loader = self._make_loader()

        # Vectorize per-sample gradient computation via torch.func.vmap.
        # This is much faster than a Python per-sample loop and still
        # exposes both DP primitives (clipping + noise) explicitly.
        from torch.func import functional_call, grad, vmap

        params_dict = {n: p.detach() for n, p in local_model.named_parameters()}
        buffers_dict = {n: b for n, b in local_model.named_buffers()}

        def compute_loss(params, buffers, x, y):
            logits = functional_call(local_model, (params, buffers), (x.unsqueeze(0),))
            return loss_fn(logits, y.unsqueeze(0)).mean()

        per_sample_grad_fn = vmap(
            grad(compute_loss), in_dims=(None, None, 0, 0)
        )

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                B = xb.shape[0]

                per_sample_grads = per_sample_grad_fn(params_dict, buffers_dict, xb, yb)

                # Flatten per-sample grads across all params, compute L2 norm per sample.
                flat_per_sample = torch.cat(
                    [g.reshape(B, -1) for g in per_sample_grads.values()], dim=1
                )
                norms = flat_per_sample.norm(dim=1)
                coef = torch.clamp(self.dp.clip_C / (norms + 1e-12), max=1.0)

                # Clip + sum + noise + average + SGD step.
                for name, g in per_sample_grads.items():
                    coef_shape = (-1,) + (1,) * (g.dim() - 1)
                    clipped = g * coef.view(*coef_shape)
                    summed = clipped.sum(dim=0)
                    noisy = summed + torch.randn_like(summed) * (
                        self.dp.noise_sigma * self.dp.clip_C
                    )
                    params_dict[name] = params_dict[name] - self.local_lr * (noisy / B)

        # Write trained params back into the model so state_dict() reflects them.
        with torch.no_grad():
            for n, p in local_model.named_parameters():
                p.data.copy_(params_dict[n])

        return {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}, self.n_samples()
