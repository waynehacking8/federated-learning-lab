"""Deep Leakage from Gradients (Zhu et al., 2019) -- pedagogical demo.

Shows that a single training image can be reconstructed from the
gradient it produced, and that DP-SGD noise (clip C + Gaussian sigma)
breaks the reconstruction. This is the empirical answer to "why does FL
still need DP/SecAgg if it only shares gradients?"

Attack: given target gradient g* = dL(f(x), y)/dtheta, optimise a dummy
image x' and dummy label y' to minimise || dL(f(x'), y')/dtheta - g* ||^2
using LBFGS. With no noise x' converges to the original digit; with
DP-noised g* it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DLGConfig:
    steps: int = 300            # LBFGS iterations
    lr: float = 1.0
    dp_clip_C: Optional[float] = None    # if set, clip g* to this L2 norm
    dp_sigma: float = 0.0                # if >0, add N(0, (sigma*C)^2) to g*
    seed: int = 0


def _flat_grad(grads) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def compute_target_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """g* = dL/dtheta for a single (x, y). Returns a list of grad tensors."""
    model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
    return [g.detach().clone() for g in grads]


def _apply_dp(grads, clip_C, sigma, gen):
    """Apply DP-SGD post-processing to a single example's gradient.

    Clip the full gradient vector to L2 norm clip_C, then add Gaussian
    noise with std sigma*clip_C. This is exactly what DPSGDClient does to
    each per-sample gradient before it would be shared.
    """
    if clip_C is None and sigma == 0.0:
        return grads
    flat = _flat_grad(grads)
    out = grads
    if clip_C is not None:
        norm = flat.norm()
        scale = min(1.0, clip_C / (norm.item() + 1e-12))
        out = [g * scale for g in out]
        if sigma > 0.0:
            out = [g + torch.randn(g.shape, generator=gen, device=g.device) * (sigma * clip_C) for g in out]
    return out


def reconstruct(model: nn.Module, target_grads, image_shape, num_classes, cfg: DLGConfig):
    """Recover (x', y') by matching gradients. Returns (x', history of losses)."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed)

    x_dummy = torch.randn(image_shape, generator=gen, device=device, requires_grad=True)
    # Dummy label as learnable logits (soft label), per the DLG paper.
    y_dummy = torch.randn(1, num_classes, generator=gen, device=device, requires_grad=True)

    opt = torch.optim.LBFGS([x_dummy, y_dummy], lr=cfg.lr, max_iter=20)
    params = list(model.parameters())
    losses = []

    def closure():
        opt.zero_grad(set_to_none=True)
        pred = model(x_dummy)
        log_probs = torch.log_softmax(pred, dim=1)
        soft_y = torch.softmax(y_dummy, dim=1)
        loss = -(soft_y * log_probs).sum()
        dummy_grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, target_grads))
        grad_diff.backward()
        losses.append(grad_diff.item())
        return grad_diff

    for _ in range(cfg.steps // 20 + 1):
        opt.step(closure)
        if len(losses) >= cfg.steps:
            break
    return x_dummy.detach(), losses
