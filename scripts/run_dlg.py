"""Phase 8.3 experiment: Deep Leakage from Gradients, with and without DP.

Reconstructs a single MNIST image from its gradient (Zhu et al. 2019),
then repeats with DP-SGD-noised gradients (C=1, sigma=1) and shows the
reconstruction fails.

Activation note: the DLG paper and follow-ups use SMOOTH activations
(sigmoid/tanh), because LBFGS gradient-matching needs continuous second
derivatives -- ReLU + MaxPool (our MnistCNN) have zero/undefined second
derivatives that make the attack unreliable. We therefore run the demo
on a small smooth-activation CNN. This does not weaken the lesson: the
point is "a shared gradient leaks the input, and DP noise stops it",
which is activation-independent. The production MnistCNN is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from privacy.dlg import DLGConfig, compute_target_gradient, reconstruct

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class SmoothCNN(nn.Module):
    """Small smooth-activation net suitable for DLG gradient matching."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, 5, stride=2, padding=2), nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, stride=2, padding=2), nn.Sigmoid(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(12 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.body(x))


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)

    # No normalization so the reconstructed image is in [0,1]-ish pixel space.
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tfm)
    x, y = ds[7]  # a clear digit
    x = x.unsqueeze(0).to(device)
    y = torch.tensor([y], device=device)

    model = SmoothCNN().to(device)
    model.eval()

    # --- Attack without DP ---
    g_clean = compute_target_gradient(model, x, y)
    x_rec_clean, losses_clean = reconstruct(
        model, g_clean, x.shape, 10, DLGConfig(steps=300, lr=1.0, seed=0)
    )

    # --- Attack with DP-SGD noise on the shared gradient ---
    gen = torch.Generator(device=device); gen.manual_seed(0)
    from privacy.dlg import _apply_dp
    g_dp = _apply_dp(g_clean, clip_C=1.0, sigma=1.0, gen=gen)
    x_rec_dp, losses_dp = reconstruct(
        model, g_dp, x.shape, 10, DLGConfig(steps=300, lr=1.0, seed=0)
    )

    # Reconstruction quality: MSE to the true image (lower = better leak).
    mse_clean = ((x_rec_clean - x) ** 2).mean().item()
    mse_dp = ((x_rec_dp - x) ** 2).mean().item()

    out = Path("results/dlg"); out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, img, title in [
        (axes[0], x, "original"),
        (axes[1], x_rec_clean, f"no DP\nMSE={mse_clean:.3f}"),
        (axes[2], x_rec_dp, f"DP-SGD (C=1, sigma=1)\nMSE={mse_dp:.3f}"),
    ]:
        ax.imshow(img.detach().cpu().squeeze().clamp(0, 1), cmap="gray")
        ax.set_title(title); ax.axis("off")
    fig.suptitle("Deep Leakage from Gradients -- DP breaks the reconstruction")
    fig.tight_layout()
    fig.savefig("results/dlg_with_and_without_dp.png", dpi=120)
    plt.close(fig)

    # The lesson holds if the no-DP reconstruction is much closer than the DP one.
    leak_gate = mse_clean < 0.5 * mse_dp and mse_clean < 0.1

    import json
    (out / "metrics.json").write_text(json.dumps({
        "mse_no_dp": mse_clean, "mse_dp": mse_dp,
        "final_match_loss_no_dp": losses_clean[-1], "final_match_loss_dp": losses_dp[-1],
        "leak_demonstrated": bool(leak_gate),
    }, indent=2))

    lines = [
        "# Deep Leakage from Gradients -- with and without DP (Phase 8.3)",
        "",
        "Single MNIST image reconstructed from its gradient by LBFGS",
        "gradient-matching (Zhu et al. 2019), on a smooth-activation CNN.",
        "",
        "| Setting | Reconstruction MSE vs original | Gradient-match loss |",
        "|---|---|---|",
        f"| No DP | {mse_clean:.4f} | {losses_clean[-1]:.3e} |",
        f"| DP-SGD (C=1, sigma=1) | {mse_dp:.4f} | {losses_dp[-1]:.3e} |",
        "",
        f"**Leak demonstrated (no-DP MSE << DP MSE): "
        f"{'PASS' if leak_gate else 'FAIL'}**",
        "",
        "![dlg](../dlg_with_and_without_dp.png)",
        "",
        "## Interpretation",
        "",
        "Without privacy, the gradient of a single example carries enough",
        "information to reconstruct that example -- so 'we only share",
        "gradients, not data' is NOT a privacy guarantee. Clipping + Gaussian",
        "noise (the same DP-SGD primitives in `privacy/dp.py`) destroy the",
        "fine structure the attack relies on, and the reconstruction fails.",
        "This is the empirical case for pairing FL with DP and/or SecAgg.",
        "",
        "Activation note: DLG needs smooth activations (sigmoid) for the",
        "LBFGS gradient-matching to converge; ReLU+MaxPool have ill-defined",
        "second derivatives. The lesson is activation-independent.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"DLG: no-DP MSE={mse_clean:.4f}, DP MSE={mse_dp:.4f}, "
          f"leak_demonstrated={'PASS' if leak_gate else 'FAIL'}")
    print("saved results/dlg/ and results/dlg_with_and_without_dp.png")


if __name__ == "__main__":
    main()
