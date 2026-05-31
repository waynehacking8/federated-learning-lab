# Deep Leakage from Gradients -- with and without DP (Phase 8.3)

Single MNIST image reconstructed from its gradient by LBFGS
gradient-matching (Zhu et al. 2019), on a smooth-activation CNN.
MSE is measured in valid pixel space [0,1] (both images clamped
before scoring), so it is bounded in [0,1] and matches the rendered
figure.

| Setting | Reconstruction MSE (clamped [0,1]) | Gradient-match loss |
|---|---|---|
| No DP | 0.0000 | 3.374e-07 |
| DP-SGD (C=1, sigma=1) | 0.4623 | 9.937e+03 |

**Leak demonstrated (no-DP MSE << DP MSE): PASS**

![dlg](../dlg_with_and_without_dp.png)

## Interpretation

Without privacy, the gradient of a single example carries enough
information to reconstruct that example -- so 'we only share
gradients, not data' is NOT a privacy guarantee. Clipping + Gaussian
noise (the same DP-SGD primitives in `privacy/dp.py`) destroy the
fine structure the attack relies on, and the reconstruction fails.
This is the empirical case for pairing FL with DP and/or SecAgg.

Note: on the DP-noised gradient LBFGS diverges, so the *unclamped*
pixels blow up (raw MSE ~ 3.8e+11); that large number is a
numerical artefact of the unconstrained optimiser, not the quality
measure. The clamped-[0,1] MSE above is the meaningful metric.

Activation note: DLG needs smooth activations (sigmoid) for the
LBFGS gradient-matching to converge; ReLU+MaxPool have ill-defined
second derivatives. The lesson is activation-independent.
