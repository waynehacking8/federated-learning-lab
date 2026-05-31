# FedPer mechanism verification -- label-permutation test

5 clients, IID images, each client applies a fixed random label
permutation (client 0 = identity control). A single global head CANNOT
serve conflicting label maps, so a correctly-implemented FedPer
(per-client head on a shared body) must beat FedAvg by a wide margin.
Evaluation uses the **actually-deployed** model (FedAvg: shared global
model; FedPer: aggregated body + that client's own head), NOT each
client's locally-overfit copy.

| Method | Mean per-client acc | Per client |
|---|---|---|
| FedAvg | 0.3001 | [0.360, 0.117, 0.300, 0.348, 0.377] |
| FedPer | 0.9701 | [0.965, 0.962, 0.973, 0.979, 0.973] |

Delta (FedPer - FedAvg) = **+67.0pp**. **Mechanism proven? YES.**

FedAvg is near-random per-client (one shared head cannot satisfy five
conflicting label maps); FedPer's per-client head fits each client's
permutation. Positive evidence the FedPer implementation is correct: the
earlier Phase 7 "FAIL" on Dir(0.1)/label_skew was a metric-saturation
artefact on near-trivial MNIST slices, not a broken mechanism.

## Harness-bug note (caught during this verification)

The first version of this test scored each client's `_local_model`
(just trained on its own permutation), making FedAvg spuriously hit
~0.97 too -- the same eval-leakage class fixed in FedLoRA. The corrected
eval loads the deployed aggregated model; only then does FedAvg correctly
collapse to ~0.30.
