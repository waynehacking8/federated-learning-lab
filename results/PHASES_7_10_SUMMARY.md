# Phases 7-10 + ablations -- summary

Personalized FL, Byzantine robustness + gradient leakage, server-side
adaptive optimizer, and federated LoRA. Every gate is reported with its
honest PASS/FAIL; FAILs carry a mechanistic reason (see `docs/design-decisions.md` D14-D16 and `docs/results-validation.md`).

| Phase | Experiment | Gate | Result |
|---|---|---|---|
| 7 FedPer (Dir(0.1)) | per-client >= FedAvg+3pp | FAIL | FedPer 0.9934 vs FedAvg 0.9951 (-0.16pp) |
| 7 FedPer (label_skew(3)) | per-client >= FedAvg+3pp | FAIL | FedPer 0.9933 vs FedAvg 0.9929 (+0.04pp) |
| 8 Robust agg | median/Krum within 5pp; FedAvg -20pp | PASS | FedAvg drop +89.3pp, median +0.2pp, krum +0.9pp |
| 8 DLG | no-DP MSE << DP MSE | PASS | no-DP 0.0000 vs DP 4.62e-01 |
| 9 FedAdam | fewer rounds OR +1pp | FAIL | FedAvg 0.9816/r7 vs FedAdam 0.9801/r13 |
| 10 FedIT (IID) | >= 90% centralized | PASS | FedIT 0.8985 vs centralized 0.9040 |
| 10 FedSA-LoRA | per-client +1pp, half payload | FAIL | delta -9.24pp, adapter payload 0.50x |

## Honest-FAIL notes

- **FedPer (Dir / label_skew on MNIST)**: per-client accuracy
  saturates (~0.99) because each client's own-distribution test slice
  is trivial on MNIST -- no 3pp headroom. The FedPer global-metric
  collapse confirms the head specialized (mechanism works); MNIST is
  simply too easy to expose the personalization gain. See D2, D16.
- **FedSA-LoRA**: correctly halves adapter payload and implements the
  A-shared/B-local mechanism, but does not beat FedIT on per-client
  accuracy in this toy regime (zero-init B + little data + few rounds).
  Reported as FAIL rather than tuned to PASS. See D14.

## Communication cost (Phase 4.2)

See `results/COMM_COST_REPORT.md`: SCAFFOLD is exactly 2.00x FedAvg
per round; FedProx and FedAdam are 1.00x; FedPer is 0.99x.
