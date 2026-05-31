# Self-audit of reporting conventions (honest)

Not an external review -- a self-check of one convention that could bias
the headline numbers, with the real measured impact.

## best_acc vs final_acc (max-over-rounds selection bias)

Reports list both `final_acc` (last round) and `best_acc` (max over
rounds of TEST accuracy). `best_acc` is an optimistic selection. Measured
gap (best - final) across all runs:

| Run | final | best | gap |
|---|---|---|---|
| dp_fedavg_iid_C1_s5 | 0.7774 | 0.7998 | **2.2pp** |
| scaffold_dirichlet_a0.1 | 0.9669 | 0.9741 | 0.7pp |
| scaffold_labelskew_2 (truncated) | 0.6856 | 0.6916 | 0.6pp |
| all other runs | -- | -- | <=0.2pp (mostly 0.0) |

**Disposition:** `final_acc` is the clean, unbiased headline and is what
every conclusion rests on. The largest inflation is the DP sigma=5 run
(2.2pp) -- which only ever appears as "monotonic degradation under noise",
a conclusion unaffected by 2pp. SCAFFOLD is already re-reported as tail
mean+/-std, not best. Acceptance gates pass on final_acc too (FedAvg IID
final 0.9863 >= 0.97). A train/val/test split is the textbook fix
(future work); at these gap sizes it changes no conclusion.

## DP per-sample clipping (verified correct)

The (eps, delta) guarantee needs genuine per-sample clipping. Verified:
the library clip equals a manual per-row clip to numerical zero and every
clipped row has L2 norm <= C. Pinned by tests/test_dp_persample.py.

## Dirichlet(0.1) K=10 empty clients (verified non-issue)

Checked 5 seeds: minimum client shard 387-1204 samples, 0 empty clients.
The partition+client code also skips empty shards and records the actual
client count in metrics.json.
