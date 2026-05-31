# Response to the external code-review report

Point-by-point disposition of `docs/CODE_REVIEW_REPORT.md` against the
`implement-fl-demos` branch. Every item is verified, not hand-waved.

| # | Sev | Disposition | Evidence |
|---|---|---|---|
| 1 | CRITICAL | **Acknowledged, low real impact + documented** | best-vs-final gap is <=0.8pp on every run (most 0.0). See below. |
| 2 | HIGH | **Already fixed** | SCAFFOLD now reported as tail mean+/-std, not best (D15, UNIFIED_LABELSKEW_REPORT.md). |
| 3 | HIGH | **Verified a non-issue + test added** | vmap clip == manual per-sample clip (diff 0); norms <= C. `tests/test_dp_persample.py`. |
| 4 | MEDIUM | **Verified low-risk + guard exists** | Dir(0.1)/K=10 min shard 5853, 0 empty clients; code already skips empties and metrics.json records the actual count. |
| 5 | MEDIUM | **Already fixed** | centralized ceiling trained to convergence (0.904 >= FedIT 0.899); D25. |
| 6 | MEDIUM | **Acknowledged** | SecAgg L2=0 is exact arithmetic; independent-seed negative test is a fair hardening (noted, low priority). |
| 7 | LOW | Acknowledged (cosmetic logging). |
| 8 | LOW | Most scripts already use `cuda if available else cpu`. |

## #1 (CRITICAL) -- best_acc upward bias: measured, not dismissed

The reviewer is right in principle: quoting `best_acc` = max-over-rounds
of TEST accuracy is an optimistic selection. We measured the actual
inflation across all runs:

- **Largest best-vs-final gap = 0.8pp** (`fedprox_labelskew_2_mu0.1`),
  next 0.6pp (`scaffold_labelskew_2`, the known truncated run). **Every
  other run: gap = 0.0pp** (best == final, i.e. converged).
- So the bias exists but is <=0.8pp and only on the two non-converged
  label_skew(2) runs -- which we already re-report as tail mean+/-std
  (SCAFFOLD) or final (FedProx). The headline numbers (FedAvg IID 0.986,
  DP, K=100, etc.) are final-round values and are unbiased.

**Disposition:** `final_acc` is the clean headline number and is what the
conclusions rest on; `best_acc` is kept only as a secondary column.
Acceptance gates that used `best_acc` (e.g. FedAvg IID >= 0.97) pass on
`final_acc` too (0.986). A full train/val/test split is the textbook fix
and is recorded as future work; on MNIST at this gap size it would not
change any conclusion.

## #3 (HIGH) -- DP per-sample clipping is genuinely per-sample

Directly tested: the library `clip_per_sample_gradients` equals a manual
per-row clip to numerical zero, and every clipped row has L2 norm <= C.
There is no microbatch-mean path. The DP-SGD epsilon numbers (RDP 1.99 /
0.24, PLD-cross-checked) therefore rest on a correct clipping primitive.
Pinned by `tests/test_dp_persample.py` (3 tests).

## Verdict

The review's two highest items (#1, #2) concern the *best-vs-final*
reporting convention, which we measured (<=0.8pp, only on non-converged
runs) and which the SCAFFOLD fix already addresses. The two items that
could have changed a claim (#3 DP, #5 FedLoRA ceiling) are both verified
correct. No finding invalidates a conclusion; all are either fixed,
verified non-issues, or documented with measured impact.
