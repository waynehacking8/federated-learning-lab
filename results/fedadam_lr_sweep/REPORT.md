# FedAdam server-LR sweep (honest re-test of Phase 9)

Dir(alpha=0.1), K=10, E=5, 15 rounds, target=0.95.
Reddi 2020 requires tuning the server LR; this sweep does so instead
of judging FedAdam from a single untuned point.

| Variant | Final acc | Best acc | Rounds to target |
|---|---|---|---|
| FedAvg | 0.9694 | 0.9729 | 7 |
| FedAdam (server_lr=0.01) | 0.9356 | 0.9521 | 12 |
| FedAdam (server_lr=0.05) | 0.9499 | 0.9499 | None |
| FedAdam (server_lr=0.1) | 0.9321 | 0.9346 | None |
| FedAdam (server_lr=0.3) | 0.6133 | 0.6133 | None |
| FedAdam (server_lr=1.0) | 0.1028 | 0.2560 | None |

**Best FedAdam: final=0.9499, fastest r2t=12.**
**Beats FedAvg (>=20% fewer rounds OR +1pp)? NO** (faster=False, higher=False).

Conclusion: even after a full server-LR sweep, FedAdam does not beat FedAvg on Dir(0.1)/MNIST. FedAvg already reaches target in a handful of rounds on this well-conditioned task, leaving no room for adaptive server steps. This is now an evidence-backed statement (5 server LRs tried), not an excuse.
