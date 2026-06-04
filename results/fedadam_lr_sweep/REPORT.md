# FedAdam server-LR sweep (honest re-test of Phase 9)

Dir(alpha=0.1), K=10, E=5, 15 rounds, target=0.95.
Reddi 2020 requires tuning the server LR; this sweep does so instead
of judging FedAdam from a single untuned point.

| Variant | Final acc | Best acc | Rounds to target |
|---|---|---|---|
| FedAvg | 0.9724 | 0.9724 | 7 |
| FedAdam (server_lr=0.01) | 0.9549 | 0.9549 | 14 |
| FedAdam (server_lr=0.05) | 0.9611 | 0.9611 | 13 |
| FedAdam (server_lr=0.1) | 0.9566 | 0.9566 | 14 |
| FedAdam (server_lr=0.3) | 0.0892 | 0.2129 | None |
| FedAdam (server_lr=1.0) | 0.0974 | 0.1925 | None |

**Best FedAdam: final=0.9611, fastest r2t=13.**
**Beats FedAvg (>=20% fewer rounds OR +1pp)? NO** (faster=False, higher=False).

Conclusion: even after a full server-LR sweep, FedAdam does not beat FedAvg on Dir(0.1)/MNIST. FedAvg already reaches target in a handful of rounds on this well-conditioned task, leaving no room for adaptive server steps. This is now an evidence-backed statement (5 server LRs tried), not an excuse.
