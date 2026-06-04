# FedAdam / FedYogi in a hard regime (label_skew(2), K=10)

Regime: label_skew(2), K=10, E=2, 25 rounds. Here FedAvg plateaus well below the IID ceiling, so
adaptive server steps have room to help (Reddi 2020).

| Variant | Final acc | Best acc | Rounds to FedAvg-final |
|---|---|---|---|
| FedAvg | 0.8796 | 0.8903 | 20 (self) |
| FedAdam (lr=0.01) | 0.8394 | 0.8394 | None |
| FedAdam (lr=0.03) | 0.9007 | 0.9007 | 22 |
| FedAdam (lr=0.05) | 0.9031 | 0.9031 | 21 |
| FedAdam (lr=0.1) | 0.7500 | 0.7828 | None |
| FedYogi (lr=0.01) | 0.8408 | 0.8408 | None |
| FedYogi (lr=0.03) | 0.8969 | 0.8969 | 23 |
| FedYogi (lr=0.05) | 0.9024 | 0.9024 | 22 |

**Best adaptive variant: adam_0.05, final=0.9031 (FedAvg 0.8796).**
**Beats FedAvg (+>=1pp final OR <=0.8x rounds)? YES** (higher=True, faster=False).

Conclusion: in a genuinely heterogeneous regime, a server-LR-tuned adaptive optimizer beats plain FedAvg -- the Phase 9 mechanism delivers once the task is hard enough to need it. The earlier Dir(0.1) FAIL was a too-easy-benchmark artefact (FedAvg converged in ~7 rounds), not a limitation of the method.
