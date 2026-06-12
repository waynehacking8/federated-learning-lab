# FedAdam / FedYogi in a hard regime (label_skew(2), K=10)

Regime: label_skew(2), K=10, E=2, 25 rounds. Here FedAvg plateaus well below the IID ceiling, so
adaptive server steps have room to help (Reddi 2020).

| Variant | Final acc | Best acc | Rounds to FedAvg-final |
|---|---|---|---|
| FedAvg | 0.8866 | 0.8866 | 25 (self) |
| FedAdam (lr=0.01) | 0.7935 | 0.7975 | None |
| FedAdam (lr=0.03) | 0.8439 | 0.8439 | None |
| FedAdam (lr=0.05) | 0.8496 | 0.8496 | None |
| FedAdam (lr=0.1) | 0.6308 | 0.6308 | None |
| FedYogi (lr=0.01) | 0.7984 | 0.8015 | None |
| FedYogi (lr=0.03) | 0.8572 | 0.8572 | None |
| FedYogi (lr=0.05) | 0.8656 | 0.8656 | None |

**Best adaptive variant: yogi_0.05, final=0.8656 (FedAvg 0.8866).**
**Beats FedAvg (+>=1pp final OR <=0.8x rounds)? NO** (higher=False, faster=False).

Conclusion: even in this harder regime and after a full optimizer/LR sweep, the adaptive variants do not beat FedAvg by the gate margin. Reported as measured.
