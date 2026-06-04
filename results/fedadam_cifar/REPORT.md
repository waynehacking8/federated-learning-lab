# FedAdam / FedYogi vs FedAvg on CIFAR-10 (Reddi 2020's regime)

CIFAR-10, Dirichlet(alpha=0.1), K=10, E=1, 40 rounds. A genuinely hard vision task where FedAvg plateaus
below the centralized ceiling and adaptive server steps can help.

| Variant | Final acc | Best acc | Rounds to 0.50 |
|---|---|---|---|
| FedAvg | 0.6824 | 0.6866 | 8 |
| FedAdam (lr=0.003) | 0.5666 | 0.5666 | 31 |
| FedAdam (lr=0.01) | 0.6174 | 0.6174 | 17 |
| FedAdam (lr=0.03) | 0.5756 | 0.5758 | 11 |
| FedYogi (lr=0.003) | 0.5601 | 0.5601 | 31 |
| FedYogi (lr=0.01) | 0.6213 | 0.6223 | 17 |
| FedYogi (lr=0.03) | 0.5820 | 0.5848 | 10 |

**Best adaptive: yogi_0.01, final=0.6213 (FedAvg 0.6824).**
**Beats FedAvg (+>=1pp OR <=0.8x rounds to 0.5)? FAIL** (higher=False, faster=False).

Conclusion: even on CIFAR-10 Dir non-IID, the adaptive variants did not beat FedAvg by the gate margin at this scale/round budget. Reported as measured.
