# FedSA-LoRA failure root-cause isolation

Sweeping the factors the 'under-training / zero-init B' hypothesis
implicates (rounds, local epochs, LoRA rank) to see whether FedSA's
per-client gap to FedIT closes when given more budget.

| Config | rounds | E | rank | FedSA pc | FedIT pc | delta |
|---|---|---|---|---|---|---|
| baseline | 20 | 1 | 8 | 0.5652 | 0.6324 | -6.72pp |
| 2x_rounds | 40 | 1 | 8 | 0.5659 | 0.6864 | -12.05pp |
| 3x_epochs | 20 | 3 | 8 | 0.5657 | 0.6530 | -8.73pp |
| 2x_rank | 20 | 1 | 16 | 0.5662 | 0.6768 | -11.05pp |
| all_up | 40 | 3 | 16 | 0.5644 | 0.6745 | -11.00pp |

Baseline delta = -6.72pp; best delta over the grid = -6.72pp.

**Conclusion: the gap does NOT close with more budget/rank** -- so the earlier 'under-training' explanation is NOT supported. FedSA-LoRA simply does not help at this model scale (DistilBERT/AG-News); the honest statement is 'no benefit here', and a definitive test would need the larger LLMs the FedSA-LoRA paper uses.
