# FedSA-LoRA under feature skew (the regime its mechanism targets)

Prior FedSA tests used LABEL skew, where all clients share one
feature->label mapping, so pooling B (FedIT) is strictly better and
FedSA loses regardless of budget. FedSA-LoRA's premise is that B
captures *client-specific* knowledge -- which only matters under
FEATURE skew. Here two synthetic dialects (normal vs token-reversed
text, IID labels) force genuinely different representations; the head
stays shared so the comparison isolates B specialization.

4 clients, dialects reversed?=[False, True, False, True], rank 8, 25 rounds, E=2.

| Method | Mean per-client acc | Per client | Adapter payload |
|---|---|---|---|
| FedIT | 0.8863 | [0.865, 0.902, 0.887, 0.89] | 1.00x |
| FedSA-LoRA | 0.8844 | [0.88, 0.902, 0.885, 0.87] | 0.50x |

Delta (FedSA - FedIT) = **-0.19pp**; adapter payload ratio = **0.50x**.

**Gate (FedSA >= FedIT + 1pp AND payload <= 0.6x)? FAIL.**

Conclusion: even under feature skew, FedSA did not clear the +1pp gate at this scale. Reported as measured.
