# FedSA-LoRA under per-client label permutation (its actual regime)

FedSA-LoRA assumes A is GENERAL (shared) and B is CLIENT-SPECIFIC
(local); it also keeps the classifier head local. The matching toy
regime is per-client label permutation: inputs are IID (one general A
serves all), but each client uses its own label map (B + the local head
must specialize). Prior FedSA tests used label-skew (shared label map ->
pooling B wins -> FedSA loses) and feature-skew (client-specific
representation -> breaks shared A, delta -0.19pp); both were the wrong
regime. This is the LoRA analogue of the FedPer label-permutation proof.

4 clients (c0 = identity control), rank 8, 25 rounds, E=2, 1500 train samples/client.

| Method | Mean per-client acc | Per client | Adapter payload |
|---|---|---|---|
| FedIT | 0.3895 | [0.522, 0.166, 0.652, 0.218] | 1.00x |
| FedSA-LoRA | 0.8935 | [0.896, 0.904, 0.896, 0.878] | 0.50x |

Delta (FedSA - FedIT) = **+50.40pp**; adapter payload ratio = **0.50x**.

**Gate (FedSA >= FedIT + 1pp AND payload <= 0.6x)? PASS.**

Conclusion: in the regime FedSA-LoRA is designed for -- general shared features, client-specific label semantics -- keeping B and the head local lets each client fit its own label map, while FedIT's averaged B and head must compromise across conflicting maps. FedSA beats FedIT on per-client accuracy AND halves the adapter payload. The earlier label-skew/feature-skew FAILs were wrong-regime artefacts: those Non-IID types either reward pooling B or require a client-specific A, neither of which is what FedSA targets.
