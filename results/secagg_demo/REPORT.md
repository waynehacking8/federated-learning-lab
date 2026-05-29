# SecAgg skeleton demo

Three clients, each holding a 4-d update vector. Each client
splits its update into 3 additive shares and routes share `j`
to peer `j`. The server then sums the 3 peer-aggregated shares.

## Inputs

| Client | Update |
|---|---|
| 0 | [1.0, 2.0, 3.0, 4.0] |
| 1 | [0.5, -1.0, 2.5, 0.0] |
| 2 | [-2.0, 1.0, 0.5, 3.0] |

**True sum**: `[-0.5, 2.0, 6.0, 7.0]`

## Per-peer aggregated shares (what a single eavesdropping peer sees)

These should not reveal anything about the individual updates.

| Peer | Aggregated share |
|---|---|
| 0 | [-0.253, -1.971, 1.722, 1.836] |
| 1 | [1.151, -1.504, 1.004, 0.482] |
| 2 | [-1.398, 5.476, 3.273, 4.682] |

## Server-side recovered sum

- Recovered: `[-0.5, 2.0, 6.0, 7.0]`
- True:      `[-0.5, 2.0, 6.0, 7.0]`
- L2 error:  `0.00e+00`

## Acceptance

- Server-side recovered sum equals true sum within float tolerance: **True**
