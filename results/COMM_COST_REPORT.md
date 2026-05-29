# Communication cost per round (Phase 4.2)

Model: small MNIST CNN, 46,730 float parameters (186,920 bytes = 182.5 KiB as float32 per full model).

Bytes a single client uploads per round:

| Algorithm | Upload/round (bytes) | Relative to FedAvg | What is sent |
|---|---|---|---|
| FedAvg | 186,920 | 1.00x | w |
| FedProx | 186,920 | 1.00x | w (proximal term is client-local, not communicated) |
| SCAFFOLD | 373,840 | 2.00x | w + control variate c_k (same shape) -> 2x |
| FedAdam | 186,920 | 1.00x | w (server-side m,v held on server, no client cost) |
| FedPer | 184,320 | 0.99x | shared body only; per-client head stays local |

## Takeaways

- **SCAFFOLD costs exactly 2x FedAvg per round** -- it ships the
  control variate `c_k` (same shape as the model) alongside the
  weights. This is the price of its drift correction; it pays off
  only when the reduction in rounds-to-target more than offsets the
  doubled per-round payload (and, per the K=10 vs K=100 finding, only
  when there are enough clients for the variates to be useful).
- **FedProx is 1x** -- the proximal term is computed locally from the
  broadcast weights; nothing extra is uploaded.
- **FedAdam is 1x** on the client -- the optimizer state (m, v) lives
  on the server.
- **FedPer is 0.99x** -- it never uploads the
  classifier head, so it is strictly cheaper than FedAvg as well as
  more private (the head, which is closest to the labels, never leaves
  the client).
