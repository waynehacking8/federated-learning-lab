# Byzantine-robust aggregation (Phase 8.1/8.2)

IID MNIST, n=10 clients, f=2 Byzantine (sign-flip),
30 rounds, seed 0.

No-attack FedAvg baseline: **0.9915**.

| Aggregator | Final acc (under attack) | Drop vs baseline |
|---|---|---|
| fedavg | 0.0980 | +0.8935 |
| median | 0.9890 | +0.0025 |
| krum | 0.9826 | +0.0089 |
| multikrum | 0.9908 | +0.0007 |
| trimmed | 0.9883 | +0.0032 |
| bulyan | 0.9899 | +0.0016 |

**Acceptance gate: PASS**
- Median within 5pp of baseline: PASS (+0.0025)
- Krum within 5pp of baseline: PASS (+0.0089)
- FedAvg degrades >= 20pp: PASS (+0.8935)

![robust](../robust_aggregation.png)

## Liu et al. (ICML 2023) caveat

These results are on IID data, where the distance-based guarantee of
Krum/Bulyan holds: honest updates cluster, the attacker is an outlier.
Under strong Non-IID data (e.g. Dir(0.1)), honest-client divergence
becomes comparable to attacker divergence, so distance-based
aggregators silently lose their guarantee and can discard honest
minorities. Robust aggregation and heterogeneity-robustness are
distinct problems; classical aggregators solve only the former.
