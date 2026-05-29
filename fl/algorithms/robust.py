"""Byzantine-robust aggregators (Phase 8).

Drop-in replacements for FedAvgAggregator that tolerate a fraction `f`
of malicious clients. All operate on lists of state dicts and return a
new state dict, matching the Aggregator protocol in fl/server.py.

Implemented:
    - CoordinateMedianAggregator   : coordinate-wise median (Yin et al. 2018)
    - TrimmedMeanAggregator        : coordinate-wise trimmed mean (Yin 2018)
    - KrumAggregator               : Krum / Multi-Krum (Blanchard et al. 2017)
    - BulyanAggregator             : Bulyan (Krum pool + trimmed mean; Mhamdi 2018)

Integer tensors (e.g. BN counters) are passed through from the first
client, as in FedAvg -- robustness applies to the float parameters.

Liu et al. (ICML 2023) caveat: under strong Non-IID data, honest-client
divergence becomes comparable to attacker divergence, so distance-based
aggregators (Krum/Bulyan) silently lose their guarantee. These are
demonstrated on IID data, where the guarantee holds; the caveat is
documented in docs/specifications.md section 11 and the Phase 8 report.
"""

from __future__ import annotations

import torch


def _float_keys(state: dict) -> list[str]:
    return [k for k, v in state.items() if v.is_floating_point()]


def _flatten(state: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([state[k].reshape(-1).to(torch.float32) for k in keys])


def _unflatten(vec: torch.Tensor, template: dict, keys: list[str]) -> dict:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for k in keys:
        numel = template[k].numel()
        chunk = vec[offset:offset + numel].reshape(template[k].shape)
        out[k] = chunk.to(template[k].dtype)
        offset += numel
    return out


def _passthrough_int_keys(out: dict, template: dict) -> dict:
    for k, v in template.items():
        if k not in out:
            out[k] = v.clone()
    return out


def coordinate_median(client_states: list[dict]) -> dict:
    keys = _float_keys(client_states[0])
    stacked = torch.stack([_flatten(s, keys) for s in client_states])  # (n, D)
    med = stacked.median(dim=0).values
    out = _unflatten(med, client_states[0], keys)
    return _passthrough_int_keys(out, client_states[0])


def trimmed_mean(client_states: list[dict], beta: int) -> dict:
    """Coordinate-wise trimmed mean: drop the `beta` highest and lowest, average rest."""
    keys = _float_keys(client_states[0])
    stacked = torch.stack([_flatten(s, keys) for s in client_states])  # (n, D)
    n = stacked.shape[0]
    if 2 * beta >= n:
        raise ValueError(f"trimmed_mean: 2*beta ({2*beta}) must be < n ({n})")
    sorted_vals, _ = stacked.sort(dim=0)
    trimmed = sorted_vals[beta:n - beta]  # drop beta from each end
    mean = trimmed.mean(dim=0)
    out = _unflatten(mean, client_states[0], keys)
    return _passthrough_int_keys(out, client_states[0])


def _krum_scores(flats: torch.Tensor, f: int) -> torch.Tensor:
    """Sum of squared distances to the n-f-2 closest peers, per candidate."""
    n = flats.shape[0]
    # Pairwise squared distances.
    dists = torch.cdist(flats, flats) ** 2  # (n, n)
    m = n - f - 2  # number of closest neighbours to sum
    if m < 1:
        m = 1
    scores = torch.empty(n)
    for i in range(n):
        d = dists[i]
        closest = torch.topk(d, m + 1, largest=False).values  # +1 to skip self (dist 0)
        scores[i] = closest.sum() - d[i]  # subtract self-distance (0 anyway)
    return scores


def krum(client_states: list[dict], f: int, multi_m: int = 1) -> dict:
    """Krum (multi_m=1) or Multi-Krum (multi_m>1).

    Picks the candidate(s) with smallest sum-of-distances to their
    n-f-2 closest peers; Multi-Krum averages the top `multi_m`.
    """
    keys = _float_keys(client_states[0])
    flats = torch.stack([_flatten(s, keys) for s in client_states])  # (n, D)
    scores = _krum_scores(flats, f)
    n = flats.shape[0]
    m = max(1, min(multi_m, n))
    chosen = torch.topk(scores, m, largest=False).indices
    selected = flats[chosen].mean(dim=0)
    out = _unflatten(selected, client_states[0], keys)
    return _passthrough_int_keys(out, client_states[0])


def bulyan(client_states: list[dict], f: int) -> dict:
    """Bulyan: build a Multi-Krum selection pool of size n-2f, then
    coordinate-wise trimmed mean (drop f each end) over that pool."""
    keys = _float_keys(client_states[0])
    flats = torch.stack([_flatten(s, keys) for s in client_states])
    n = flats.shape[0]
    pool_size = n - 2 * f
    if pool_size < 1:
        raise ValueError(f"bulyan: n-2f ({pool_size}) must be >= 1")

    # Iteratively pick Multi-Krum winners into the pool.
    remaining = list(range(n))
    pool_idx = []
    while len(pool_idx) < pool_size and remaining:
        sub = flats[remaining]
        scores = _krum_scores(sub, f)
        best_local = int(torch.argmin(scores).item())
        pool_idx.append(remaining[best_local])
        remaining.pop(best_local)

    pool = flats[pool_idx]  # (pool_size, D)
    # Coordinate-wise trimmed mean over the pool.
    beta = min(f, (pool.shape[0] - 1) // 2)
    sorted_vals, _ = pool.sort(dim=0)
    if beta > 0:
        sorted_vals = sorted_vals[beta:pool.shape[0] - beta]
    agg = sorted_vals.mean(dim=0)
    out = _unflatten(agg, client_states[0], keys)
    return _passthrough_int_keys(out, client_states[0])


class CoordinateMedianAggregator:
    def aggregate(self, client_states, sample_sizes):
        return coordinate_median(client_states)


class TrimmedMeanAggregator:
    def __init__(self, beta: int):
        self.beta = beta

    def aggregate(self, client_states, sample_sizes):
        return trimmed_mean(client_states, self.beta)


class KrumAggregator:
    def __init__(self, f: int, multi_m: int = 1):
        self.f = f
        self.multi_m = multi_m

    def aggregate(self, client_states, sample_sizes):
        return krum(client_states, self.f, self.multi_m)


class BulyanAggregator:
    def __init__(self, f: int):
        self.f = f

    def aggregate(self, client_states, sample_sizes):
        return bulyan(client_states, self.f)
