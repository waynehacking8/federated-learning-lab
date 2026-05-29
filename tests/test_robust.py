"""Tests for Byzantine-robust aggregators (specifications.md section 11)."""

from __future__ import annotations

import torch

from fl.algorithms.robust import bulyan, coordinate_median, krum, trimmed_mean


def _honest_and_byzantine(n_honest=8, n_byz=2, d=40, scale=10.0):
    torch.manual_seed(0)
    wstar = torch.ones(d)
    honest = [{"w": wstar + 0.05 * torch.randn(d)} for _ in range(n_honest)]
    byz = [{"w": -scale * torch.ones(d)} for _ in range(n_byz)]
    return honest + byz, wstar


def _err(state, wstar):
    return (state["w"] - wstar).abs().mean().item()


def test_median_resists_signflip() -> None:
    states, wstar = _honest_and_byzantine()
    out = coordinate_median(states)
    assert _err(out, wstar) < 0.1


def test_trimmed_mean_resists_signflip() -> None:
    states, wstar = _honest_and_byzantine()
    out = trimmed_mean(states, beta=2)
    assert _err(out, wstar) < 0.1


def test_krum_picks_honest() -> None:
    states, wstar = _honest_and_byzantine()
    out = krum(states, f=2, multi_m=1)
    assert _err(out, wstar) < 0.2


def test_multikrum_resists() -> None:
    states, wstar = _honest_and_byzantine()
    out = krum(states, f=2, multi_m=6)
    assert _err(out, wstar) < 0.1


def test_bulyan_resists() -> None:
    states, wstar = _honest_and_byzantine()
    out = bulyan(states, f=2)
    assert _err(out, wstar) < 0.1


def test_trimmed_mean_requires_valid_beta() -> None:
    states, _ = _honest_and_byzantine(n_honest=2, n_byz=1)  # n=3
    try:
        trimmed_mean(states, beta=2)  # 2*2 >= 3
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_integer_keys_passthrough() -> None:
    states = [
        {"w": torch.ones(4), "n": torch.tensor(3)},
        {"w": torch.ones(4) * 2, "n": torch.tensor(5)},
        {"w": torch.ones(4) * 3, "n": torch.tensor(7)},
    ]
    out = coordinate_median(states)
    assert "n" in out and out["n"].dtype == torch.long
