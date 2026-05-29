"""Tests for FedLoRA aggregation key selection (specifications.md section 14)."""

from __future__ import annotations

import torch

from fl.algorithms.fedlora import (
    FedITAggregator, FedSALoRAAggregator, is_classifier, is_lora_A, is_lora_B,
)

KEYS = [
    "base.layer.0.attention.q_lin.lora_A.default.weight",
    "base.layer.0.attention.q_lin.lora_B.default.weight",
    "base.layer.0.attention.v_lin.lora_A.default.weight",
    "base.layer.0.attention.v_lin.lora_B.default.weight",
    "classifier.weight",
    "pre_classifier.bias",
]


def test_key_classifiers() -> None:
    assert is_lora_A(KEYS[0]) and not is_lora_B(KEYS[0])
    assert is_lora_B(KEYS[1]) and not is_lora_A(KEYS[1])
    assert is_classifier(KEYS[4]) and is_classifier(KEYS[5])


def test_fedit_shares_everything() -> None:
    agg = FedITAggregator()
    assert set(agg.shared_keys(KEYS)) == set(KEYS)


def test_fedsa_shares_only_A() -> None:
    agg = FedSALoRAAggregator()
    shared = set(agg.shared_keys(KEYS))
    assert KEYS[0] in shared and KEYS[2] in shared  # A matrices shared
    assert KEYS[1] not in shared and KEYS[3] not in shared  # B stays local
    # Head is client-specific under label skew -> kept local too.
    assert KEYS[4] not in shared and KEYS[5] not in shared


def test_fedsa_aggregate_excludes_B() -> None:
    agg = FedSALoRAAggregator()
    s1 = {k: torch.zeros(2, 2) for k in KEYS}
    s2 = {k: torch.ones(2, 2) for k in KEYS}
    out = agg.aggregate([s1, s2], [1, 1])
    assert KEYS[1] not in out and KEYS[3] not in out  # B not in output
    torch.testing.assert_close(out[KEYS[0]], torch.full((2, 2), 0.5))  # A averaged


def test_fedsa_payload_smaller_than_fedit() -> None:
    """FedSA shares ~half the adapter params (only A, not B)."""
    fedit = FedITAggregator().shared_keys(KEYS)
    fedsa = FedSALoRAAggregator().shared_keys(KEYS)
    assert len(fedsa) < len(fedit)
