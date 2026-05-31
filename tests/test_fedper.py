"""Tests for FedPer partial-sharing split (specifications.md section 10)."""

from __future__ import annotations

import torch

from fl.algorithms.fedper import FedPerAggregator, head_param_names, is_shared
from fl.models.cnn import make_mnist_cnn


def test_head_is_final_linear() -> None:
    model = make_mnist_cnn()
    head = head_param_names(model)
    assert head == {"classifier.3.weight", "classifier.3.bias"}


def test_is_shared_excludes_head() -> None:
    model = make_mnist_cnn()
    head = head_param_names(model)
    assert is_shared("features.0.weight", head)
    assert is_shared("classifier.1.weight", head)
    assert not is_shared("classifier.3.weight", head)


def test_aggregator_averages_body_keeps_head() -> None:
    model = make_mnist_cnn()
    head = head_param_names(model)
    agg = FedPerAggregator(head)
    sd1 = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    sd2 = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    out = agg.aggregate([sd1, sd2], [1, 1])
    # Body key: averaged -> 0.5.
    body_key = "classifier.1.weight"
    torch.testing.assert_close(out[body_key], torch.full_like(out[body_key], 0.5))
    # Head key: passed through (first client's value), NOT averaged.
    torch.testing.assert_close(out["classifier.3.weight"], sd1["classifier.3.weight"])
