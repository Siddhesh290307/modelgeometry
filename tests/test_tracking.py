"""tracking.py tests: GeometryTracker and compare_checkpoints are pure
orchestration over caller-supplied metric functions, so tests focus on
record/report structure and correct dispatch, run against a toy fixture and a
tiny HF GPT-2 (two structurally different adapters).
"""

from __future__ import annotations

import copy

import pytest
import torch

from modelgeometry.adapters import resolve_adapter
from modelgeometry.linalg import effective_rank, frobenius_norm
from modelgeometry.tracking import GeometryTracker, compare_checkpoints


def _q_effective_rank(model, adapter):
    return effective_rank(adapter.qkv_weights(0).q)


def _qkv_norm_summary(model, adapter):
    # Deliberately non-numeric-diffable (a dict), to exercise the "diff omitted" path.
    qkv = adapter.qkv_weights(0)
    return {"q_frobenius": frobenius_norm(qkv.q)}


# ---- GeometryTracker ----------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_geometry_tracker_log_step_records_metrics(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    tracker = GeometryTracker(model, metrics=[("q_effective_rank", _q_effective_rank)])

    record = tracker.log_step(0)
    assert record["step"] == 0
    assert "q_effective_rank" in record
    assert record["q_effective_rank"] > 0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    tracker.log_step(1)

    assert len(tracker.history) == 2
    assert [r["step"] for r in tracker.history] == [0, 1]


def test_geometry_tracker_log_epoch_uses_epoch_key(toy_fused_model):
    tracker = GeometryTracker(toy_fused_model, metrics=[("q_effective_rank", _q_effective_rank)])
    record = tracker.log_epoch(3)
    assert record["epoch"] == 3
    assert "step" not in record


def test_geometry_tracker_multiple_metrics_and_auto_resolved_adapter(toy_fused_model):
    tracker = GeometryTracker(
        toy_fused_model,
        metrics=[("q_effective_rank", _q_effective_rank), ("qkv_norm_summary", _qkv_norm_summary)],
    )
    record = tracker.log_step(0)
    assert set(record.keys()) == {"step", "q_effective_rank", "qkv_norm_summary"}
    assert isinstance(record["qkv_norm_summary"], dict)


# ---- compare_checkpoints --------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_compare_checkpoints_numeric_metric_includes_diff(model_fixture, request):
    model_a = request.getfixturevalue(model_fixture)
    model_b = copy.deepcopy(model_a)
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(0.1 * torch.randn_like(p))

    report = compare_checkpoints(model_a, model_b, metrics=[("q_effective_rank", _q_effective_rank)])

    entry = report["q_effective_rank"]
    assert set(entry.keys()) == {"a", "b", "diff"}
    assert entry["diff"] == pytest.approx(entry["b"] - entry["a"])


def test_compare_checkpoints_identical_models_have_zero_diff(toy_fused_model):
    model_b = copy.deepcopy(toy_fused_model)
    report = compare_checkpoints(toy_fused_model, model_b, metrics=[("q_effective_rank", _q_effective_rank)])
    assert report["q_effective_rank"]["diff"] == pytest.approx(0.0, abs=1e-5)


def test_compare_checkpoints_non_numeric_metric_omits_diff(toy_fused_model):
    model_b = copy.deepcopy(toy_fused_model)
    report = compare_checkpoints(toy_fused_model, model_b, metrics=[("qkv_norm_summary", _qkv_norm_summary)])
    entry = report["qkv_norm_summary"]
    assert set(entry.keys()) == {"a", "b"}
    assert "diff" not in entry


def test_compare_checkpoints_multiple_metrics(toy_fused_model):
    model_b = copy.deepcopy(toy_fused_model)
    report = compare_checkpoints(
        toy_fused_model,
        model_b,
        metrics=[("q_effective_rank", _q_effective_rank), ("qkv_norm_summary", _qkv_norm_summary)],
    )
    assert set(report.keys()) == {"q_effective_rank", "qkv_norm_summary"}