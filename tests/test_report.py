"""report.py tests: matplotlib is optional, so everything here is skipped if
it isn't installed. Uses the non-interactive Agg backend so tests never try
to open a display window.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from modelgeometry import GeometryTracker, compare_checkpoints, effective_rank  # noqa: E402
from modelgeometry.report import plot_checkpoint_comparison, plot_tracker_history  # noqa: E402


def _qkv0_effective_rank(model, adapter):
    return effective_rank(adapter.qkv_weights(0).q)


def test_plot_tracker_history_returns_axes_with_expected_labels(toy_fused_model):
    tracker = GeometryTracker(toy_fused_model, metrics=[("q_rank", _qkv0_effective_rank)])
    tracker.log_step(0)
    tracker.log_step(1)

    ax = plot_tracker_history(tracker.history, "q_rank")
    assert ax.get_xlabel() == "step"
    assert ax.get_ylabel() == "q_rank"
    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0, 1]


def test_plot_tracker_history_uses_epoch_key_when_no_step(toy_fused_model):
    tracker = GeometryTracker(toy_fused_model, metrics=[("q_rank", _qkv0_effective_rank)])
    tracker.log_epoch(0)

    ax = plot_tracker_history(tracker.history, "q_rank")
    assert ax.get_xlabel() == "epoch"


def test_plot_checkpoint_comparison_skips_non_numeric_entries(toy_fused_model, toy_split_model):
    report = compare_checkpoints(
        toy_fused_model,
        toy_fused_model,
        metrics=[
            ("q_rank", _qkv0_effective_rank),
            ("non_numeric", lambda m, a: {"x": 1}),
        ],
    )
    ax = plot_checkpoint_comparison(report)
    assert [label.get_text() for label in ax.get_xticklabels()] == ["q_rank"]