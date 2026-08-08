"""Training-time tracking and cross-checkpoint comparison.

Both `GeometryTracker` and `compare_checkpoints` are pure orchestration: they
know nothing about which metrics are "the" metrics to compute, or which axis
of comparison (architecture, training run, seed, ...) is meaningful — the
caller supplies a list of ``(name, metric_fn)`` pairs, where
``metric_fn(model, adapter) -> value`` wraps whatever modelgeometry function
(or combination of them) the caller cares about.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn

from modelgeometry.adapters import ModelAdapter, resolve_adapter

MetricFn = Callable[[nn.Module, ModelAdapter], Any]
NamedMetrics = List[Tuple[str, MetricFn]]


class GeometryTracker:
    """Framework-agnostic training callback that logs a caller-chosen set of metrics.

    Call `log_step` / `log_epoch` directly from any training loop (vanilla
    PyTorch, an HF `Trainer` callback's `on_step_end`, a Lightning hook, ...)
    — this class has no dependency on any specific training framework.

    Example::

        tracker = GeometryTracker(model, metrics=[
            ("qkv0_effective_rank", lambda m, a: effective_rank(a.qkv_weights(0).q)),
        ])
        for step, batch in enumerate(dataloader):
            ...
            tracker.log_step(step)
        tracker.history  # list of per-call records
    """

    def __init__(self, model: nn.Module, metrics: NamedMetrics, adapter: Optional[ModelAdapter] = None):
        self.model = model
        self.metrics = metrics
        self.adapter = adapter or resolve_adapter(model)
        self.history: List[Dict[str, Any]] = []

    def _log(self, marker_key: str, marker_value: int) -> Dict[str, Any]:
        record: Dict[str, Any] = {marker_key: marker_value}
        for name, metric_fn in self.metrics:
            record[name] = metric_fn(self.model, self.adapter)
        self.history.append(record)
        return record

    def log_step(self, step: int) -> Dict[str, Any]:
        """Compute every tracked metric and append a record ``{"step": step, ...}``."""
        return self._log("step", step)

    def log_epoch(self, epoch: int) -> Dict[str, Any]:
        """Compute every tracked metric and append a record ``{"epoch": epoch, ...}``."""
        return self._log("epoch", epoch)


def _try_numeric_diff(value_a: Any, value_b: Any) -> Optional[Any]:
    try:
        return value_b - value_a
    except Exception:
        return None


def compare_checkpoints(
    model_a: nn.Module,
    model_b: nn.Module,
    metrics: NamedMetrics,
    adapter_a: Optional[ModelAdapter] = None,
    adapter_b: Optional[ModelAdapter] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generic per-metric diff report between two checkpoints.

    The library never suggests which axis of comparison (architecture,
    training run, seed, ...) is meaningful — ``model_a``/``model_b`` and the
    metric functions are entirely the caller's choice.

    Args:
        model_a, model_b: the two models to compare.
        metrics: ``[(name, metric_fn), ...]`` where ``metric_fn(model, adapter) -> value``.
        adapter_a, adapter_b: pre-resolved adapters; auto-resolved if omitted.

    Returns:
        ``{name: {"a": value_a, "b": value_b, "diff": value_b - value_a}}``.
        ``"diff"`` is omitted for any metric whose values don't support
        subtraction (e.g. a dict-valued metric) rather than raising.
    """
    adapter_a = adapter_a or resolve_adapter(model_a)
    adapter_b = adapter_b or resolve_adapter(model_b)

    report: Dict[str, Dict[str, Any]] = {}
    for name, metric_fn in metrics:
        value_a = metric_fn(model_a, adapter_a)
        value_b = metric_fn(model_b, adapter_b)
        entry: Dict[str, Any] = {"a": value_a, "b": value_b}
        diff = _try_numeric_diff(value_a, value_b)
        if diff is not None:
            entry["diff"] = diff
        report[name] = entry
    return report