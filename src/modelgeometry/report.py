"""Optional plotting helpers for `GeometryTracker` histories and
`compare_checkpoints` reports.

Deliberately kept separate from the rest of the package (not imported by
`modelgeometry/__init__.py`) so the core library stays usable headlessly —
importing `modelgeometry` never pulls in matplotlib. Install the `report`
extra (``pip install modelgeometry[report]``) to use this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_tracker_history(history: List[Dict[str, Any]], metric_name: str, ax: Optional[Axes] = None) -> Axes:
    """Line plot of one `GeometryTracker.history` metric over step/epoch.

    Args:
        history: `GeometryTracker.history` (or any list of per-call records
            containing a numeric ``"step"`` or ``"epoch"`` key plus
            ``metric_name``).
        metric_name: which tracked metric to plot; its values must be
            numeric (scalar) at every record.
        ax: existing axes to draw on; a new figure/axes is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots()

    x_key = "step" if "step" in history[0] else "epoch"
    x = [record[x_key] for record in history]
    y = [record[metric_name] for record in history]

    ax.plot(x, y, marker="o")
    ax.set_xlabel(x_key)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} over {x_key}")
    return ax


def plot_checkpoint_comparison(report: Dict[str, Dict[str, Any]], ax: Optional[Axes] = None) -> Axes:
    """Grouped bar chart of a `compare_checkpoints` report's numeric metrics.

    Non-numeric entries (e.g. dict-valued metrics, which `compare_checkpoints`
    doesn't compute a `"diff"` for) are skipped rather than raising.

    Args:
        report: output of `tracking.compare_checkpoints`.
        ax: existing axes to draw on; a new figure/axes is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots()

    names, values_a, values_b = [], [], []
    for name, entry in report.items():
        if not (isinstance(entry["a"], (int, float)) and isinstance(entry["b"], (int, float))):
            continue
        names.append(name)
        values_a.append(entry["a"])
        values_b.append(entry["b"])

    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], values_a, width=width, label="a")
    ax.bar([i + width / 2 for i in x], values_b, width=width, label="b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_title("Checkpoint comparison")
    return ax