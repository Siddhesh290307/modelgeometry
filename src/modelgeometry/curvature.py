"""Curvature-prediction validation: compares the actual loss change under a
parameter perturbation to the change a Fisher (or K-FAC block) approximation
predicts for it. General-purpose sanity check for how well a computed
curvature approximation actually describes the loss surface around the
current parameters — no dataset or architecture assumptions baked in.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

LossFn = Callable[..., Tensor]


def curvature_prediction_check(
    model: nn.Module,
    loss_fn: LossFn,
    batch: Tuple[Tensor, Tensor],
    perturbations: Dict[str, Tensor],
    fisher: Dict[str, Tensor],
    block_factors: Optional[Dict[str, Dict[str, Tensor]]] = None,
) -> Dict[str, float]:
    """Compare actual loss change under a perturbation to Fisher-predicted change.

    The standard second-order approximation around a local optimum (where the
    gradient is ~0) is::

        actual_loss_change ~= predicted_loss_change = 0.5 * perturbation^T F perturbation

    Args:
        model: the model to perturb (perturbed in place, then restored).
        loss_fn: ``loss_fn(predictions, target) -> scalar loss``.
        batch: ``(inputs, targets)`` used to evaluate the loss before/after
            perturbation.
        perturbations: dict mapping parameter name -> perturbation tensor
            (same shape as the parameter) to apply.
        fisher: diagonal Fisher dict (e.g. from `fisher.diagonal_fisher`),
            parameter name -> Fisher tensor (same shape as the parameter).
            Used for the diagonal quadratic-form prediction:
            ``0.5 * sum(fisher * perturbation^2)``.
        block_factors: optional K-FAC factors (e.g. from `kfac.kfac_factors`),
            keyed by the same names as ``perturbations``, each
            ``{"A": Tensor, "G": Tensor}``. If provided, also computes the
            Kronecker-factored block quadratic-form prediction
            ``0.5 * tr(G @ dW @ A @ dW^T)`` for every perturbation whose
            tensor is 2-D and shape-compatible with its ``A``/``G`` factors
            (``dW.shape == (G.shape[0], A.shape[0])``); incompatible entries
            are skipped, since the caller may homogenize `A` for bias terms
            that a given perturbation doesn't cover.

    Returns:
        Dict with ``baseline_loss``, ``perturbed_loss``, ``actual_change``,
        ``predicted_change_diagonal``, and (if ``block_factors`` given)
        ``predicted_change_block``.
    """
    inputs, targets = batch
    params = dict(model.named_parameters())

    with torch.no_grad():
        baseline_loss = float(loss_fn(model(inputs), targets))

    predicted_diagonal = 0.0
    for name, delta in perturbations.items():
        if name in fisher:
            predicted_diagonal += 0.5 * float((fisher[name] * delta**2).sum())

    predicted_block = None
    if block_factors is not None:
        predicted_block = 0.0
        for name, delta in perturbations.items():
            factors = block_factors.get(name)
            if factors is None or delta.dim() != 2:
                continue
            a, g = factors["A"], factors["G"]
            if delta.shape != (g.shape[0], a.shape[0]):
                continue
            predicted_block += 0.5 * float(torch.trace(g @ delta @ a @ delta.T))

    originals = {}
    with torch.no_grad():
        for name, delta in perturbations.items():
            params[name].add_(delta)
            originals[name] = delta

        perturbed_loss = float(loss_fn(model(inputs), targets))

        for name, delta in originals.items():
            params[name].sub_(delta)

    result = {
        "baseline_loss": baseline_loss,
        "perturbed_loss": perturbed_loss,
        "actual_change": perturbed_loss - baseline_loss,
        "predicted_change_diagonal": predicted_diagonal,
    }
    if predicted_block is not None:
        result["predicted_change_block"] = predicted_block
    return result