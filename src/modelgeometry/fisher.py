"""Empirical diagonal Fisher information and per-layer summaries.

Kirkpatrick et al., 2017 ("Overcoming catastrophic forgetting in neural
networks") uses the diagonal of the empirical Fisher information matrix as a
per-parameter importance weight (the basis of Elastic Weight Consolidation).

Implementation note (correctness-critical): the empirical diagonal Fisher is
the average, over samples, of the *squared per-sample* gradient of the loss::

    diag(F) ~= (1/N) * sum_i (grad_theta L(theta; x_i))^2

Averaging gradients across a batch *before* squaring underestimates the
Fisher for any model whose per-sample gradients aren't identical (i.e. any
model past initialization, since squaring is convex and per-sample gradients
partially cancel when averaged first). `diagonal_fisher` always computes true
per-sample gradients via ``torch.func.vmap(torch.func.grad(...))`` rather
than a batched-then-squared approximation.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from modelgeometry.linalg import effective_rank_from_spectrum

LossFn = Callable[..., Tensor]


def diagonal_fisher(
    model: nn.Module,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    n_samples: int,
    loss_fn: LossFn,
) -> Dict[str, Tensor]:
    """Empirical diagonal Fisher information, accumulated over per-sample gradients.

    Args:
        model: any ``nn.Module``. Call ``model.eval()`` before passing it in
            if it contains dropout or other training-mode-only stochastic
            layers — per-sample gradients under ``vmap`` require a
            deterministic forward pass.
        dataloader: yields ``(inputs, targets)`` batches, where ``inputs`` is
            a single tensor consumed by ``model(inputs)`` (e.g. token ids,
            images). For models needing multiple forward arguments, wrap the
            model so its forward takes one tensor.
        n_samples: total number of individual samples to accumulate over,
            drawn from as many dataloader batches as needed; the caller
            decides how many samples are enough for their model and data.
        loss_fn: ``loss_fn(predictions, target) -> scalar loss`` for a single
            sample, where ``predictions`` is ``model``'s raw forward output.
            This is the task the Fisher is computed with respect to — the
            library does not prescribe a default task, loss, or dataset.

    Returns:
        Dict mapping parameter name to its diagonal Fisher tensor (same shape
        as the parameter).
    """
    params = {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name: b.detach() for name, b in model.named_buffers()}

    def compute_loss(params_: Dict[str, Tensor], single_input: Tensor, single_target: Tensor) -> Tensor:
        predictions = functional_call(model, (params_, buffers), (single_input.unsqueeze(0),))
        return loss_fn(predictions, single_target.unsqueeze(0))

    per_sample_grad_fn = vmap(grad(compute_loss), in_dims=(None, 0, 0))

    fisher_sum = {name: torch.zeros_like(p) for name, p in params.items()}
    seen = 0
    for inputs, targets in dataloader:
        if seen >= n_samples:
            break
        take = min(inputs.shape[0], n_samples - seen)
        per_sample_grads = per_sample_grad_fn(params, inputs[:take], targets[:take])
        for name, sample_grads in per_sample_grads.items():
            fisher_sum[name] += (sample_grads**2).sum(dim=0)
        seen += take

    if seen == 0:
        raise ValueError("dataloader produced no samples before reaching n_samples.")

    return {name: value / seen for name, value in fisher_sum.items()}


def fisher_layer_summary(fisher_dict: Dict[str, Tensor], top_k: int = 5) -> Dict[str, object]:
    """Per-layer summary of a diagonal Fisher dict: mass, top-k mass fraction, effective rank.

    Args:
        fisher_dict: output of `diagonal_fisher` (parameter name -> Fisher tensor).
        top_k: number of highest-mass parameters included in the top-k mass
            fraction.

    Returns:
        A dict with ``per_parameter_mass`` (name -> total Fisher mass for
        that parameter), ``total_mass``, ``top_k_mass_fraction`` (fraction of
        total mass held by the ``top_k`` highest-mass parameters), and
        ``effective_rank`` (effective rank of the per-parameter mass
        distribution, via `linalg.effective_rank_from_spectrum`).
    """
    per_parameter_mass = {name: float(tensor.sum()) for name, tensor in fisher_dict.items()}
    masses = np.array(sorted(per_parameter_mass.values(), reverse=True))
    total_mass = float(masses.sum())
    top_k_mass = float(masses[:top_k].sum())
    return {
        "per_parameter_mass": per_parameter_mass,
        "total_mass": total_mass,
        "top_k_mass_fraction": (top_k_mass / total_mass) if total_mass > 0 else 0.0,
        "effective_rank": effective_rank_from_spectrum(masses) if total_mass > 0 else 0.0,
    }