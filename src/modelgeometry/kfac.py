"""K-FAC (Kronecker-factored approximate curvature) factors.

Martens & Grosse, 2015 ("Optimizing Neural Networks with Kronecker-factored
Approximate Curvature"). For a linear layer ``y = Wx``, K-FAC approximates
the Fisher/curvature block for ``W`` as a Kronecker product ``A (x) G`` of
two much smaller matrices:

- ``A``: the (``in_features`` [+1 if homogenized for bias]) covariance of the
  layer's input activations, ``E[a a^T]``.
- ``G``: the (``out_features``, ``out_features``) covariance of the loss
  gradient with respect to the layer's output (pre-activation), ``E[g g^T]``.

Both factors are accumulated from ordinary forward+backward passes. This is
distinct from `fisher.diagonal_fisher`'s requirement for true per-sample
*parameter* gradients (which collapse per-sample structure when accumulated
through a standard batched backward pass): here, the captured *activation*
and *output-gradient* tensors retain their individual per-token values
throughout an ordinary backward pass, so no per-sample looping or
``torch.func`` machinery is needed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from modelgeometry.adapters import ModelAdapter, resolve_adapter
from modelgeometry.hooks import HookRegistry

LossFn = Callable[..., Tensor]


def kfac_factors(
    model: nn.Module,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    n_samples: int,
    loss_fn: LossFn,
    adapter: Optional[ModelAdapter] = None,
    include_bias: bool = True,
) -> Dict[str, Dict[str, Tensor]]:
    """K-FAC activation-covariance (A) and gradient-covariance (G) factors per attention projection.

    Args:
        model: any ``nn.Module`` resolvable by `resolve_adapter` (or pass one
            explicitly via ``adapter``).
        dataloader: yields ``(inputs, targets)`` batches, as in `fisher.diagonal_fisher`.
        n_samples: total number of individual (batch) samples to accumulate
            activation/gradient statistics over.
        loss_fn: ``loss_fn(predictions, target) -> scalar loss``, the task
            the curvature is computed with respect to.
        adapter: a pre-resolved `ModelAdapter`; auto-resolved from ``model``
            if omitted.
        include_bias: if True (default) and the target module has a bias,
            homogenize activations with a constant 1 before computing ``A``,
            so the bias's contribution to curvature is included (the
            standard K-FAC convention).

    Returns:
        Dict keyed by the target module's real weight parameter name (e.g.
        ``"transformer.h.0.attn.c_attn.weight"``) — the same naming
        `model.named_parameters()` and `fisher.diagonal_fisher` use, so a
        `curvature.curvature_prediction_check` block-factors prediction can
        be matched against a diagonal Fisher and a set of perturbations
        without any remapping. Each entry maps to ``{"A": Tensor, "G": Tensor}``.
    """
    adapter = adapter or resolve_adapter(model)
    dotted_names = {id(module): name for name, module in model.named_modules()}

    modules: Dict[str, nn.Module] = {}
    for layer_idx in range(adapter.num_layers()):
        for role, module in adapter.qkv_modules(layer_idx).items():
            dotted = dotted_names.get(id(module), f"layer{layer_idx}.{role}")
            modules[f"{dotted}.weight"] = module

    registry = HookRegistry()
    a_sum: Dict[str, Tensor] = {}
    g_sum: Dict[str, Tensor] = {}
    n_rows: Dict[str, int] = {name: 0 for name in modules}

    with registry:
        for name, module in modules.items():
            registry.capture_input(f"{name}:in", module)
            registry.capture_grad_output(f"{name}:grad", module)

        seen = 0
        for inputs, targets in dataloader:
            if seen >= n_samples:
                break
            take = min(inputs.shape[0], n_samples - seen)
            batch_inputs, batch_targets = inputs[:take], targets[:take]

            model.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()

            for name, module in modules.items():
                activations = registry.captured[f"{name}:in"].detach()
                grad_output = registry.captured[f"{name}:grad"].detach()
                flat_a = activations.reshape(-1, activations.shape[-1])
                flat_g = grad_output.reshape(-1, grad_output.shape[-1])
                if include_bias and getattr(module, "bias", None) is not None:
                    ones = torch.ones(flat_a.shape[0], 1, dtype=flat_a.dtype)
                    flat_a = torch.cat([flat_a, ones], dim=-1)

                a_contribution = flat_a.T @ flat_a
                g_contribution = flat_g.T @ flat_g
                a_sum[name] = a_sum.get(name, torch.zeros_like(a_contribution)) + a_contribution
                g_sum[name] = g_sum.get(name, torch.zeros_like(g_contribution)) + g_contribution
                n_rows[name] += flat_a.shape[0]

            seen += take

    if seen == 0:
        raise ValueError("dataloader produced no samples before reaching n_samples.")

    return {name: {"A": a_sum[name] / n_rows[name], "G": g_sum[name] / n_rows[name]} for name in modules}


def kfac_offdiagonal_energy(factors: Dict[str, Dict[str, Tensor]]) -> Dict[str, float]:
    """Off-diagonal energy fraction of each layer's K-FAC factors.

    Generic curvature diagnostic: how much of each factor's squared-Frobenius
    energy sits off the diagonal, i.e. how far the factor is from being
    diagonal. Near 0 means the factor is close to diagonal; near 1 means
    strong cross-feature correlations the Kronecker approximation is
    actively capturing.

    Args:
        factors: output of `kfac_factors`.

    Returns:
        Dict keyed the same as ``factors``, each mapping to the combined
        off-diagonal energy fraction across that layer's A and G factors.
    """
    result: Dict[str, float] = {}
    for name, ag in factors.items():
        total_energy = 0.0
        offdiag_energy = 0.0
        for factor in (ag["A"], ag["G"]):
            matrix = factor.detach().cpu().numpy() if isinstance(factor, Tensor) else factor
            total_energy += float((matrix**2).sum())
            offdiag_energy += float((matrix**2).sum() - (matrix.diagonal() ** 2).sum())
        result[name] = (offdiag_energy / total_energy) if total_energy > 0 else 0.0
    return result