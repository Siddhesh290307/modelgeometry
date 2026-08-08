"""Composable regularization penalties: established, published formulations only.

Each class exposes a single ``penalty() -> Tensor`` method so it composes
into an arbitrary training loop as ``loss = task_loss + reg.penalty()``. None
of these prescribe a dataset, task, or hyperparameter grid — the caller
supplies the model, any Fisher/K-FAC statistics, and anchor parameters.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


def _zero_scalar(model: nn.Module) -> Tensor:
    """A 0-valued scalar tensor on the model's device/dtype.

    Used as the running total's initial value in every `penalty()` below, so
    the return type stays a `Tensor` (never a plain Python float) even when
    no parameter ends up contributing a term — `loss = task_loss +
    reg.penalty()` must always type-check the same way.
    """
    p = next(model.parameters())
    return torch.zeros((), device=p.device, dtype=p.dtype)


class L2Penalty:
    """Standard L2 (weight decay) penalty: ``0.5 * weight * sum(p^2)`` over trainable parameters."""

    def __init__(self, model: nn.Module, weight: float = 1.0):
        self.model = model
        self.weight = weight

    def penalty(self) -> Tensor:
        total = _zero_scalar(self.model)
        for p in self.model.parameters():
            if p.requires_grad:
                total = total + (p**2).sum()
        return 0.5 * self.weight * total


class EWCPenalty:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    ``0.5 * weight * sum(fisher[name] * (p - anchor[name])^2)`` over every
    parameter present in both ``fisher`` and ``anchor_params``.
    """

    def __init__(
        self,
        model: nn.Module,
        fisher: Dict[str, Tensor],
        anchor_params: Dict[str, Tensor],
        weight: float = 1.0,
    ):
        self.model = model
        self.fisher = fisher
        self.anchor_params = anchor_params
        self.weight = weight

    def penalty(self) -> Tensor:
        total = _zero_scalar(self.model)
        for name, p in self.model.named_parameters():
            if name not in self.fisher or name not in self.anchor_params:
                continue
            delta = p - self.anchor_params[name]
            total = total + (self.fisher[name] * delta**2).sum()
        return 0.5 * self.weight * total


class SynapticIntelligencePenalty:
    """Synaptic Intelligence (Zenke et al., 2017).

    Accumulates a running per-parameter path-integral importance ``w`` during
    training (call `step()` once per optimizer step, while the just-taken
    step's gradients are still populated on ``p.grad``), then folds it into a
    per-parameter importance ``Omega`` at each task boundary (call
    `consolidate()`). The penalty at any point is
    ``weight * sum(Omega * (p - reference)^2)``, where ``reference`` is the
    parameter value as of the last `consolidate()` call.
    """

    def __init__(self, model: nn.Module, damping: float = 0.1, weight: float = 1.0):
        self.model = model
        self.damping = damping
        self.weight = weight
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self._omega = {name: torch.zeros_like(p) for name, p in trainable.items()}
        self._path_integral = {name: torch.zeros_like(p) for name, p in trainable.items()}
        self._prev_params = {name: p.detach().clone() for name, p in trainable.items()}
        self._reference_params = {name: p.detach().clone() for name, p in trainable.items()}

    def step(self) -> None:
        """Accumulate this step's contribution to the path integral. Call after `optimizer.step()`, before `optimizer.zero_grad()`."""
        for name, p in self.model.named_parameters():
            if name not in self._path_integral or p.grad is None:
                continue
            delta = p.detach() - self._prev_params[name]
            self._path_integral[name] += -p.grad.detach() * delta
            self._prev_params[name] = p.detach().clone()

    def consolidate(self) -> None:
        """Fold the accumulated path integral into `Omega` and reset the reference point. Call at a task boundary."""
        for name, p in self.model.named_parameters():
            if name not in self._omega:
                continue
            total_change = p.detach() - self._reference_params[name]
            self._omega[name] += self._path_integral[name] / (total_change**2 + self.damping)
            self._path_integral[name] = torch.zeros_like(p)
            self._reference_params[name] = p.detach().clone()

    def penalty(self) -> Tensor:
        total = _zero_scalar(self.model)
        for name, p in self.model.named_parameters():
            if name not in self._omega:
                continue
            delta = p - self._reference_params[name]
            total = total + (self._omega[name] * delta**2).sum()
        return self.weight * total


class KFACPenalty:
    """Kronecker-factored curvature penalty (Martens & Grosse, 2015).

    ``0.5 * weight * sum_layer tr(G @ (W - W_anchor) @ A @ (W - W_anchor)^T)``
    using K-FAC factors (e.g. from `kfac.kfac_factors`). Layers whose anchor
    is missing, or whose parameter shape doesn't match its ``A``/``G``
    factors (e.g. a bias-homogenized ``A`` with no corresponding augmented
    anchor), are skipped rather than guessed at.
    """

    def __init__(
        self,
        model: nn.Module,
        kfac_factors: Dict[str, Dict[str, Tensor]],
        anchor_params: Dict[str, Tensor],
        weight: float = 1.0,
    ):
        self.model = model
        self.kfac_factors = kfac_factors
        self.anchor_params = anchor_params
        self.weight = weight

    def penalty(self) -> Tensor:
        total = _zero_scalar(self.model)
        params = dict(self.model.named_parameters())
        for name, factors in self.kfac_factors.items():
            if name not in params or name not in self.anchor_params:
                continue
            p = params[name]
            a, g = factors["A"], factors["G"]
            if p.shape != (g.shape[0], a.shape[0]):
                continue
            delta = p - self.anchor_params[name]
            total = total + torch.trace(g @ delta @ a @ delta.T)
        return 0.5 * self.weight * total