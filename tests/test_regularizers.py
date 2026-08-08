"""regularizers.py tests: each penalty computed against real models (checking
gradient flow and zero-at-anchor properties where applicable), run across
structurally different adapters where the penalty depends on adapter-derived
statistics (K-FAC), or across a toy fixture and a tiny HF GPT-2 otherwise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from modelgeometry.adapters import resolve_adapter
from modelgeometry.fisher import diagonal_fisher
from modelgeometry.kfac import kfac_factors
from modelgeometry.regularizers import EWCPenalty, KFACPenalty, L2Penalty, SynapticIntelligencePenalty


def _loss_fn(predictions, target):
    logits = predictions.logits if hasattr(predictions, "logits") else predictions
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


# ---- L2Penalty --------------------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_l2_penalty_matches_manual_computation_and_flows_gradient(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    reg = L2Penalty(model, weight=2.0)
    expected = 0.5 * 2.0 * sum((p**2).sum() for p in model.parameters() if p.requires_grad)
    torch.testing.assert_close(reg.penalty(), expected)

    model.zero_grad(set_to_none=True)
    reg.penalty().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)


# ---- EWCPenalty ---------------------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_ewc_penalty_zero_at_anchor_and_positive_after_drift(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    inputs = torch.randint(0, 100, (2, 6))
    fisher = diagonal_fisher(model, [(inputs, inputs)], n_samples=2, loss_fn=_loss_fn)
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}

    reg = EWCPenalty(model, fisher=fisher, anchor_params=anchor, weight=1.0)
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-8)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))

    assert reg.penalty().item() > 0.0

    model.zero_grad(set_to_none=True)
    reg.penalty().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)


# ---- SynapticIntelligencePenalty ----------------------------------------------


def test_synaptic_intelligence_zero_after_consolidate_then_positive_after_drift(toy_fused_model, toy_token_batch):
    model = toy_fused_model
    model.train()
    reg = SynapticIntelligencePenalty(model, damping=0.1, weight=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(3):
        optimizer.zero_grad()
        loss = _loss_fn(model(toy_token_batch), toy_token_batch)
        loss.backward()
        optimizer.step()
        reg.step()

    reg.consolidate()
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-8)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))

    assert reg.penalty().item() > 0.0

    model.zero_grad(set_to_none=True)
    reg.penalty().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)


def test_synaptic_intelligence_omega_accumulates_after_training():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    reg = SynapticIntelligencePenalty(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(3, 4)
    target = torch.randn(3, 4)

    for _ in range(5):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        reg.step()

    reg.consolidate()
    assert any(torch.any(omega != 0) for omega in reg._omega.values())


# ---- KFACPenalty ---------------------------------------------------------------


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "toy_split_model", "tiny_gpt2"])
def test_kfac_penalty_zero_at_anchor_and_positive_after_drift(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    inputs = torch.randint(0, 100, (2, 6))
    dataloader = [(inputs, inputs)]
    factors = kfac_factors(model, dataloader, n_samples=2, loss_fn=_loss_fn, include_bias=False)
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}

    reg = KFACPenalty(model, kfac_factors=factors, anchor_params=anchor, weight=1.0)
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-6)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in factors:
                p.add_(0.05 * torch.randn_like(p))

    assert reg.penalty().item() > 0.0

    model.zero_grad(set_to_none=True)
    reg.penalty().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)


def test_kfac_penalty_skips_shape_incompatible_layers(toy_fused_model):
    model = toy_fused_model
    model.eval()
    inputs = torch.randint(0, 100, (2, 6))
    # include_bias=True homogenizes A, making its shape incompatible with the
    # raw (out_features, in_features) parameter — the penalty should skip
    # these layers rather than raise.
    factors = kfac_factors(model, [(inputs, inputs)], n_samples=2, loss_fn=_loss_fn, include_bias=True)
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    reg = KFACPenalty(model, kfac_factors=factors, anchor_params=anchor, weight=1.0)
    penalty = reg.penalty()
    assert isinstance(penalty, torch.Tensor)  # never silently degrades to a plain float
    assert penalty.item() == pytest.approx(0.0)