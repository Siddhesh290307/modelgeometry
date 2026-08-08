"""curvature.py tests: an exact synthetic check (quadratic loss, perturbation
at a zero-gradient point, so the second-order Taylor prediction is exact),
plus structural/sanity checks against two real models (toy fixture + tiny HF
GPT-2) where the loss isn't exactly quadratic so only invariants (not exact
numeric equality) are checked.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from modelgeometry.adapters import resolve_adapter
from modelgeometry.curvature import curvature_prediction_check
from modelgeometry.fisher import diagonal_fisher
from modelgeometry.kfac import kfac_factors


def _loss_fn(predictions, target):
    logits = predictions.logits if hasattr(predictions, "logits") else predictions
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


def test_diagonal_prediction_is_exact_for_a_quadratic_loss_at_a_stationary_point():
    # y = w . x (no bias); L = 0.5 * (y - target)^2 is exactly quadratic in w.
    # Choosing target = w0 . x makes the base point's gradient zero, so the
    # second-order Taylor prediction has no missing linear term.
    x = torch.tensor([1.0, 2.0])
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.4]]))
        target = model(x.unsqueeze(0))  # residual is exactly zero at this w

    def mse_loss(predictions, target):
        return 0.5 * (predictions - target).pow(2).mean()

    # Perturb only the first weight component to avoid diagonal-Fisher's
    # missing cross-term (off-diagonal Hessian entries) contaminating the comparison.
    delta = torch.tensor([[0.05, 0.0]])
    exact_diagonal_hessian = torch.tensor([[x[0] ** 2, x[1] ** 2]])  # diag(x x^T)

    result = curvature_prediction_check(
        model,
        loss_fn=mse_loss,
        batch=(x.unsqueeze(0), target),
        perturbations={"weight": delta},
        fisher={"weight": exact_diagonal_hessian},
    )

    assert result["baseline_loss"] == pytest.approx(0.0, abs=1e-10)
    assert result["actual_change"] == pytest.approx(result["predicted_change_diagonal"], abs=1e-8)


def test_diagonal_prediction_is_nonnegative_and_returns_expected_keys():
    x = torch.tensor([1.0, 2.0])
    model = torch.nn.Linear(2, 1, bias=False)
    target = torch.tensor([[1.0]])

    def mse_loss(predictions, target):
        return 0.5 * (predictions - target).pow(2).mean()

    result = curvature_prediction_check(
        model,
        loss_fn=mse_loss,
        batch=(x.unsqueeze(0), target),
        perturbations={"weight": torch.tensor([[0.01, -0.02]])},
        fisher={"weight": torch.tensor([[1.0, 1.0]])},
    )
    assert set(result.keys()) == {"baseline_loss", "perturbed_loss", "actual_change", "predicted_change_diagonal"}
    assert result["predicted_change_diagonal"] >= 0.0


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_curvature_prediction_check_against_real_diagonal_fisher(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    torch.manual_seed(0)
    inputs = torch.randint(0, 100, (4, 6))
    targets = torch.randint(0, 100, (4, 6))

    fisher = diagonal_fisher(model, [(inputs, targets)], n_samples=4, loss_fn=_loss_fn)

    # Small random perturbation of every parameter, scaled down so the model
    # stays close to its starting point (where the quadratic approximation is
    # most meaningful).
    perturbations = {name: 1e-3 * torch.randn_like(p) for name, p in model.named_parameters()}

    result = curvature_prediction_check(
        model,
        loss_fn=_loss_fn,
        batch=(inputs, targets),
        perturbations=perturbations,
        fisher=fisher,
    )
    assert result["predicted_change_diagonal"] >= 0.0
    assert torch.isfinite(torch.tensor(result["actual_change"]))
    assert torch.isfinite(torch.tensor(result["baseline_loss"]))


def test_curvature_prediction_check_restores_original_parameters(toy_fused_model, toy_token_batch):
    toy_fused_model.eval()
    original = {name: p.detach().clone() for name, p in toy_fused_model.named_parameters()}
    fisher = diagonal_fisher(toy_fused_model, [(toy_token_batch, toy_token_batch)], n_samples=2, loss_fn=_loss_fn)
    perturbations = {name: 0.01 * torch.randn_like(p) for name, p in toy_fused_model.named_parameters()}

    curvature_prediction_check(
        toy_fused_model,
        loss_fn=_loss_fn,
        batch=(toy_token_batch, toy_token_batch),
        perturbations=perturbations,
        fisher=fisher,
    )

    for name, p in toy_fused_model.named_parameters():
        torch.testing.assert_close(p, original[name])


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_curvature_prediction_check_with_block_factors_is_positive(model_fixture, request, toy_token_batch):
    # tiny_gpt2's fused projection is a Conv1D storing weight as (in, out) —
    # the transpose of nn.Linear's (out, in). This regression-tests the
    # block prediction actually engaging (not silently skipping every layer
    # and returning a phantom 0.0) regardless of that storage orientation.
    model = request.getfixturevalue(model_fixture)
    model.eval()
    adapter = resolve_adapter(model)
    dataloader = [(toy_token_batch, toy_token_batch)]
    fisher = diagonal_fisher(model, dataloader, n_samples=2, loss_fn=_loss_fn)
    block_factors = kfac_factors(model, dataloader, n_samples=2, loss_fn=_loss_fn, include_bias=False)

    qkv_weight_name = next(name for name, _ in model.named_parameters() if name == "transformer.h.0.attn.c_attn.weight")
    delta_shape = adapter.qkv_modules(0)["qkv"].weight.shape
    perturbations = {qkv_weight_name: 1e-3 * torch.randn(*delta_shape)}

    result = curvature_prediction_check(
        model,
        loss_fn=_loss_fn,
        batch=(toy_token_batch, toy_token_batch),
        perturbations=perturbations,
        fisher=fisher,
        block_factors=block_factors,
    )
    assert "predicted_change_block" in result
    # Strictly positive, not just non-negative: a silently-skipped layer
    # (the bug this test guards against) would produce exactly 0.0.
    assert result["predicted_change_block"] > 1e-10