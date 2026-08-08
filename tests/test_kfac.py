"""kfac.py tests: shape/symmetry/PSD invariants on the A/G factors, run
against a fused-QKV toy model, a split-QKV toy model, and a tiny HF GPT-2
(whose fused projection is a Conv1D rather than nn.Linear, exercising a third
structurally different module type).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modelgeometry.adapters import resolve_adapter
from modelgeometry.kfac import kfac_factors, kfac_offdiagonal_energy


def _loss_fn(predictions, target):
    logits = predictions.logits if hasattr(predictions, "logits") else predictions
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


def _assert_symmetric_psd(matrix: torch.Tensor, atol=1e-4):
    np.testing.assert_allclose(matrix, matrix.T, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert torch.all(eigenvalues >= -atol)


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "toy_split_model", "tiny_gpt2"])
def test_kfac_factors_shape_symmetry_and_psd(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    torch.manual_seed(0)
    inputs = torch.randint(0, 100, (4, 6))
    targets = torch.randint(0, 100, (4, 6))
    dataloader = [(inputs, targets)]

    adapter = resolve_adapter(model)
    factors = kfac_factors(model, dataloader, n_samples=4, loss_fn=_loss_fn, adapter=adapter)

    named_modules = {id(module): name for name, module in model.named_modules()}
    expected_keys = set()
    for layer_idx in range(adapter.num_layers()):
        for module in adapter.qkv_modules(layer_idx).values():
            expected_keys.add(f"{named_modules[id(module)]}.weight")
    assert set(factors.keys()) == expected_keys
    # Keys match real parameter names, so they align directly with
    # `fisher.diagonal_fisher` output and `model.named_parameters()`.
    param_names = {name for name, _ in model.named_parameters()}
    assert expected_keys <= param_names

    for ag in factors.values():
        _assert_symmetric_psd(ag["A"])
        _assert_symmetric_psd(ag["G"])


def test_kfac_factors_bias_homogenization_changes_a_shape(toy_fused_model, toy_token_batch):
    toy_fused_model.eval()
    dataloader = [(toy_token_batch, toy_token_batch)]
    adapter = resolve_adapter(toy_fused_model)

    with_bias = kfac_factors(toy_fused_model, dataloader, n_samples=2, loss_fn=_loss_fn, include_bias=True)
    without_bias = kfac_factors(toy_fused_model, dataloader, n_samples=2, loss_fn=_loss_fn, include_bias=False)

    hidden = adapter.hidden_size()
    key = "transformer.h.0.attn.c_attn.weight"
    a_with = with_bias[key]["A"]
    a_without = without_bias[key]["A"]
    assert a_with.shape == (hidden + 1, hidden + 1)
    assert a_without.shape == (hidden, hidden)


def test_kfac_factors_raises_on_empty_dataloader(toy_fused_model):
    with pytest.raises(ValueError):
        kfac_factors(toy_fused_model, [], n_samples=4, loss_fn=_loss_fn)


# ---- kfac_offdiagonal_energy -----------------------------------------------


def test_kfac_offdiagonal_energy_diagonal_factors_is_zero():
    factors = {"layer0.qkv": {"A": torch.eye(4), "G": torch.diag(torch.tensor([1.0, 2.0, 3.0]))}}
    energy = kfac_offdiagonal_energy(factors)
    assert energy["layer0.qkv"] == pytest.approx(0.0, abs=1e-8)


def test_kfac_offdiagonal_energy_dense_factor_is_positive():
    dense = torch.ones(3, 3)  # fully off-diagonal-correlated (all entries equal)
    factors = {"layer0.qkv": {"A": dense, "G": dense}}
    energy = kfac_offdiagonal_energy(factors)
    assert energy["layer0.qkv"] > 0.5