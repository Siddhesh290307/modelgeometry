"""fisher.py tests. The critical correctness property (documented in the
module docstring): per-sample squared-then-averaged gradients differ from
batch-averaged-then-squared gradients, and `diagonal_fisher` must match the
former — verified two ways: (1) against a batch-averaged-then-squared
computation showing they differ, and (2) against an independently computed
per-sample loop using plain autograd (not `torch.func`) as ground truth.
Run against both a toy fixture and a tiny HF GPT-2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modelgeometry.fisher import diagonal_fisher, fisher_layer_summary


def _loss_fn(predictions, target):
    logits = predictions.logits if hasattr(predictions, "logits") else predictions
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


def _per_sample_autograd_ground_truth(model, inputs, targets):
    """Reference diagonal Fisher via a plain per-sample autograd loop (no torch.func)."""
    params = [p for p in model.parameters() if p.requires_grad]
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n = inputs.shape[0]
    for i in range(n):
        model.zero_grad(set_to_none=True)
        predictions = model(inputs[i : i + 1])
        loss = _loss_fn(predictions, targets[i : i + 1])
        grads = torch.autograd.grad(loss, params)
        for (name, _), g in zip(model.named_parameters(), grads):
            fisher[name] += g**2
    return {name: value / n for name, value in fisher.items()}


def _batch_averaged_then_squared(model, inputs, targets):
    """The incorrect approximation the module docstring warns against."""
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = _loss_fn(predictions, targets)  # averaged over the batch internally by cross_entropy
    grads = torch.autograd.grad(loss, params)
    return {name: g**2 for (name, _), g in zip(model.named_parameters(), grads)}


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_diagonal_fisher_matches_per_sample_ground_truth_not_batch_averaged(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    torch.manual_seed(0)
    inputs = torch.randint(0, 100, (6, 6))
    targets = torch.randint(0, 100, (6, 6))
    dataloader = [(inputs, targets)]

    fisher = diagonal_fisher(model, dataloader, n_samples=6, loss_fn=_loss_fn)
    ground_truth = _per_sample_autograd_ground_truth(model, inputs, targets)
    wrong = _batch_averaged_then_squared(model, inputs, targets)

    # Matches the correct per-sample computation...
    for name in ground_truth:
        torch.testing.assert_close(fisher[name], ground_truth[name], atol=1e-5, rtol=1e-4)

    # ...and measurably differs from the batch-averaged-then-squared approximation
    # for at least one parameter (the model is past initialization / has nonzero
    # gradient variance across samples, so the two must diverge).
    any_differs = any(not torch.allclose(fisher[name], wrong[name], atol=1e-6) for name in fisher)
    assert any_differs


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "tiny_gpt2"])
def test_diagonal_fisher_respects_n_samples_across_batches(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    torch.manual_seed(1)
    batch1 = (torch.randint(0, 100, (4, 6)), torch.randint(0, 100, (4, 6)))
    batch2 = (torch.randint(0, 100, (4, 6)), torch.randint(0, 100, (4, 6)))
    dataloader = [batch1, batch2]

    fisher = diagonal_fisher(model, dataloader, n_samples=6, loss_fn=_loss_fn)
    all_inputs = torch.cat([batch1[0], batch2[0]])[:6]
    all_targets = torch.cat([batch1[1], batch2[1]])[:6]
    ground_truth = _per_sample_autograd_ground_truth(model, all_inputs, all_targets)

    for name in ground_truth:
        torch.testing.assert_close(fisher[name], ground_truth[name], atol=1e-5, rtol=1e-4)


def test_diagonal_fisher_raises_on_empty_dataloader(toy_fused_model):
    with pytest.raises(ValueError):
        diagonal_fisher(toy_fused_model, [], n_samples=4, loss_fn=_loss_fn)


def test_fisher_layer_summary_structure(toy_fused_model, toy_token_batch):
    toy_fused_model.eval()
    dataloader = [(toy_token_batch, toy_token_batch)]
    fisher = diagonal_fisher(toy_fused_model, dataloader, n_samples=2, loss_fn=_loss_fn)
    summary = fisher_layer_summary(fisher, top_k=3)

    assert set(summary.keys()) == {"per_parameter_mass", "total_mass", "top_k_mass_fraction", "effective_rank"}
    assert summary["total_mass"] == pytest.approx(sum(summary["per_parameter_mass"].values()))
    assert 0.0 <= summary["top_k_mass_fraction"] <= 1.0 + 1e-8
    assert 1.0 - 1e-6 <= summary["effective_rank"] <= len(fisher) + 1e-6


def test_fisher_layer_summary_all_mass_on_one_param_gives_rank_one():
    fisher = {
        "a": torch.tensor([1.0, 1.0]),
        "b": torch.zeros(3),
        "c": torch.zeros(1),
    }
    summary = fisher_layer_summary(fisher)
    assert summary["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert summary["top_k_mass_fraction"] == pytest.approx(1.0)