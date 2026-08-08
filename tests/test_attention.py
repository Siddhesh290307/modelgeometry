"""attention.py tests: known-value checks on synthetic attention matrices,
plus real captured attention weights from both a toy fixture and a tiny
HF GPT-2, and real Q/K/V activations derived from two structurally different
adapters (fused-QKV and split-QKV).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modelgeometry.adapters import resolve_adapter
from modelgeometry.attention import attention_effective_rank, attention_entropy, qkv_norm_stats
from modelgeometry.hooks import HookRegistry, capture_attention_weights

# ---- attention_entropy: known values --------------------------------------


def test_attention_entropy_uniform_distribution_is_log_n():
    n = 8
    uniform = np.full((n,), 1.0 / n)
    assert attention_entropy(uniform) == pytest.approx(np.log(n), abs=1e-8)


def test_attention_entropy_one_hot_distribution_is_zero():
    one_hot = np.array([0.0, 0.0, 1.0, 0.0])
    assert attention_entropy(one_hot) == pytest.approx(0.0, abs=1e-8)


def test_attention_entropy_batched_shape():
    weights = np.random.default_rng(0).dirichlet(np.ones(5), size=(2, 3, 4))  # (batch, heads, seq_q, seq_k)
    entropy = attention_entropy(weights)
    assert entropy.shape == (2, 3, 4)
    assert np.all(entropy >= -1e-8)
    assert np.all(entropy <= np.log(5) + 1e-8)


# ---- attention_effective_rank: known values -------------------------------


def test_attention_effective_rank_uniform_rows_is_rank_one():
    n = 6
    uniform_rows = np.full((n, n), 1.0 / n)
    assert attention_effective_rank(uniform_rows) == pytest.approx(1.0, abs=1e-6)


def test_attention_effective_rank_identity_is_full_rank():
    n = 6
    identity = np.eye(n)  # a valid "attention matrix": each row sums to 1
    assert attention_effective_rank(identity) == pytest.approx(float(n), abs=1e-6)


def test_attention_effective_rank_batched_shape():
    rng = np.random.default_rng(1)
    weights = rng.dirichlet(np.ones(5), size=(2, 3, 5))  # (batch, heads, seq_q, seq_k)
    ranks = attention_effective_rank(weights)
    assert ranks.shape == (2, 3)
    assert np.all(ranks >= 1.0 - 1e-6)
    assert np.all(ranks <= 5.0 + 1e-6)


# ---- qkv_norm_stats: known values ------------------------------------------


def test_qkv_norm_stats_known_values():
    q = np.array([[3.0, 4.0], [0.0, 0.0]])  # norms: 5, 0
    k = np.array([[1.0, 0.0]])  # norm: 1
    v = np.array([[0.0, 2.0]])  # norm: 2
    stats = qkv_norm_stats(q, k, v)
    assert stats["q"]["mean_norm"] == pytest.approx(2.5)
    assert stats["q"]["max_norm"] == pytest.approx(5.0)
    assert stats["k"]["mean_norm"] == pytest.approx(1.0)
    assert stats["v"]["mean_norm"] == pytest.approx(2.0)


# ---- against real captured attention weights, two different capture conventions ----


def test_attention_entropy_and_rank_on_toy_fixture(toy_fused_model, toy_token_batch):
    adapter = resolve_adapter(toy_fused_model)
    registry = HookRegistry()
    with registry:
        capture_attention_weights(registry, "attn0", adapter.attention_module(0))
        toy_fused_model(toy_token_batch)
    weights = registry.captured["attn0"]  # (batch, heads, seq, seq)

    entropy = attention_entropy(weights)
    assert entropy.shape == weights.shape[:-1]
    assert np.all(entropy >= -1e-6)

    rank = attention_effective_rank(weights)
    assert rank.shape == weights.shape[:-2]
    assert np.all(rank >= 1.0 - 1e-4)
    assert np.all(rank <= weights.shape[-1] + 1e-4)


def test_attention_entropy_and_rank_on_tiny_gpt2(tiny_gpt2, toy_token_batch):
    tiny_gpt2.eval()
    adapter = resolve_adapter(tiny_gpt2)
    registry = HookRegistry()
    with registry:
        capture_attention_weights(registry, "attn0", adapter.attention_module(0))
        tiny_gpt2(toy_token_batch, output_attentions=True)
    weights = registry.captured["attn0"]

    entropy = attention_entropy(weights)
    assert entropy.shape == weights.shape[:-1]
    assert np.all(entropy >= -1e-6)

    rank = attention_effective_rank(weights)
    assert rank.shape == weights.shape[:-2]
    assert np.all(rank >= 1.0 - 1e-4)


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "toy_split_model"])
def test_qkv_norm_stats_on_real_projected_activations(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    qkv = adapter.qkv_weights(0)
    num_heads, head_dim = adapter.num_heads(), adapter.head_dim()

    batch, seq = 2, 5
    x = torch.randn(batch, seq, adapter.hidden_size())
    q = (x @ qkv.q.T).view(batch, seq, num_heads, head_dim)
    k = (x @ qkv.k.T).view(batch, seq, num_heads, head_dim)
    v = (x @ qkv.v.T).view(batch, seq, num_heads, head_dim)

    stats = qkv_norm_stats(q, k, v)
    assert set(stats.keys()) == {"q", "k", "v"}
    for name in ("q", "k", "v"):
        assert stats[name]["mean_norm"] > 0
        assert stats[name]["max_norm"] >= stats[name]["mean_norm"]