"""Phase 0 gate: resolve_adapter and ModelAdapter must work identically across
structurally different models — a fused-QKV toy transformer, a split-QKV toy
transformer, a tiny HuggingFace GPT-2, and a tiny timm ViT. No later phase
should proceed until every test in this file passes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from modelgeometry.adapters import FusedQKVAdapter, SplitQKVAdapter, resolve_adapter
from toy_transformer import ToyFusedTransformer
from toy_transformer_split import ToySplitTransformer

HIDDEN_SIZE = 32
NUM_HEADS = 4
NUM_LAYERS = 2

_FUSED_CASES = ["toy_fused_model", "tiny_gpt2", "tiny_vit"]
_SPLIT_CASES = ["toy_split_model"]
_ALL_CASES = _FUSED_CASES + _SPLIT_CASES


@pytest.mark.parametrize("model_fixture", _FUSED_CASES)
def test_resolve_adapter_detects_fused(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    assert isinstance(adapter, FusedQKVAdapter)


@pytest.mark.parametrize("model_fixture", _SPLIT_CASES)
def test_resolve_adapter_detects_split(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    assert isinstance(adapter, SplitQKVAdapter)


@pytest.mark.parametrize("model_fixture", _ALL_CASES)
def test_num_layers(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    assert adapter.num_layers() == NUM_LAYERS


@pytest.mark.parametrize("model_fixture", _ALL_CASES)
def test_qkv_weights_shape_and_dtype(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    for layer_idx in range(adapter.num_layers()):
        qkv = adapter.qkv_weights(layer_idx)
        for weight in (qkv.q, qkv.k, qkv.v):
            assert isinstance(weight, torch.Tensor)
            assert weight.shape == (HIDDEN_SIZE, HIDDEN_SIZE)


@pytest.mark.parametrize("model_fixture", _ALL_CASES)
def test_hidden_size(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    assert adapter.hidden_size() == HIDDEN_SIZE


@pytest.mark.parametrize("model_fixture", _ALL_CASES)
def test_num_heads_and_head_dim(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    assert adapter.num_heads() == NUM_HEADS
    assert adapter.head_dim() == HIDDEN_SIZE // NUM_HEADS


@pytest.mark.parametrize("model_fixture", _ALL_CASES)
def test_attention_module_is_reachable(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    for layer_idx in range(adapter.num_layers()):
        assert isinstance(adapter.attention_module(layer_idx), nn.Module)


@pytest.mark.parametrize("model_fixture", _FUSED_CASES)
def test_qkv_modules_fused_returns_single_module(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    modules = adapter.qkv_modules(0)
    assert set(modules.keys()) == {"qkv"}
    assert isinstance(modules["qkv"], nn.Module)


@pytest.mark.parametrize("model_fixture", _SPLIT_CASES)
def test_qkv_modules_split_returns_three_modules(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    modules = adapter.qkv_modules(0)
    assert set(modules.keys()) == {"q", "k", "v"}
    for module in modules.values():
        assert isinstance(module, nn.Module)


def test_explicit_overrides_bypass_autodetection():
    model = ToySplitTransformer(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
    adapter = resolve_adapter(
        model,
        layer_path="model.layers",
        attn_name="self_attn",
        qkv_names=("q_proj", "k_proj", "v_proj"),
    )
    assert isinstance(adapter, SplitQKVAdapter)
    assert adapter.num_layers() == NUM_LAYERS
    assert adapter.qkv_weights(0).q.shape == (HIDDEN_SIZE, HIDDEN_SIZE)


def test_fused_qkv_names_override():
    model = ToyFusedTransformer(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
    # Exercise the explicit qkv_names length-1 (fused) branch directly, rather
    # than relying on auto-detection, and confirm it matches auto-detection.
    explicit = resolve_adapter(
        model,
        layer_path="transformer.h",
        attn_name="attn",
        qkv_names=("c_attn",),
    )
    auto = resolve_adapter(model)
    assert isinstance(explicit, FusedQKVAdapter)
    assert torch.equal(explicit.qkv_weights(0).q, auto.qkv_weights(0).q)


def test_unrecognized_model_raises_with_guidance():
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="Could not auto-detect the transformer block container"):
        resolve_adapter(model)


def test_missing_attention_submodule_raises_with_guidance():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(4, 4)

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block()])

    with pytest.raises(ValueError, match="Could not auto-detect the attention submodule"):
        resolve_adapter(Container())
