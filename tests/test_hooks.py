"""HookRegistry tests, run against both a toy fixture (attribute-convention
attention capture) and a tiny HF GPT-2 (output-tuple-convention attention
capture) to cover both conventions documented on capture_attention_weights.
"""

from __future__ import annotations

import torch

from modelgeometry.adapters import resolve_adapter
from modelgeometry.hooks import HookRegistry, capture_attention_weights


def test_capture_output_and_cleanup(toy_fused_model, toy_token_batch):
    registry = HookRegistry()
    block0 = toy_fused_model.transformer.h[0]
    with registry:
        registry.capture_output("block0", block0)
        toy_fused_model(toy_token_batch)
        assert "block0" in registry.captured
        assert registry.captured["block0"].shape == (toy_token_batch.shape[0], toy_token_batch.shape[1], 32)

    # Hooks are removed on exit: a subsequent forward pass must not update the capture.
    registry.captured.clear()
    toy_fused_model(toy_token_batch)
    assert registry.captured == {}


def test_capture_input(toy_fused_model, toy_token_batch):
    registry = HookRegistry()
    block0 = toy_fused_model.transformer.h[0]
    with registry:
        registry.capture_input("block0_in", block0)
        toy_fused_model(toy_token_batch)
    assert registry.captured["block0_in"].shape == (toy_token_batch.shape[0], toy_token_batch.shape[1], 32)


def test_capture_grad_output(toy_fused_model, toy_token_batch):
    registry = HookRegistry()
    block0 = toy_fused_model.transformer.h[0]
    with registry:
        registry.capture_grad_output("block0_grad", block0)
        out = toy_fused_model(toy_token_batch)
        out.sum().backward()
    assert registry.captured["block0_grad"].shape == (toy_token_batch.shape[0], toy_token_batch.shape[1], 32)


def test_capture_attention_weights_attribute_convention(toy_fused_model, toy_token_batch):
    adapter = resolve_adapter(toy_fused_model)
    registry = HookRegistry()
    with registry:
        capture_attention_weights(registry, "attn0", adapter.attention_module(0))
        toy_fused_model(toy_token_batch)
    weights = registry.captured["attn0"]
    assert weights.shape == (toy_token_batch.shape[0], 4, toy_token_batch.shape[1], toy_token_batch.shape[1])
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_capture_attention_weights_output_tuple_convention(tiny_gpt2, toy_token_batch):
    tiny_gpt2.eval()  # attention dropout would otherwise break the sums-to-1 check below
    adapter = resolve_adapter(tiny_gpt2)
    registry = HookRegistry()
    with registry:
        capture_attention_weights(registry, "attn0", adapter.attention_module(0))
        tiny_gpt2(toy_token_batch, output_attentions=True)
    weights = registry.captured["attn0"]
    assert weights.shape == (toy_token_batch.shape[0], 4, toy_token_batch.shape[1], toy_token_batch.shape[1])
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
