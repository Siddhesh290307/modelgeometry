"""Shared fixtures: toy fused/split transformers plus tiny (untrained, randomly
initialized) HuggingFace GPT-2 and timm ViT models. Models are built from small
custom configs rather than downloaded pretrained checkpoints, so the suite is
fast and network-free.
"""

from __future__ import annotations

import pytest
import torch

from toy_transformer import ToyFusedTransformer
from toy_transformer_split import ToySplitTransformer

HIDDEN_SIZE = 32
NUM_HEADS = 4
NUM_LAYERS = 2
VOCAB_SIZE = 100
SEQ_LEN = 6
BATCH_SIZE = 2


@pytest.fixture
def toy_fused_model():
    return ToyFusedTransformer(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE)


@pytest.fixture
def toy_split_model():
    return ToySplitTransformer(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE)


@pytest.fixture
def toy_token_batch():
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))


@pytest.fixture
def tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_embd=HIDDEN_SIZE,
        n_head=NUM_HEADS,
        n_layer=NUM_LAYERS,
        n_positions=16,
        n_ctx=16,
        vocab_size=VOCAB_SIZE,
    )
    # Fused kernels (sdpa/flash) never materialize attention-weight tensors;
    # "eager" is required for anything that captures attention weights.
    config._attn_implementation = "eager"
    return transformers.GPT2LMHeadModel(config)


@pytest.fixture
def tiny_vit():
    timm = pytest.importorskip("timm")
    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        patch_size=8,
        embed_dim=HIDDEN_SIZE,
        depth=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_classes=10,
    )


@pytest.fixture
def tiny_image_batch():
    return torch.randn(BATCH_SIZE, 3, 32, 32)
