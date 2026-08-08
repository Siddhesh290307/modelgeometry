"""Minimal hand-written decoder-only transformer with separate Q/K/V projections.

Mirrors a LLaMA/GPT-NeoX-style layout (``model.model.layers[i].self_attn.{q,k,v}_proj``)
so it exercises a structurally different auto-detection path than
``toy_transformer.ToyFusedTransformer`` (separate projections instead of a
fused one, ``model.layers`` instead of ``transformer.h``, ``self_attn``
instead of ``attn``).
"""

from __future__ import annotations

import torch
from torch import nn


class ToySplitAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        self.attn_weights = weights
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(out)


class ToySplitBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.self_attn = ToySplitAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Model(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers


class ToySplitTransformer(nn.Module):
    """Hand-written transformer exposing a ``model.layers`` block container."""

    def __init__(self, hidden_size: int = 32, num_heads: int = 4, num_layers: int = 2, vocab_size: int = 100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = _Model(nn.ModuleList([ToySplitBlock(hidden_size, num_heads) for _ in range(num_layers)]))
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = block(x)
        return self.head(self.ln_f(x))
