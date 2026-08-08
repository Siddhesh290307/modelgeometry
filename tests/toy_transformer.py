"""Minimal hand-written decoder-only transformer with a fused QKV projection.

Mirrors GPT-2's layout (``model.transformer.h[i].attn.c_attn``) so it exercises
the same auto-detection path as a real GPT-2 model, but stays tiny for fast
tests. Its attention module stores its most recent attention probabilities on
``self.attn_weights``, following the attribute convention documented in
``modelgeometry.hooks.capture_attention_weights``.
"""

from __future__ import annotations

import torch
from torch import nn


class ToyFusedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        self.attn_weights = weights
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(out)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = ToyFusedAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Transformer(nn.Module):
    def __init__(self, h: nn.ModuleList):
        super().__init__()
        self.h = h


class ToyFusedTransformer(nn.Module):
    """Hand-written transformer exposing a ``transformer.h`` block container."""

    def __init__(self, hidden_size: int = 32, num_heads: int = 4, num_layers: int = 2, vocab_size: int = 100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer = _Transformer(nn.ModuleList([ToyBlock(hidden_size, num_heads) for _ in range(num_layers)]))
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.transformer.h:
            x = block(x)
        return self.head(self.ln_f(x))
