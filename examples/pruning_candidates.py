"""Pruning-candidate detection: flag attention heads that attend narrowly
(low entropy, low effective rank) across a batch of inputs, following the
head-pruning signal described in Michel et al., 2019 and Voita et al., 2019.

Self-contained: builds a small, randomly initialized GPT-2-style model (no
download, no external dataset) and runs it on synthetic token batches. Swap
in your own pretrained model and real dataloader — nothing here depends on
that choice.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from modelgeometry import (
    HookRegistry,
    attention_effective_rank,
    attention_entropy,
    capture_attention_weights,
    resolve_adapter,
)


def main() -> None:
    torch.manual_seed(0)
    config = GPT2Config(n_embd=64, n_head=8, n_layer=4, n_positions=32, n_ctx=32, vocab_size=200)
    config._attn_implementation = "eager"  # required to capture attention weights
    model = GPT2LMHeadModel(config)
    model.eval()

    adapter = resolve_adapter(model)
    n_batches, batch_size, seq_len = 5, 4, 16

    # Accumulate per-head entropy/effective-rank across several batches and layers.
    per_head_entropy = {}
    per_head_rank = {}
    for layer_idx in range(adapter.num_layers()):
        entropies, ranks = [], []
        registry = HookRegistry()
        with registry:
            capture_attention_weights(registry, "attn", adapter.attention_module(layer_idx))
            for _ in range(n_batches):
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                with torch.no_grad():
                    model(input_ids, output_attentions=True)
                weights = registry.captured["attn"]  # (batch, heads, seq, seq)
                entropies.append(attention_entropy(weights).mean(axis=(0, 2)))  # per head
                ranks.append(attention_effective_rank(weights).mean(axis=0))  # per head
        per_head_entropy[layer_idx] = np.stack(entropies).mean(axis=0)
        per_head_rank[layer_idx] = np.stack(ranks).mean(axis=0)

    print(f"{'layer':>5} {'head':>5} {'mean entropy':>14} {'mean eff. rank':>16}")
    candidates = []
    for layer_idx, entropies in per_head_entropy.items():
        ranks = per_head_rank[layer_idx]
        for head_idx, (entropy, rank) in enumerate(zip(entropies.tolist(), ranks.tolist())):
            print(f"{layer_idx:>5} {head_idx:>5} {entropy:>14.4f} {rank:>16.4f}")
            candidates.append((entropy + rank, layer_idx, head_idx))

    candidates.sort()
    print("\nLowest-entropy / lowest-rank heads (pruning candidates first):")
    for score, layer_idx, head_idx in candidates[:3]:
        print(f"  layer {layer_idx}, head {head_idx} (score={score:.4f})")


if __name__ == "__main__":
    main()