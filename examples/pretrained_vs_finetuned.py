"""Pretrained-vs-finetuned comparison: use `compare_checkpoints` to see what
shifted in a model's weight-space geometry between two checkpoints.

Self-contained: builds a small GPT-2-style "pretrained" stand-in and derives
a "finetuned" stand-in from it via a few synthetic gradient-descent steps on
random data (no download, no real dataset, no specific finetuning protocol —
swap in your own two real checkpoints and this workflow doesn't change).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from modelgeometry import compare_checkpoints, distributional_distance, effective_rank, row_cosine_similarity


def qkv0_effective_rank(model, adapter):
    return effective_rank(adapter.qkv_weights(0).q)


def qkv0_row_similarity_mean(model, adapter):
    sim = row_cosine_similarity(adapter.qkv_weights(0).q)
    off_diagonal_mask = ~np.eye(sim.shape[0], dtype=bool)
    return float(sim[off_diagonal_mask].mean())


def qkv0_weight_distance(model, adapter):
    # A dict-valued (non-numeric-diffable) custom metric, to show
    # compare_checkpoints handling both kinds side by side.
    qkv = adapter.qkv_weights(0)
    return {"q_vs_k_distance": distributional_distance(qkv.q.flatten(), qkv.k.flatten())}


def main() -> None:
    torch.manual_seed(0)
    config = GPT2Config(n_embd=64, n_head=8, n_layer=4, n_positions=32, n_ctx=32, vocab_size=200)
    pretrained = GPT2LMHeadModel(config)
    pretrained.eval()

    finetuned = copy.deepcopy(pretrained)
    finetuned.train()
    optimizer = torch.optim.SGD(finetuned.parameters(), lr=0.05)
    for _ in range(20):
        input_ids = torch.randint(0, config.vocab_size, (4, 16))
        optimizer.zero_grad()
        loss = F.cross_entropy(finetuned(input_ids).logits.view(-1, config.vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
    finetuned.eval()

    report = compare_checkpoints(
        pretrained,
        finetuned,
        metrics=[
            ("qkv0_effective_rank", qkv0_effective_rank),
            ("qkv0_row_similarity_mean", qkv0_row_similarity_mean),
            ("qkv0_weight_distance", qkv0_weight_distance),
        ],
    )

    for name, entry in report.items():
        print(f"{name}:")
        for key, value in entry.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()