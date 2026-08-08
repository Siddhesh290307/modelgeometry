"""Training-health monitoring: log weight-space geometry every few steps of
an ordinary PyTorch training loop with `GeometryTracker`, to watch for
things like effective-rank collapse in a projection matrix.

Self-contained: trains a small GPT-2-style model on synthetic random token
sequences (no download, no real dataset) — GeometryTracker itself has no
opinion about the training loop, data, or task; call `log_step` from any of
them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from modelgeometry import GeometryTracker, effective_rank


def qkv0_effective_rank(model, adapter):
    return effective_rank(adapter.qkv_weights(0).q)


def main() -> None:
    torch.manual_seed(0)
    config = GPT2Config(n_embd=64, n_head=8, n_layer=4, n_positions=32, n_ctx=32, vocab_size=200)
    model = GPT2LMHeadModel(config)
    model.train()

    tracker = GeometryTracker(model, metrics=[("qkv0_effective_rank", qkv0_effective_rank)])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    print(f"{'step':>5} {'loss':>10} {'qkv0_effective_rank':>20}")
    for step in range(30):
        input_ids = torch.randint(0, config.vocab_size, (4, 16))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(input_ids).logits.view(-1, config.vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            record = tracker.log_step(step)
            print(f"{step:>5} {loss.item():>10.4f} {record['qkv0_effective_rank']:>20.4f}")

    print(f"\nLogged {len(tracker.history)} records; inspect tracker.history for the full series.")


if __name__ == "__main__":
    main()