"""Attention and activation geometry metrics.

These functions operate on tensors already captured from a forward pass (e.g.
via `modelgeometry.hooks.HookRegistry` and `capture_attention_weights`)
rather than on a model directly, so they compose with any hook-capture
strategy and require no assumptions about how the caller ran the model.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from modelgeometry.linalg import ArrayLike, _to_numpy, effective_rank


def attention_entropy(attn_weights: ArrayLike) -> np.ndarray:
    """Per-query-position entropy of attention probability distributions.

    Michel et al., 2019 ("Are Sixteen Heads Really Better than One?");
    Voita et al., 2019 ("Analyzing Multi-Head Self-Attention: Specialized
    Heads Do the Heavy Lifting, the Rest Can Be Pruned"). Used as a
    head-pruning / redundancy signal: low-entropy heads attend sharply to a
    few positions, high-entropy heads spread attention broadly.

    Args:
        attn_weights: attention probabilities of shape ``(..., seq_k)``,
            with the last axis summing to 1 (e.g.
            ``(batch, heads, seq_q, seq_k)``).

    Returns:
        Array of shape ``attn_weights.shape[:-1]`` with the Shannon entropy
        of each attention distribution.
    """
    weights = _to_numpy(attn_weights).astype(np.float64)
    # 0 * log(0) is mathematically 0 but is `nan` in floating point; replace
    # zero-probability entries with 1.0 before the log so they contribute 0.
    safe_weights = np.where(weights > 0, weights, 1.0)
    return -np.sum(weights * np.log(safe_weights), axis=-1)


def attention_effective_rank(attn_weights: ArrayLike) -> np.ndarray:
    """Effective rank of each attention matrix in a batch of attention weights.

    Applies `modelgeometry.linalg.effective_rank` to the trailing
    ``(seq_q, seq_k)`` matrix at every leading index, reusing the same
    Shannon-entropy-of-singular-values formulation used for weight matrices.

    Args:
        attn_weights: attention probabilities of shape ``(..., seq_q, seq_k)``.

    Returns:
        Array of shape ``attn_weights.shape[:-2]`` with one effective-rank
        value per attention matrix.
    """
    weights = _to_numpy(attn_weights)
    leading_shape = weights.shape[:-2]
    matrices = weights.reshape(-1, weights.shape[-2], weights.shape[-1])
    ranks = np.array([effective_rank(matrix) for matrix in matrices])
    return ranks.reshape(leading_shape)


def qkv_norm_stats(q: ArrayLike, k: ArrayLike, v: ArrayLike) -> Dict[str, Dict[str, float]]:
    """Summary of per-token (row) norms for captured Q/K/V activations.

    Args:
        q, k, v: activation tensors of shape ``(..., dim)``, e.g.
            ``(batch, heads, seq, head_dim)`` captured from a forward pass.

    Returns:
        ``{"q": {"mean_norm", "std_norm", "max_norm"}, "k": {...}, "v": {...}}``
    """
    result: Dict[str, Dict[str, float]] = {}
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        norms = np.linalg.norm(_to_numpy(tensor), axis=-1)
        result[name] = {
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "max_norm": float(norms.max()),
        }
    return result