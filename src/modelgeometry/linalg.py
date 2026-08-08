"""Weight-space and general linear-algebra primitives shared across modelgeometry.

These functions operate on plain matrices/vectors (numpy arrays or torch
tensors) and require no model, dataloader, or forward pass — they work on any
pretrained checkpoint's weights directly. Per the package's separation of
computation from reporting, every function here returns plain Python floats
or numpy arrays, never a framework-specific tensor type.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from scipy import stats

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _singular_values(weight_matrix: ArrayLike) -> np.ndarray:
    return np.linalg.svd(_to_numpy(weight_matrix), compute_uv=False)


def effective_rank_from_spectrum(values: ArrayLike) -> float:
    """Effective rank of an arbitrary nonnegative spectrum.

    Standard information-theoretic formulation: exp(Shannon entropy of
    ``values`` normalized to a probability distribution). Ranges from 1 (all
    mass on one component) to ``len(values)`` (a perfectly flat spectrum).

    This is the primitive `effective_rank` applies to a matrix's singular
    values; it's exposed separately because other spectra worth measuring
    this way (e.g. a per-layer Fisher-information mass distribution) aren't
    themselves singular values of a matrix.
    """
    x = _to_numpy(values).astype(np.float64).ravel()
    if np.any(x < 0):
        raise ValueError("effective_rank_from_spectrum expects a nonnegative spectrum.")
    total = x.sum()
    if total <= 0:
        return 0.0
    probabilities = x / total
    nonzero = probabilities[probabilities > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(np.exp(entropy))


def effective_rank(weight_matrix: ArrayLike) -> float:
    """Effective rank of a matrix's singular value spectrum.

    Standard information-theoretic formulation: exp(Shannon entropy of the
    singular values normalized to a probability distribution). Ranges from 1
    (rank-1 matrix) to min(matrix.shape) (a matrix with a perfectly flat
    spectrum).
    """
    return effective_rank_from_spectrum(_singular_values(weight_matrix))


def spectral_norm(weight_matrix: ArrayLike) -> float:
    """Largest singular value of a matrix."""
    return float(_singular_values(weight_matrix)[0])


def frobenius_norm(weight_matrix: ArrayLike) -> float:
    """Frobenius norm of a matrix."""
    return float(np.linalg.norm(_to_numpy(weight_matrix)))


def row_cosine_similarity(weight_matrix: ArrayLike) -> np.ndarray:
    """Pairwise cosine similarity between all rows of a matrix.

    Returns an ``(n_rows, n_rows)`` matrix (diagonal is 1 by construction).
    Useful for measuring redundancy across a projection matrix's rows (e.g.
    key/query/value projection rows) — the caller decides what summary (mean
    off-diagonal, max, ...) is meaningful for their use case.
    """
    matrix = _to_numpy(weight_matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def nullspace_projection(dim: int, direction: Optional[ArrayLike] = None) -> np.ndarray:
    """Projector onto the subspace orthogonal to a supplied direction vector.

    Defaults to the all-ones direction, but any nonzero vector may be
    supplied — a generic linear-algebra utility, not tied to any specific
    normalization scheme. Returns the ``(dim, dim)`` projection matrix
    ``P = I - vv^T / (v^T v)``.
    """
    if direction is None:
        v = np.ones(dim, dtype=np.float64)
    else:
        v = _to_numpy(direction).astype(np.float64)
        if v.shape != (dim,):
            raise ValueError(f"direction must have shape ({dim},), got {v.shape}")
    denom = v @ v
    if denom == 0:
        raise ValueError("direction must be a nonzero vector.")
    return np.eye(dim) - np.outer(v, v) / denom


def participation_ratio(values: ArrayLike) -> float:
    """Participation ratio of a nonnegative spectrum (eigenvalues, activation magnitudes, ...).

    Standard information-theoretic effective-dimensionality measure:
    ``PR = (sum(x))^2 / sum(x^2)``. Ranges from 1 (all mass on one component)
    to ``len(x)`` (a uniform spectrum). This is a distinct formulation from
    `effective_rank`'s Shannon-entropy approach and is kept as a separate
    primitive rather than folded into it.
    """
    x = _to_numpy(values).astype(np.float64)
    if np.any(x < 0):
        raise ValueError("participation_ratio expects a nonnegative spectrum (eigenvalues/magnitudes).")
    denom = np.sum(x**2)
    if denom == 0:
        return 0.0
    return float(np.sum(x) ** 2 / denom)


def distributional_distance(sample_a: ArrayLike, sample_b: ArrayLike, method: str = "wasserstein") -> float:
    """Generic 1-D distance between two sample distributions.

    The caller supplies whatever two 1-D arrays they want compared (two
    layers, two checkpoints, two heads, two training epochs, ...) — this
    function never prescribes which pair is meaningful.

    Args:
        method: ``"wasserstein"`` (earth mover's distance, default) or
            ``"ks"`` (two-sample Kolmogorov-Smirnov statistic).
    """
    a = _to_numpy(sample_a).ravel()
    b = _to_numpy(sample_b).ravel()
    if method == "wasserstein":
        return float(stats.wasserstein_distance(a, b))
    if method == "ks":
        return float(stats.ks_2samp(a, b).statistic)
    raise ValueError(f"Unknown method '{method}'; expected 'wasserstein' or 'ks'.")