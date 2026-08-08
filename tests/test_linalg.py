"""linalg.py tests: known-value/property checks on synthetic matrices, plus a
pass over real Q/K/V weight matrices pulled through two structurally
different adapters (fused-QKV and split-QKV toy models) to make sure nothing
here silently assumes one architecture's weight-shape conventions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modelgeometry.adapters import resolve_adapter
from modelgeometry.linalg import (
    distributional_distance,
    effective_rank,
    effective_rank_from_spectrum,
    frobenius_norm,
    nullspace_projection,
    participation_ratio,
    row_cosine_similarity,
    spectral_norm,
)

# ---- effective_rank ----------------------------------------------------


def test_effective_rank_identity_is_full_rank():
    assert effective_rank(np.eye(5)) == pytest.approx(5.0, abs=1e-6)


def test_effective_rank_rank_one_matrix():
    v = np.array([[1.0], [2.0], [3.0]])
    rank_one = v @ v.T
    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_accepts_torch_tensor():
    assert effective_rank(torch.eye(4)) == pytest.approx(4.0, abs=1e-6)


def test_effective_rank_from_spectrum_uniform_and_one_hot():
    assert effective_rank_from_spectrum(np.ones(6)) == pytest.approx(6.0, abs=1e-6)
    one_hot = np.array([0.0, 0.0, 5.0, 0.0])
    assert effective_rank_from_spectrum(one_hot) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_from_spectrum_rejects_negative_values():
    with pytest.raises(ValueError):
        effective_rank_from_spectrum(np.array([1.0, -1.0]))


# ---- spectral_norm / frobenius_norm -------------------------------------


def test_spectral_norm_diagonal_matrix():
    assert spectral_norm(np.diag([3.0, 2.0, 1.0])) == pytest.approx(3.0)


def test_frobenius_norm_identity():
    assert frobenius_norm(np.eye(4)) == pytest.approx(2.0)  # sqrt(4)


# ---- row_cosine_similarity ------------------------------------------------


def test_row_cosine_similarity_orthogonal_rows():
    sim = row_cosine_similarity(np.eye(3))
    np.testing.assert_allclose(sim, np.eye(3), atol=1e-10)


def test_row_cosine_similarity_duplicate_rows():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sim = row_cosine_similarity(matrix)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(0.0, abs=1e-10)


# ---- nullspace_projection -------------------------------------------------


def test_nullspace_projection_default_ones_direction():
    p = nullspace_projection(3)
    ones = np.ones(3)
    np.testing.assert_allclose(p @ ones, np.zeros(3), atol=1e-10)


def test_nullspace_projection_leaves_orthogonal_vector_unchanged():
    p = nullspace_projection(3)
    orthogonal = np.array([1.0, -1.0, 0.0])  # orthogonal to the all-ones direction
    np.testing.assert_allclose(p @ orthogonal, orthogonal, atol=1e-10)


def test_nullspace_projection_custom_direction():
    p = nullspace_projection(2, direction=np.array([1.0, 0.0]))
    np.testing.assert_allclose(p @ np.array([1.0, 0.0]), np.zeros(2), atol=1e-10)
    np.testing.assert_allclose(p @ np.array([0.0, 1.0]), np.array([0.0, 1.0]), atol=1e-10)


def test_nullspace_projection_rejects_zero_direction():
    with pytest.raises(ValueError):
        nullspace_projection(2, direction=np.zeros(2))


# ---- participation_ratio ---------------------------------------------------


def test_participation_ratio_uniform_spectrum():
    assert participation_ratio(np.ones(10)) == pytest.approx(10.0)


def test_participation_ratio_one_hot_spectrum():
    x = np.zeros(10)
    x[0] = 5.0
    assert participation_ratio(x) == pytest.approx(1.0)


def test_participation_ratio_rejects_negative_values():
    with pytest.raises(ValueError):
        participation_ratio(np.array([1.0, -1.0]))


# ---- distributional_distance -----------------------------------------------


def test_distributional_distance_identical_samples_is_zero():
    x = np.random.default_rng(0).normal(size=50)
    assert distributional_distance(x, x) == pytest.approx(0.0, abs=1e-10)


def test_distributional_distance_wasserstein_shifted_constants():
    a = np.full(20, 1.0)
    b = np.full(20, 4.0)
    assert distributional_distance(a, b, method="wasserstein") == pytest.approx(3.0)


def test_distributional_distance_ks_identical_is_zero():
    x = np.random.default_rng(1).normal(size=50)
    assert distributional_distance(x, x, method="ks") == pytest.approx(0.0)


def test_distributional_distance_unknown_method_raises():
    with pytest.raises(ValueError):
        distributional_distance(np.zeros(3), np.zeros(3), method="bogus")


# ---- against real weight matrices, across two structurally different adapters ----


@pytest.mark.parametrize("model_fixture", ["toy_fused_model", "toy_split_model"])
def test_linalg_functions_on_real_qkv_weights(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    adapter = resolve_adapter(model)
    qkv = adapter.qkv_weights(0)

    for weight in (qkv.q, qkv.k, qkv.v):
        rank = effective_rank(weight)
        assert 1.0 <= rank <= min(weight.shape) + 1e-6
        assert spectral_norm(weight) > 0
        assert frobenius_norm(weight) > 0
        sim = row_cosine_similarity(weight)
        assert sim.shape == (weight.shape[0], weight.shape[0])
        np.testing.assert_allclose(np.diag(sim), np.ones(weight.shape[0]), atol=1e-5)

    distance = distributional_distance(qkv.q.detach().numpy().ravel(), qkv.k.detach().numpy().ravel())
    assert distance >= 0.0