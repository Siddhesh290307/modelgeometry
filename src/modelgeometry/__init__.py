"""modelgeometry: architecture-agnostic inspection of weight-space geometry,
attention geometry, and curvature (Fisher / K-FAC) for trained PyTorch models.
"""

from modelgeometry.adapters import (
    FusedQKVAdapter,
    ModelAdapter,
    QKVWeights,
    SplitQKVAdapter,
    resolve_adapter,
)
from modelgeometry.attention import attention_effective_rank, attention_entropy, qkv_norm_stats
from modelgeometry.curvature import curvature_prediction_check
from modelgeometry.fisher import diagonal_fisher, fisher_layer_summary
from modelgeometry.hooks import HookRegistry, capture_attention_weights
from modelgeometry.kfac import kfac_factors, kfac_offdiagonal_energy
from modelgeometry.regularizers import EWCPenalty, KFACPenalty, L2Penalty, SynapticIntelligencePenalty
from modelgeometry.tracking import GeometryTracker, compare_checkpoints
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

__all__ = [
    "ModelAdapter",
    "FusedQKVAdapter",
    "SplitQKVAdapter",
    "QKVWeights",
    "resolve_adapter",
    "HookRegistry",
    "capture_attention_weights",
    "effective_rank",
    "effective_rank_from_spectrum",
    "spectral_norm",
    "frobenius_norm",
    "row_cosine_similarity",
    "nullspace_projection",
    "participation_ratio",
    "distributional_distance",
    "attention_entropy",
    "attention_effective_rank",
    "qkv_norm_stats",
    "diagonal_fisher",
    "fisher_layer_summary",
    "kfac_factors",
    "kfac_offdiagonal_energy",
    "curvature_prediction_check",
    "L2Penalty",
    "EWCPenalty",
    "SynapticIntelligencePenalty",
    "KFACPenalty",
    "GeometryTracker",
    "compare_checkpoints",
]
