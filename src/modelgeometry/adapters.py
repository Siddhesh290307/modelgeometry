"""Model-architecture adapters that normalize attention-weight access across naming conventions.

Every public modelgeometry function that touches a model's parameters accepts a
:class:`ModelAdapter` (or resolves one via :func:`resolve_adapter`) rather than
hardcoding a specific model's attribute path (e.g. it never assumes
``transformer.h[i].attn.c_attn`` is the only valid layout). This is what lets
the same metric run unmodified on GPT-2-style, BERT-style, LLaMA-style, and
ViT-style architectures, as well as arbitrary user-defined models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from torch import Tensor, nn

try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except ImportError:  # transformers not installed; fused-Conv1D detection just won't fire
    _HFConv1D = None


# Ordered, documented list of common transformer block container paths. This is
# not an assumption that any one of these is "the" convention — resolve_adapter
# tries each in turn and falls back to raising with instructions to pass
# `layer_path=` explicitly for anything that doesn't match.
_LAYER_CONTAINER_PATHS = (
    "transformer.h",  # GPT-2 style, task-head-wrapped (e.g. GPT2LMHeadModel)
    "h",  # GPT-2 style, bare base model (e.g. GPT2Model)
    "model.layers",  # LLaMA / GPT-NeoX style
    "encoder.layer",  # BERT / HF ViT style
    "blocks",  # timm ViT style
    "vit.encoder.layer",  # HF ViTModel wrapped in a task head
)

_ATTN_NAMES = ("attn", "attention", "self_attn", "self_attention")

_FUSED_QKV_NAMES = ("c_attn", "qkv")
_SPLIT_Q_NAMES = ("q_proj", "query")
_SPLIT_K_NAMES = ("k_proj", "key")
_SPLIT_V_NAMES = ("v_proj", "value")


@dataclass
class QKVWeights:
    """Q, K, V projection weight matrices, each as an (out_features, in_features) matrix.

    Regardless of whether the source model uses a fused QKV projection or three
    separate projections, :meth:`ModelAdapter.qkv_weights` normalizes to this
    shape so every downstream metric only ever deals with three separate
    matrices.
    """

    q: Tensor
    k: Tensor
    v: Tensor


def _resolve_path(obj, dotted_path: str):
    for attr in dotted_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _is_indexable_sequence(obj) -> bool:
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes))


def _detect_attr_name(obj, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if hasattr(obj, name):
            return name
    return None


def _find_layer_container(model: nn.Module, layer_path: Optional[str]) -> Sequence[nn.Module]:
    if layer_path is not None:
        return _resolve_path(model, layer_path)
    for path in _LAYER_CONTAINER_PATHS:
        try:
            candidate = _resolve_path(model, path)
        except AttributeError:
            continue
        if _is_indexable_sequence(candidate) and len(candidate) > 0:
            return candidate
    raise ValueError(
        "Could not auto-detect the transformer block container on this model. "
        f"Tried {_LAYER_CONTAINER_PATHS}. Pass `layer_path=` explicitly, e.g. "
        "resolve_adapter(model, layer_path='model.decoder.layers')."
    )


def _linear_like_weight(module: nn.Module) -> Tensor:
    """Return a projection module's weight as an (out_features, in_features) matrix.

    ``nn.Linear`` already stores weights this way. HuggingFace's GPT-2 ``Conv1D``
    stores the transpose (``in_features, out_features``), so it's normalized here.
    """
    if _HFConv1D is not None and isinstance(module, _HFConv1D):
        return module.weight.t()
    return module.weight


class ModelAdapter(ABC):
    """Uniform interface onto a transformer's attention blocks.

    Construct via :func:`resolve_adapter` (auto-detection with explicit
    overrides) rather than instantiating subclasses directly.
    """

    def __init__(self, model: nn.Module, layers: Sequence[nn.Module], attn_name: str):
        self.model = model
        self._layers = list(layers)
        self._attn_name = attn_name

    def layers(self) -> Sequence[nn.Module]:
        return self._layers

    def num_layers(self) -> int:
        return len(self._layers)

    def attention_module(self, layer_idx: int) -> nn.Module:
        return getattr(self._layers[layer_idx], self._attn_name)

    @abstractmethod
    def qkv_weights(self, layer_idx: int) -> QKVWeights:
        """Return this layer's Q/K/V projection weights, split into three matrices."""

    @abstractmethod
    def qkv_modules(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Return the underlying hookable projection module(s) for this layer's QKV.

        Unlike `qkv_weights` (which always normalizes to three separate
        matrices regardless of source convention), this exposes whatever
        module(s) actually execute in the forward pass, keyed by role — a
        fused adapter returns a single ``{"qkv": module}`` entry, a split
        adapter returns three: ``{"q": ..., "k": ..., "v": ...}``. Used by
        anything that needs to register real forward/backward hooks (e.g.
        K-FAC's activation/gradient capture), since hooks can't be placed on
        a weight slice that isn't itself a module.
        """

    def hidden_size(self) -> int:
        return self.qkv_weights(0).q.shape[0]

    def num_heads(self, layer_idx: int = 0) -> int:
        attn = self.attention_module(layer_idx)
        config = getattr(self.model, "config", None)
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            if config is not None and hasattr(config, attr):
                return getattr(config, attr)
        for attr in ("num_heads", "n_head", "num_attention_heads"):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        raise ValueError(
            "Could not determine num_heads: no `model.config.num_attention_heads` "
            "(or `n_head`) and no matching attribute on the attention module. Pass "
            "num_heads explicitly to the calling metric function."
        )

    def head_dim(self, layer_idx: int = 0) -> int:
        attn = self.attention_module(layer_idx)
        for attr in ("head_dim", "attention_head_size"):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        hidden = self.qkv_weights(layer_idx).q.shape[0]
        num_heads = self.num_heads(layer_idx)
        if hidden % num_heads != 0:
            raise ValueError(
                f"Q projection output dim ({hidden}) is not divisible by num_heads "
                f"({num_heads}); pass head_dim explicitly."
            )
        return hidden // num_heads


class FusedQKVAdapter(ModelAdapter):
    """Adapter for a single projection producing a concatenated ``[q; k; v]`` output.

    Covers GPT-2-style (``c_attn``, a ``Conv1D``) and timm-ViT-style (``qkv``,
    an ``nn.Linear``) attention blocks.
    """

    def __init__(self, model: nn.Module, layers: Sequence[nn.Module], attn_name: str, qkv_name: str):
        super().__init__(model, layers, attn_name)
        self._qkv_name = qkv_name

    def qkv_weights(self, layer_idx: int) -> QKVWeights:
        attn = self.attention_module(layer_idx)
        module = getattr(attn, self._qkv_name)
        weight = _linear_like_weight(module)
        out_features = weight.shape[0]
        if out_features % 3 != 0:
            raise ValueError(
                f"Fused QKV projection '{self._qkv_name}' has output dim {out_features}, "
                "which is not divisible by 3; cannot split into q/k/v."
            )
        third = out_features // 3
        return QKVWeights(q=weight[:third], k=weight[third : 2 * third], v=weight[2 * third :])

    def qkv_modules(self, layer_idx: int) -> Dict[str, nn.Module]:
        attn = self.attention_module(layer_idx)
        return {"qkv": getattr(attn, self._qkv_name)}


class SplitQKVAdapter(ModelAdapter):
    """Adapter for three separate Q/K/V projection modules.

    Covers BERT-style (``query``/``key``/``value``) and LLaMA/GPT-NeoX-style
    (``q_proj``/``k_proj``/``v_proj``) attention blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[nn.Module],
        attn_name: str,
        q_name: str,
        k_name: str,
        v_name: str,
    ):
        super().__init__(model, layers, attn_name)
        self._q_name, self._k_name, self._v_name = q_name, k_name, v_name

    def qkv_weights(self, layer_idx: int) -> QKVWeights:
        attn = self.attention_module(layer_idx)
        q = _linear_like_weight(getattr(attn, self._q_name))
        k = _linear_like_weight(getattr(attn, self._k_name))
        v = _linear_like_weight(getattr(attn, self._v_name))
        return QKVWeights(q=q, k=k, v=v)

    def qkv_modules(self, layer_idx: int) -> Dict[str, nn.Module]:
        attn = self.attention_module(layer_idx)
        return {
            "q": getattr(attn, self._q_name),
            "k": getattr(attn, self._k_name),
            "v": getattr(attn, self._v_name),
        }


def resolve_adapter(
    model: nn.Module,
    layer_path: Optional[str] = None,
    attn_name: Optional[str] = None,
    qkv_names: Optional[Sequence[str]] = None,
) -> ModelAdapter:
    """Auto-detect and construct the right :class:`ModelAdapter` for ``model``.

    Tries a documented list of common layer-container paths and attention
    submodule names, then inspects the attention module for either a fused QKV
    projection or three separate Q/K/V projections. Any step can be overridden
    explicitly for models that don't match the common conventions —
    auto-detection raises rather than guessing past what it can confirm.

    Args:
        model: Any ``nn.Module`` containing one or more transformer blocks.
        layer_path: Dotted attribute path to the block container (e.g.
            ``"model.decoder.layers"``). Auto-detected if omitted.
        attn_name: Attribute name of the attention submodule on each block
            (e.g. ``"self_attn"``). Auto-detected if omitted.
        qkv_names: ``("qkv_attr",)`` for a fused projection, or
            ``("q_attr", "k_attr", "v_attr")`` for separate projections.
            Auto-detected if omitted.
    """
    layers = _find_layer_container(model, layer_path)
    first_layer = layers[0]

    resolved_attn_name = attn_name or _detect_attr_name(first_layer, _ATTN_NAMES)
    if resolved_attn_name is None:
        raise ValueError(
            f"Could not auto-detect the attention submodule on layer {type(first_layer).__name__}. "
            f"Tried {_ATTN_NAMES}. Pass `attn_name=` explicitly."
        )
    first_attn = getattr(first_layer, resolved_attn_name)

    if qkv_names is not None:
        if len(qkv_names) == 1:
            return FusedQKVAdapter(model, layers, resolved_attn_name, qkv_names[0])
        if len(qkv_names) == 3:
            return SplitQKVAdapter(model, layers, resolved_attn_name, *qkv_names)
        raise ValueError("qkv_names must have length 1 (fused) or 3 (split q/k/v).")

    fused_name = _detect_attr_name(first_attn, _FUSED_QKV_NAMES)
    if fused_name is not None:
        return FusedQKVAdapter(model, layers, resolved_attn_name, fused_name)

    q_name = _detect_attr_name(first_attn, _SPLIT_Q_NAMES)
    k_name = _detect_attr_name(first_attn, _SPLIT_K_NAMES)
    v_name = _detect_attr_name(first_attn, _SPLIT_V_NAMES)
    if q_name and k_name and v_name:
        return SplitQKVAdapter(model, layers, resolved_attn_name, q_name, k_name, v_name)

    raise ValueError(
        f"Could not auto-detect a fused or split QKV projection on attention module "
        f"{type(first_attn).__name__}. Tried fused names {_FUSED_QKV_NAMES} and split "
        f"names {_SPLIT_Q_NAMES}/{_SPLIT_K_NAMES}/{_SPLIT_V_NAMES}. Pass `qkv_names=` "
        "explicitly, e.g. qkv_names=('in_proj_weight',) or "
        "qkv_names=('q_proj', 'k_proj', 'v_proj')."
    )
