"""Context-managed forward/backward hook installer.

``HookRegistry`` captures activations, attention weights, or gradients from a
model without permanently mutating it — every hook it installs is removed on
``__exit__`` (or by calling :meth:`HookRegistry.remove_all`).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from torch import Tensor, nn

_ATTENTION_WEIGHTS_ATTR = "attn_weights"


class HookRegistry:
    """Context manager that installs forward/backward hooks and removes them on exit.

    Example::

        registry = HookRegistry()
        with registry:
            registry.capture_output("block0", model.transformer.h[0])
            model(input_ids)
        activations = registry.captured["block0"]
    """

    def __init__(self):
        self.captured: Dict[str, Tensor] = {}
        self._handles = []

    def __enter__(self) -> "HookRegistry":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_all()

    def remove_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def capture_output(self, name: str, module: nn.Module, transform: Optional[Callable] = None) -> None:
        """Store a module's forward output (or ``transform(output)``) under ``name``."""

        def hook(_module, _inputs, output):
            self.captured[name] = transform(output) if transform is not None else output

        self._handles.append(module.register_forward_hook(hook))

    def capture_input(self, name: str, module: nn.Module, transform: Optional[Callable] = None) -> None:
        """Store a module's forward input under ``name`` (e.g. K-FAC activation covariance)."""

        def hook(_module, inputs, _output):
            value = inputs[0] if len(inputs) == 1 else inputs
            self.captured[name] = transform(value) if transform is not None else value

        self._handles.append(module.register_forward_hook(hook))

    def capture_grad_output(self, name: str, module: nn.Module, transform: Optional[Callable] = None) -> None:
        """Store a module's output gradient under ``name`` during backward (e.g. K-FAC gradient covariance)."""

        def hook(_module, _grad_input, grad_output):
            value = grad_output[0] if len(grad_output) == 1 else grad_output
            self.captured[name] = transform(value) if transform is not None else value

        self._handles.append(module.register_full_backward_hook(hook))


def capture_attention_weights(registry: HookRegistry, name: str, attn_module: nn.Module) -> None:
    """Register a hook that captures ``attn_module``'s attention probabilities under ``name``.

    Tries two conventions, in order:

    1. HuggingFace-style: the module's forward output is a tuple whose second
       element is the attention-weights tensor, as returned when a model is
       called with ``output_attentions=True``. Note this requires the model
       to use its ``"eager"`` attention implementation — fused kernels
       (``sdpa``, flash-attention) never materialize this tensor, so
       construct/load the model with ``attn_implementation="eager"`` (or set
       ``model.config._attn_implementation = "eager"``) before capturing.
    2. Attribute-style: the module stores its most recently computed
       attention weights on ``self.attn_weights`` — the convention this
       package's own toy fixtures follow, useful for custom architectures.

    Raises ``ValueError`` (surfaced when the forward pass runs the hook) if
    neither convention yields a tensor, rather than silently capturing
    ``None`` or the wrong value.
    """

    def extract(output):
        if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], Tensor):
            return output[1]
        value = getattr(attn_module, _ATTENTION_WEIGHTS_ATTR, None)
        if isinstance(value, Tensor):
            return value
        raise ValueError(
            f"Could not extract attention weights from module {type(attn_module).__name__}. "
            "Expected either a forward-output tuple with attention weights as the second "
            "element (requires calling the model with output_attentions=True and an "
            "eager attention implementation), or a `self.attn_weights` attribute set "
            "during forward. For other conventions, use HookRegistry.capture_output "
            "directly with a custom `transform=`."
        )

    registry.capture_output(name, attn_module, transform=extract)
