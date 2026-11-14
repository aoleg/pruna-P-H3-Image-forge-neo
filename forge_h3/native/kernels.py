"""The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests)."""

import contextlib

import torch

try:
    from backend.quant_ops import ck
except ImportError:  # outside Forge Neo: the same pinned package, eager backend on CPU
    import comfy_kitchen as ck

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
    from backend.memory_management import get_free_memory as _forge_free_memory
    from backend.operations import main_stream_worker, weights_manual_cast
except ImportError:
    _forge_attention = None
    _forge_cast_to = None
    _forge_free_memory = None
    main_stream_worker = None
    weights_manual_cast = None

try:
    # operations first: operations_mixed_precision is half of it and imports it back
    import backend.operations  # noqa: F401
    from backend.operations_mixed_precision import (
        linear_input_act as _forge_linear_input_act,
    )
except ImportError:  # outside Forge Neo, or a Forge Neo without fused input activations
    _forge_linear_input_act = None

__all__ = ["attention", "cast_to", "ck", "free_memory", "fused", "linear_swiglu", "modulated_norm", "report",
           "swiglu_fold", "weight_and_bias", "why_not"]

# The fused path of the DiT block, as Forge Neo's FIBO 2 and MOSS-Image ports run it: the modulated RMSNorm as one
# rms_adaln pass (the norm weight folded into the scale), and SwiGLU folded into fc2's INT8 ConvRot GEMM through
# Forge's linear_input_act. The q/k norms with RoPE and the gated residuals were fused already. It runs inside Forge,
# on CUDA, during inference, with a bf16 or float32 compute dtype; anything else keeps the eager path, which stays as
# it was. MiniMaxH3Model._forward decides once per forward and hands the flag to the blocks.
FUSED_DTYPES = (torch.bfloat16, torch.float32)
FUSED_SETTING = "h3_fused_kernels"


def _setting_on() -> bool:
    try:
        from modules import shared
        return bool(getattr(shared.opts, FUSED_SETTING, True))
    except (ImportError, AttributeError):
        return True


def why_not(x, dtype) -> str | None:
    """Why a forward on x, computing in dtype, takes the eager path; None when it takes the fused one."""
    if _forge_attention is None:
        return "outside Forge"
    if not hasattr(ck, "rms_adaln"):
        return "comfy-kitchen has no rms_adaln"
    if not x.is_cuda:
        return "not on CUDA"
    if torch.is_grad_enabled():
        return "gradients are enabled"
    if dtype not in FUSED_DTYPES:
        return f"{str(dtype).removeprefix('torch.')} compute keeps the eager path"
    if not _setting_on():
        return "the [H3] Fused kernels setting is off"
    return None


def fused(x, dtype) -> bool:
    return why_not(x, dtype) is None


_reported = None


def report(model, x, dtype, is_fused) -> None:
    """One console line each time the DiT's path changes: fused or eager and why, and whether the SwiGLU fold engages
    on this model's weights. Quiet outside Forge."""
    global _reported
    if is_fused:
        fold = swiglu_fold(model.blocks[0].mlp.fc2)
        state = ("on", str(dtype).removeprefix("torch."), fold)
        line = (f"fused comfy-kitchen kernels: on ({state[1]}), SwiGLU folded into the INT8 ConvRot GEMM: "
                f"{'yes' if fold else 'no (not an INT8 ConvRot weight)'}")
    else:
        state = ("off", why_not(x, dtype))
        line = f"fused comfy-kitchen kernels: off ({state[1]})"
    if state != _reported and _forge_attention is not None:
        print(f"[MiniMax H3] {line}")
    _reported = state


def swiglu_fold(linear) -> bool:
    """Whether Forge folds SwiGLU into this linear's GEMM: an INT8 ConvRot weight under linear_input_act. Other
    weights (bf16, GGUF, a weight under an on-the-fly LoRA) run SwiGLU eagerly inside the same call."""
    weight = getattr(linear, "weight", None)
    params = getattr(weight, "_params", None)
    return (_forge_linear_input_act is not None and params is not None
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout" and getattr(params, "convrot", False)
            and len(getattr(linear, "weight_function", ())) == 0)


def _low_rank_active(linear) -> bool:
    # native/lora.py: a plain LoRA on an INT8 layer runs as a low-rank term inside the layer's own forward, which
    # linear_input_act would skip
    hook = linear.__dict__.get("forward")
    return hook is not None and getattr(hook, "active", True)


def linear_swiglu(linear, x):
    """linear(silu(gate) * up) for x = [gate | up] (fc1's output, the order Forge's swiglu takes).

    Inside Forge the call is ComfyUI's: linear_input_act folds SwiGLU into comfy-kitchen's INT8 ConvRot activation
    quantization when the weight allows (swiglu_fold), and runs it eagerly before the linear otherwise. A layer under
    the extension's low-rank LoRA term, and anything outside Forge, takes the plain chunk, silu, mul and linear.
    """
    if _forge_linear_input_act is not None and not _low_rank_active(linear):
        return _forge_linear_input_act(linear, x, "swiglu")
    gate, up = x.chunk(2, dim=-1)
    return linear(torch.nn.functional.silu(gate).mul_(up))


def _norm_weight(norm, device) -> torch.Tensor:
    """A norm's weight in float32 with Forge's on-the-fly LoRA patches applied (the module's weight_function)."""
    weight = cast_to(norm.weight, torch.float32, device)
    for patch in getattr(norm, "weight_function", ()) or ():
        weight = patch(weight)
    return weight


def modulated_norm(norm, x, shift, scale, segments):
    """norm(x) * (1 + scale[row]) + shift[row] per segment, as one rms_adaln pass over the stream plus one per segment
    of another modulation row; segments [(start, stop, row)] cover x. The norm weight folds into the scale:
    rmsnorm(x) * w * (1 + s) = rmsnorm(x) * (1 + (w * (1 + s) - 1)), summed in float32 and rounded once."""
    weight = _norm_weight(norm, x.device)
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    folded = {}

    def row_terms(row):
        if row not in folded:
            s = scale[row].to(device=x.device, dtype=torch.float32)
            folded[row] = ((weight * (1.0 + s) - 1.0).to(x.dtype), shift[row].to(device=x.device, dtype=x.dtype))
        return folded[row]

    # the picture's segment is nearly all the rows: one pass over everything with its row, then the others rewritten
    base = max(segments, key=lambda seg: seg[1] - seg[0])[2]
    h = ck.rms_adaln(x, *row_terms(base), eps)
    for start, stop, row in segments:
        if row != base:
            h[start:stop] = ck.rms_adaln(x[start:stop], *row_terms(row), eps)
    return h

# what free_memory reports outside Forge: enough for one decoder tile at a time
_UNKNOWN_FREE_MEMORY = 512 * 2**20


def free_memory(device):
    if _forge_free_memory is not None:
        return _forge_free_memory(device)
    return _UNKNOWN_FREE_MEMORY


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def _select_attention():
    # Forge Neo's choice, --use-ck-attention included: comfy-kitchen 0.2.37, the minimum (compat.py), runs H3 with it
    if _forge_attention is None:
        return None
    return _forge_attention.attention_function


_attention = _select_attention()


def attention(q, k, v, heads, transformer_options=None):
    """q, k, v: [1, heads, S, head_dim] -> [1, S, heads * head_dim].

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension."""
    if _attention is not None:
        return _attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=transformer_options or {})
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])


@contextlib.contextmanager
def weight_and_bias(layer, ref):
    """A linear layer's weight and bias on ref's device and dtype, LoRA patches applied, as Forge's Linear does."""
    if weights_manual_cast is not None and hasattr(layer, "weight_function"):
        weight, bias, signal = weights_manual_cast(layer, ref)
        with main_stream_worker(weight, bias, signal):
            yield weight, bias
        return
    yield layer.weight.to(ref), layer.bias.to(ref)
