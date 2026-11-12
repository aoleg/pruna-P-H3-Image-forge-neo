"""Dropping a replaced H3 model without copying it back to system RAM first.

When another checkpoint is selected, Forge Neo's forge_model_reload clears the active model and calls
unload_all_models, which detaches every loaded model to its offload device: the H3 DiT (about 20 GiB) moves from
VRAM to system RAM, where the text encoder (about 25 GiB) still is, only to be freed right after. On a 50 GB machine
that copy alone ran out of memory. A replaced H3 model is discarded anyway, so its tensors are emptied first and the
detach has nothing to copy.

A settings change (Diffusion in Low Bits, the selected modules) makes Forge unload every model while H3 stays
active: the DiT and the text encoder, about 45 GiB, would all land in system RAM. Reloading H3 from disk instead goes
through the release above and costs a model load.
"""

import torch
import torch.nn as nn


def h3_module_types() -> tuple[type, ...]:
    from .image_vae import H3ImageVAE
    from .text_encoder import PH3ImageQwen3VL, Qwen3VL32B
    from .transformer import MiniMaxH3Model
    return (MiniMaxH3Model, Qwen3VL32B, PH3ImageQwen3VL, H3ImageVAE)


def is_h3_module(module, types: tuple[type, ...]) -> bool:
    return isinstance(module, nn.Module) and any(isinstance(m, types) for m in module.modules())


def empty_tensors(module: nn.Module) -> None:
    """Replace every parameter and buffer with an empty tensor, freeing the storage wherever it lives."""
    for m in module.modules():
        for name, tensor in list(m._parameters.items()):
            if tensor is not None:
                m._parameters[name] = nn.Parameter(torch.empty(0), requires_grad=False)
        for name, tensor in list(m._buffers.items()):
            if tensor is not None:
                m._buffers[name] = torch.empty(0)


def reload_instead_of_unload(need_global_unload: bool, active_model) -> bool:
    """Whether Forge's global unload with H3 active should become a reload from disk."""
    return bool(need_global_unload) and getattr(active_model, "is_h3", False)


def _finish_gpu_work() -> None:
    # Forge copies offloaded weights on a side stream; nothing may still read a tensor that is about to be freed
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def release_replaced(loaded_models, active_model, types: tuple[type, ...]) -> int:
    """Empty the H3 models Forge still lists as loaded once H3 is no longer the active model; returns how many."""
    if getattr(active_model, "is_h3", False):
        return 0
    replaced = []
    for loaded in list(loaded_models):
        patcher = getattr(loaded, "model", None)
        module = getattr(patcher, "model", None)
        if is_h3_module(module, types):
            replaced.append((patcher, module))
    if not replaced:
        return 0
    _finish_gpu_work()
    for patcher, module in replaced:
        # with --pin-shared-memory Forge registers the offloaded CPU weights with CUDA (cudaHostRegister) and finds
        # them again by name to unregister them on detach; once emptied it cannot, and the registered host memory
        # would be freed under CUDA, which breaks the CUDA context until Forge restarts. Unpin them first.
        unpin = getattr(patcher, "unpin_all_weights", None)
        if callable(unpin):
            unpin()
        # the original weights a LoRA patch kept would be copied back on detach
        backup = getattr(patcher, "backup", None)
        if isinstance(backup, dict):
            backup.clear()
        empty_tensors(module)
    return len(replaced)
