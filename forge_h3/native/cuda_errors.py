"""Showing the CUDA error that Forge Neo reports as out of memory.

Forge Neo's backend.memory_management.is_oom counts every torch.AcceleratorError as out of memory, and its main
thread then prints only "Encountered Out of Memory during Sampling" with no traceback. A real CUDA fault (an illegal
memory access, a failed host registration) is an AcceleratorError too. It breaks the CUDA context: every later call
fails at once with VRAM free, until Forge restarts, and the console says nothing about the cause. The wrapper below
prints such an error once, with its traceback, and leaves Forge's own handling as it is.
"""

import traceback

import torch

NOTE = ("[MiniMax H3] Forge reports the error below as out of memory, but it is a CUDA error, not a lack of VRAM. "
        "It leaves the GPU unusable until Forge restarts. Please include this traceback in a bug report.")


def is_disguised_cuda_error(error: BaseException) -> bool:
    """An AcceleratorError that is not an out-of-memory error, which Forge would still call one."""
    accelerator = getattr(torch, "AcceleratorError", None)
    oom = getattr(torch, "OutOfMemoryError", None)
    if accelerator is None or not isinstance(error, accelerator):
        return False
    if oom is not None and isinstance(error, oom):
        return False
    return "out of memory" not in str(error).lower()


def report_disguised(is_oom):
    """Wrap Forge's is_oom so a CUDA error it takes for out of memory is printed once with its traceback."""
    def wrapped(error):
        result = is_oom(error)
        if result and is_disguised_cuda_error(error) and not getattr(error, "_h3_reported", False):
            try:
                error._h3_reported = True
            except AttributeError:
                pass
            print(NOTE)
            traceback.print_exception(type(error), error, error.__traceback__)
        return result

    wrapped.__wrapped__ = is_oom
    return wrapped
