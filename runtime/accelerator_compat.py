"""Minimal `torch.accelerator` namespace for the torch 2.5.1 H3 benchmark env.

transformers >= 5.x calls `torch.accelerator.current_accelerator()` from
`utils.import_utils.get_device_type()`. torch only gained that namespace in 2.6.
This shim delegates the whole surface to torch.cuda so the existing 2.5.1
runtime keeps working; it must be installed before anything imports transformers.
"""
import sys
import types

__all__ = ["install"]


def _build(torch):
    def current_accelerator(check_available=False):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def set_device_index(index):
        torch.cuda.set_device(index)

    module = types.ModuleType("torch.accelerator")
    module.current_accelerator = current_accelerator
    module.is_available = torch.cuda.is_available
    module.device_count = torch.cuda.device_count
    module.current_device_index = torch.cuda.current_device
    module.set_device_index = set_device_index
    module.set_device = set_device_index
    module.synchronize = torch.cuda.synchronize
    module.memory_allocated = torch.cuda.memory_allocated
    module.memory_reserved = torch.cuda.memory_reserved
    module.empty_cache = torch.cuda.empty_cache
    return module


def install():
    """Return True when the shim was installed, False when torch already has it."""
    try:
        import torch
    except ImportError:
        return False
    if hasattr(torch, "accelerator"):
        return False
    module = _build(torch)
    sys.modules["torch.accelerator"] = module
    torch.accelerator = module
    return True
