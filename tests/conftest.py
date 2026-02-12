"""Shared test fixtures and helpers."""

import unittest


def get_gpu_vram_gb() -> float:
    """Return available GPU VRAM in GB, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


_vram_gb = get_gpu_vram_gb()

skip_low_vram = unittest.skipIf(
    _vram_gb < 4.0,
    f"GPU VRAM too low ({_vram_gb:.1f} GB < 4 GB required for SAM3)",
)
