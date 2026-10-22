"""Pick the array backend: CuPy on an NVIDIA GPU when one is present, NumPy otherwise.

FinAgent is designed to run without a GPU. When one is available the Monte
Carlo risk engine moves to it with no code change - the same functions receive
``cupy`` instead of ``numpy`` as their array module.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def array_backend() -> tuple[object, str]:
    try:
        import cupy as cp  # optional: pip install "finagent[gpu]"

        if cp.cuda.runtime.getDeviceCount() > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            return cp, f"GPU (CuPy, {name})"
    except Exception:
        pass
    return np, "CPU (NumPy)"
