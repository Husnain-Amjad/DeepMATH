"""
Shared determinism helpers. Import and call set_all_seeds() at the top of any
script (sft_train.py, grpo_train.py, run_eval.py, run_augmentation.py) before
building the model/dataset.

Honest scope of what this can and can't guarantee:
  - CAN make repeated runs on the SAME machine/library-version stack close to
    reproducible: same weight init, same dropout masks, same data shuffle order.
  - CANNOT guarantee bit-identical results across different GPUs, driver
    versions, or CUDA/ROCm library versions - floating-point reduction order
    in matmul/attention kernels differs at that level regardless of seed, and
    that difference compounds over training steps. The realistic target
    across environments is statistically consistent results (differences
    within a few standard errors, i.e. within eval-noise), not bit-identical
    numbers. See README's "Reproducibility" section.
"""

import os
import random


def set_all_seeds(seed: int, strict_deterministic: bool = False):
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if strict_deterministic:
            # Opt-in only: torch.use_deterministic_algorithms can raise on ops
            # without a deterministic kernel, and forcing deterministic cuBLAS
            # algorithms measurably slows training. Off by default so seeding
            # doesn't silently cost you throughput unless you ask for it.
            is_rocm = bool(getattr(torch.version, "hip", None))
            if not is_rocm:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as e:
                print(f"[determinism] use_deterministic_algorithms unavailable/partial: {e}")
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass

    print(f"[determinism] seeded random/numpy/torch/transformers with seed={seed}"
          f"{' (strict deterministic algorithms requested)' if strict_deterministic else ''}")
