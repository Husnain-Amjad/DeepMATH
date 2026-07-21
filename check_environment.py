"""
Flexible environment check: reports what's already installed and only tells
you what's ACTUALLY missing or too old - never proposes reinstalling something
that's already present and adequate. This matters because many training
environments (vendor Docker images, Kaggle/Colab, cluster modules) come with
torch/vllm/etc. already installed and pinned to a version that works with that
specific hardware/driver stack; blindly running `pip install -r requirements.txt`
in those environments can downgrade or break a working setup.

Usage:
  python check_environment.py            # report only, installs nothing
  python check_environment.py --install-missing   # also pip-installs whatever
                                                    # is genuinely absent (never
                                                    # touches already-installed
                                                    # packages, even if older
                                                    # than the suggested minimum)
"""

import argparse
import importlib
import subprocess
import sys

try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:  # Python <3.8 fallback, unlikely but harmless to keep
    from importlib_metadata import version as pkg_version, PackageNotFoundError


# (import name, pip package name, minimum version or None, install hint)
CORE_PACKAGES = [
    ("transformers", "transformers", "4.46", None),
    ("accelerate", "accelerate", "0.34", None),
    ("peft", "peft", "0.13", None),
    ("trl", "trl", "0.12", None),
    ("datasets", "datasets", "2.20", None),
    ("sympy", "sympy", "1.12", None),
    ("matplotlib", "matplotlib", "3.7", None),
    ("huggingface_hub", "huggingface_hub", "0.25", None),
]

# Backend-specific: checked, but never auto-installed even with --install-missing,
# since the RIGHT install command depends on your hardware (see hardware_utils.py)
BACKEND_SPECIFIC = [
    ("torch", "torch", None,
     "CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
     "         ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"),
    ("vllm", "vllm", None,
     "CUDA: pip install vllm\n"
     "         ROCm: needs the ROCm-specific build - see ROCM_TUTORIAL.md Step 3. "
     "Not required - the pipeline falls back to HF-generate automatically if absent."),
    ("bitsandbytes", "bitsandbytes", None,
     "CUDA only, optional (only needed for --use_bnb 4-bit QLoRA): pip install bitsandbytes\n"
     "         ROCm: skip - unsupported, sft_train.py auto-disables --use_bnb on ROCm anyway."),
]


def _installed_version(import_name):
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return None
    try:
        return pkg_version(import_name)
    except PackageNotFoundError:
        return getattr(mod, "__version__", "unknown (installed, version string unavailable)")


def _version_ok(installed, minimum):
    if minimum is None or installed in (None, "unknown (installed, version string unavailable)"):
        return True
    try:
        from packaging.version import Version
        return Version(installed) >= Version(minimum)
    except Exception:
        return True  # can't compare - don't block on it, just report


def check_all(install_missing=False):
    print("=" * 70)
    print("CORE PACKAGES (version-flexible - installed and adequate = left alone)")
    print("=" * 70)
    missing = []
    for import_name, pip_name, minimum, _ in CORE_PACKAGES:
        installed = _installed_version(import_name)
        if installed is None:
            print(f"  [MISSING]  {pip_name}")
            missing.append(pip_name)
        elif not _version_ok(installed, minimum):
            print(f"  [OLD?]     {pip_name}=={installed} (suggested >= {minimum}) - "
                  f"left as-is; upgrade manually if you hit a real incompatibility")
        else:
            print(f"  [OK]       {pip_name}=={installed}")

    print()
    print("=" * 70)
    print("BACKEND-SPECIFIC (never auto-installed - depends on your hardware)")
    print("=" * 70)
    for import_name, pip_name, minimum, hint in BACKEND_SPECIFIC:
        installed = _installed_version(import_name)
        if installed is None:
            print(f"  [MISSING]  {pip_name}\n             -> {hint}")
        else:
            print(f"  [OK]       {pip_name}=={installed}")

    print()
    try:
        from hardware_utils import detect_platform
        info = detect_platform()
        print("=" * 70)
        print("HARDWARE")
        print("=" * 70)
        print(f"  backend={info.backend}  gpu={info.gpu_name}  count={info.gpu_count}  "
              f"memory={info.gpu_memory_gb}GB")
        for note in info.notes:
            print(f"  NOTE: {note}")
    except Exception as e:
        print(f"[check_environment] could not run hardware detection yet ({e}) - "
              f"normal if torch itself is one of the MISSING packages above.")

    if install_missing and missing:
        print()
        print(f"[check_environment] installing {len(missing)} missing core package(s): {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", *missing])
    elif missing:
        print()
        print(f"[check_environment] {len(missing)} core package(s) missing. Re-run with "
              f"--install-missing to install them, or install manually.")

    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-missing", action="store_true")
    args = ap.parse_args()
    check_all(install_missing=args.install_missing)
