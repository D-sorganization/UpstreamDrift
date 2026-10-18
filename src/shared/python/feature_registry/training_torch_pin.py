"""PyTorch CUDA wheel pin for the Docker training stage (issue #12938)."""

from __future__ import annotations

TORCH_CUDA_VERSION = "2.9.0"
TORCH_CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu128"


def torch_cuda_pip_install_command() -> str:
    """Return the documented pip one-liner for GPU PyTorch in training images."""
    if not TORCH_CUDA_VERSION.strip():
        raise ValueError("TORCH_CUDA_VERSION must not be empty")
    if not TORCH_CUDA_WHEEL_INDEX.startswith("https://"):
        raise ValueError("TORCH_CUDA_WHEEL_INDEX must be an https URL")
    return (
        f"pip install 'torch=={TORCH_CUDA_VERSION}' "
        f"--index-url {TORCH_CUDA_WHEEL_INDEX}"
    )
