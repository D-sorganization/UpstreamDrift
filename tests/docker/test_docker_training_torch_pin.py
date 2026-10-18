"""Docker training-stage PyTorch pin contract (UpstreamDrift #12938)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.shared.python.feature_registry import get_feature
from src.shared.python.feature_registry.training_torch_pin import (
    TORCH_CUDA_VERSION,
    TORCH_CUDA_WHEEL_INDEX,
    torch_cuda_pip_install_command,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "Dockerfile"
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _training_stage_block() -> str:
    content = _DOCKERFILE.read_text(encoding="utf-8")
    start = content.index("FROM runtime AS training")
    next_from = content.find("\nFROM ", start + 1)
    if next_from == -1:
        return content[start:]
    return content[start:next_from]


def test_training_compose_service_requires_profile() -> None:
    """Explorer/profile builds must not pull the GPU training stage by default."""
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    training = compose["services"]["training"]
    profiles = training.get("profiles")
    assert profiles, "training service must declare compose profiles"
    assert "training" in profiles


def test_feature_registry_torch_cuda_pin_matches_shared_constants() -> None:
    feature = get_feature("torch-cuda")
    assert feature.install_command == torch_cuda_pip_install_command()
    assert f"torch=={TORCH_CUDA_VERSION}" in feature.install_command
    assert TORCH_CUDA_WHEEL_INDEX in feature.install_command
    assert "cu124" not in feature.install_command


def test_dockerfile_training_stage_uses_shared_torch_pin() -> None:
    block = _training_stage_block()
    assert f'"torch=={TORCH_CUDA_VERSION}"' in block
    assert TORCH_CUDA_WHEEL_INDEX in block
    assert "cu124" not in block


def test_pinned_torch_is_available_on_cuda_wheel_index() -> None:
    """Fail closed if PyTorch drops the pinned wheel from the CUDA index."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_CUDA_VERSION}",
            "--index-url",
            TORCH_CUDA_WHEEL_INDEX,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Pinned torch must resolve on the CUDA wheel index; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_torch_cuda_display_name_matches_index() -> None:
    feature = get_feature("torch-cuda")
    assert re.search(r"CUDA 12\.\d+", feature.display_name)
    assert "12.4" not in feature.display_name
