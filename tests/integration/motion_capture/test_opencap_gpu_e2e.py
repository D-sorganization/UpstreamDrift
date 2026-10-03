"""End-to-end test for OpenCap sidecar runner on GPU runner (#11406).

Acceptance requirement:
- End-to-end test with a short sample video runs on a GPU runner and skips elsewhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from motion_capture.opencap_ingest.launcher import (
    OpenCapLaunchConfig,
    OpenCapLauncher,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_gpu]


def _is_gpu_available() -> bool:
    """Check if an NVIDIA GPU is accessible via torch or nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    return shutil.which("nvidia-smi") is not None


def test_opencap_gpu_e2e(tmp_path: Path) -> None:
    """Run end-to-end test with OpenCap on a GPU runner, skipping elsewhere."""
    if not _is_gpu_available():
        pytest.skip("GPU runner required for OpenCap end-to-end execution")

    launcher = OpenCapLauncher()
    if not launcher.is_available():
        pytest.skip(
            f"opencap-core not available on this host: {launcher.get_install_hint()}"
        )

    # When running on a configured GPU runner with opencap-core:
    session_dir = tmp_path / "gpu_e2e_session"
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    cfg = OpenCapLaunchConfig(
        session_dir=session_dir,
        video_dir=video_dir,
        detector="hrnet",
    )
    result = launcher.launch(cfg)
    assert result.success is True
    assert result.session is not None
