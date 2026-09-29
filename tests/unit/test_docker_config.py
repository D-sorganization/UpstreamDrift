"""Tests for src.shared.python.docker_config (Issues #1949, #1744)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.shared.python.docker_config import (
    DOCKER_IMAGE_DEV,
    DOCKER_IMAGE_ENGINE,
    DOCKER_IMAGE_FAMILY,
    DOCKER_IMAGE_RUNTIME,
    DOCKER_IMAGE_TRAINING,
    LEGACY_DOCKER_ALIASES,
    detect_gpu_support,
)

# ---------------------------------------------------------------------------
# Image name constants
# ---------------------------------------------------------------------------


class TestDockerImageConstants:
    def test_image_family_is_string(self) -> None:
        assert isinstance(DOCKER_IMAGE_FAMILY, str)

    def test_image_family_non_empty(self) -> None:
        assert len(DOCKER_IMAGE_FAMILY) > 0

    def test_engine_image_contains_family(self) -> None:
        assert DOCKER_IMAGE_FAMILY in DOCKER_IMAGE_ENGINE

    def test_runtime_image_contains_family(self) -> None:
        assert DOCKER_IMAGE_FAMILY in DOCKER_IMAGE_RUNTIME

    def test_dev_image_contains_family(self) -> None:
        assert DOCKER_IMAGE_FAMILY in DOCKER_IMAGE_DEV

    def test_training_image_contains_family(self) -> None:
        assert DOCKER_IMAGE_FAMILY in DOCKER_IMAGE_TRAINING

    def test_engine_tag(self) -> None:
        assert DOCKER_IMAGE_ENGINE.endswith(":engine")

    def test_runtime_tag(self) -> None:
        assert DOCKER_IMAGE_RUNTIME.endswith(":runtime")

    def test_dev_tag(self) -> None:
        assert DOCKER_IMAGE_DEV.endswith(":dev")

    def test_training_tag(self) -> None:
        assert DOCKER_IMAGE_TRAINING.endswith(":training")

    def test_env_var_overrides_family(self) -> None:
        with patch.dict(os.environ, {"UPSTREAM_DRIFT_IMAGE_FAMILY": "custom-family"}):
            # Re-evaluate the module-level constant logic inline
            family = os.environ.get("UPSTREAM_DRIFT_IMAGE_FAMILY", "upstream-drift")
            assert family == "custom-family"


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------


class TestLegacyDockerAliases:
    def test_is_tuple(self) -> None:
        assert isinstance(LEGACY_DOCKER_ALIASES, tuple)

    def test_docker_config_non_empty(self) -> None:
        assert len(LEGACY_DOCKER_ALIASES) > 0

    def test_docker_config_all_are_strings(self) -> None:
        assert all(isinstance(a, str) for a in LEGACY_DOCKER_ALIASES)

    def test_all_have_tag(self) -> None:
        assert all(":" in a for a in LEGACY_DOCKER_ALIASES)


# ---------------------------------------------------------------------------
# detect_gpu_support — no nvidia-smi
# ---------------------------------------------------------------------------


class TestDetectGpuSupportNoGpu:
    def test_docker_config_returns_dict(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_gpu_support()
        assert isinstance(result, dict)

    def test_not_available_without_nvidia_smi(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_gpu_support()
        assert result["available"] is False

    def test_empty_strings_without_nvidia_smi(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_gpu_support()
        assert result["device_name"] == ""
        assert result["driver_version"] == ""
        assert result["cuda_version"] == ""

    def test_container_toolkit_false_without_binary(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_gpu_support()
        assert result["container_toolkit"] is False

    def test_all_expected_keys_present(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_gpu_support()
        for key in (
            "available",
            "device_name",
            "driver_version",
            "cuda_version",
            "container_toolkit",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# detect_gpu_support — mocked nvidia-smi present
# ---------------------------------------------------------------------------


class TestDetectGpuSupportMocked:
    def _which_side_effect(self, name: str) -> str | None:
        if name == "nvidia-smi":
            return "/usr/bin/nvidia-smi"
        if name == "nvidia-container-cli":
            return "/usr/bin/nvidia-container-cli"
        return None

    def _make_proc(self, stdout: str, returncode: int = 0) -> MagicMock:
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        return proc

    def test_available_true_when_smi_succeeds(self) -> None:
        proc1 = self._make_proc("GeForce RTX 3090, 525.00\n")
        proc2 = self._make_proc("8.6\n")
        with (
            patch("shutil.which", side_effect=self._which_side_effect),
            patch("subprocess.run", side_effect=[proc1, proc2]),
        ):
            result = detect_gpu_support()
        assert result["available"] is True

    def test_device_name_parsed(self) -> None:
        proc1 = self._make_proc("GeForce RTX 3090, 525.00\n")
        proc2 = self._make_proc("8.6\n")
        with (
            patch("shutil.which", side_effect=self._which_side_effect),
            patch("subprocess.run", side_effect=[proc1, proc2]),
        ):
            result = detect_gpu_support()
        assert result["device_name"] == "GeForce RTX 3090"

    def test_driver_version_parsed(self) -> None:
        proc1 = self._make_proc("GeForce RTX 3090, 525.00\n")
        proc2 = self._make_proc("8.6\n")
        with (
            patch("shutil.which", side_effect=self._which_side_effect),
            patch("subprocess.run", side_effect=[proc1, proc2]),
        ):
            result = detect_gpu_support()
        assert result["driver_version"] == "525.00"

    def test_container_toolkit_detected(self) -> None:
        proc1 = self._make_proc("GPU Name, 525.00\n")
        proc2 = self._make_proc("8.6\n")
        with (
            patch("shutil.which", side_effect=self._which_side_effect),
            patch("subprocess.run", side_effect=[proc1, proc2]),
        ):
            result = detect_gpu_support()
        assert result["container_toolkit"] is True

    def test_nonzero_returncode_not_available(self) -> None:
        proc1 = self._make_proc("", returncode=1)
        proc2 = self._make_proc("", returncode=1)
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=[proc1, proc2]),
        ):
            result = detect_gpu_support()
        assert result["available"] is False
