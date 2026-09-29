"""Tests for :mod:`training.resources`."""

from __future__ import annotations

import dataclasses

import pytest

from training import ResourceRequest, TrainingConfigError

pytestmark = pytest.mark.unit


class TestResourceRequest:
    def test_default_construction(self) -> None:
        req = ResourceRequest()
        assert req.cpu_cores == 1
        assert req.gpu_count == 0
        assert req.memory_mb == 1024
        assert req.gpu_memory_mb is None
        assert req.requires_gpu is False

    def test_explicit_gpu_request(self) -> None:
        req = ResourceRequest(
            cpu_cores=4, gpu_count=2, memory_mb=8192, gpu_memory_mb=12_000
        )
        assert req.requires_gpu is True
        assert req.gpu_memory_mb == 12_000

    def test_is_frozen(self) -> None:
        req = ResourceRequest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.cpu_cores = 16  # type: ignore[misc]

    @pytest.mark.parametrize("cpu_cores", [0, -1, -100])
    def test_rejects_non_positive_cpu(self, cpu_cores: int) -> None:
        with pytest.raises(TrainingConfigError):
            ResourceRequest(cpu_cores=cpu_cores)

    def test_rejects_non_int_cpu(self) -> None:
        with pytest.raises(TrainingConfigError):
            ResourceRequest(cpu_cores=1.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gpu_count", [-1, -10])
    def test_rejects_negative_gpu(self, gpu_count: int) -> None:
        with pytest.raises(TrainingConfigError):
            ResourceRequest(gpu_count=gpu_count)

    def test_rejects_tiny_memory(self) -> None:
        with pytest.raises(TrainingConfigError):
            ResourceRequest(memory_mb=32)  # below MIN_MEMORY_MB

    def test_rejects_gpu_memory_with_no_gpu(self) -> None:
        with pytest.raises(TrainingConfigError) as excinfo:
            ResourceRequest(gpu_count=0, gpu_memory_mb=4000)
        assert "gpu_memory_mb" in str(excinfo.value)

    def test_rejects_non_positive_gpu_memory(self) -> None:
        with pytest.raises(TrainingConfigError):
            ResourceRequest(gpu_count=1, gpu_memory_mb=0)
        with pytest.raises(TrainingConfigError):
            ResourceRequest(gpu_count=1, gpu_memory_mb=-1)

    def test_gpu_count_with_no_explicit_memory_is_allowed(self) -> None:
        req = ResourceRequest(gpu_count=1, gpu_memory_mb=None)
        assert req.requires_gpu is True
        assert req.gpu_memory_mb is None
