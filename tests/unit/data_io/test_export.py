"""Tests for src.shared.python.data_io.export (Issues #1949, #1744)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import src.shared.python.data_io.export as export_module
from src.shared.python.data_io.export import (
    CaptureExportData,
    get_available_export_formats,
)


class TestGetAvailableExportFormats:
    def test_returns_dict(self) -> None:
        formats = get_available_export_formats()
        assert isinstance(formats, dict)

    def test_has_json(self) -> None:
        formats = get_available_export_formats()
        assert "json" in formats

    def test_has_csv(self) -> None:
        formats = get_available_export_formats()
        assert "csv" in formats

    def test_has_mat(self) -> None:
        formats = get_available_export_formats()
        assert "mat" in formats

    def test_has_hdf5(self) -> None:
        formats = get_available_export_formats()
        assert "hdf5" in formats

    def test_has_c3d(self) -> None:
        formats = get_available_export_formats()
        assert "c3d" in formats

    def test_json_always_available(self) -> None:
        formats = get_available_export_formats()
        assert formats["json"]["available"] is True

    def test_csv_always_available(self) -> None:
        formats = get_available_export_formats()
        assert formats["csv"]["available"] is True

    def test_each_format_has_extension(self) -> None:
        formats = get_available_export_formats()
        for name, info in formats.items():
            assert "extension" in info, f"Format '{name}' missing extension"
            assert info["extension"].startswith("."), (
                f"Extension for '{name}' should start with '.'"
            )

    def test_each_format_has_name(self) -> None:
        formats = get_available_export_formats()
        for key, info in formats.items():
            assert "name" in info, f"Format '{key}' missing name"

    def test_each_format_has_available_flag(self) -> None:
        formats = get_available_export_formats()
        for key, info in formats.items():
            assert "available" in info, f"Format '{key}' missing available flag"
            assert isinstance(info["available"], bool)

    def test_each_format_has_description(self) -> None:
        formats = get_available_export_formats()
        for key, info in formats.items():
            assert "description" in info, f"Format '{key}' missing description"


class TestCaptureExportData:
    def test_construction(self) -> None:
        n = 50
        data = CaptureExportData(
            times=np.linspace(0, 1, n),
            joint_positions=np.zeros((n, 3)),
            joint_names=["hip", "knee", "ankle"],
        )
        assert data.times.shape == (n,)
        assert data.joint_positions.shape == (n, 3)
        assert len(data.joint_names) == 3

    def test_default_frame_rate(self) -> None:
        n = 10
        data = CaptureExportData(
            times=np.linspace(0, 1, n),
            joint_positions=np.zeros((n, 2)),
            joint_names=["j1", "j2"],
        )
        assert data.frame_rate == pytest.approx(60.0)

    def test_custom_frame_rate(self) -> None:
        n = 10
        data = CaptureExportData(
            times=np.linspace(0, 1, n),
            joint_positions=np.zeros((n, 2)),
            joint_names=["j1", "j2"],
            frame_rate=120.0,
        )
        assert data.frame_rate == pytest.approx(120.0)

    def test_forces_default_none(self) -> None:
        n = 10
        data = CaptureExportData(
            times=np.linspace(0, 1, n),
            joint_positions=np.zeros((n, 2)),
            joint_names=["j1", "j2"],
        )
        assert data.forces is None

    def test_default_units(self) -> None:
        n = 10
        data = CaptureExportData(
            times=np.linspace(0, 1, n),
            joint_positions=np.zeros((n, 2)),
            joint_names=["j1", "j2"],
        )
        assert "position" in data.units
        assert "force" in data.units
        assert "moment" in data.units


class TestProvenanceBackedAtomicExports:
    def test_matlab_export_writes_temp_then_replaces_final_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_path = tmp_path / "recording.mat"
        output_path.write_bytes(b"old-content")
        seen_paths: list[Path] = []

        def fake_savemat(path: str, *_args: Any, **_kwargs: Any) -> None:
            temp_path = Path(path)
            seen_paths.append(temp_path)
            assert temp_path.parent == output_path.parent
            assert temp_path != output_path
            assert output_path.read_bytes() == b"old-content"
            temp_path.write_bytes(b"new-content")

        monkeypatch.setattr(export_module, "SCIPY_AVAILABLE", True)
        monkeypatch.setattr(export_module, "savemat", fake_savemat)

        result = export_module.export_to_matlab(
            str(output_path),
            {"positions": np.array([1.0, 2.0])},
        )

        assert result is True
        assert seen_paths
        assert output_path.read_bytes() == b"new-content"

    def test_matlab_export_failure_leaves_existing_file_untouched(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_path = tmp_path / "recording.mat"
        output_path.write_bytes(b"old-content")

        def fake_savemat(path: str, *_args: Any, **_kwargs: Any) -> None:
            Path(path).write_bytes(b"partial-content")
            raise OSError("disk full")

        monkeypatch.setattr(export_module, "SCIPY_AVAILABLE", True)
        monkeypatch.setattr(export_module, "savemat", fake_savemat)

        result = export_module.export_to_matlab(
            str(output_path),
            {"positions": np.array([1.0, 2.0])},
        )

        assert result is False
        assert output_path.read_bytes() == b"old-content"

    def test_matlab_export_can_return_structured_outcome_with_sidecar_checksum(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_path = tmp_path / "recording.mat"

        def fake_savemat(path: str, *_args: Any, **_kwargs: Any) -> None:
            Path(path).write_bytes(b"matlab-payload")

        monkeypatch.setattr(export_module, "SCIPY_AVAILABLE", True)
        monkeypatch.setattr(export_module, "savemat", fake_savemat)

        outcome = export_module.export_to_matlab(
            str(output_path),
            {"positions": np.array([1.0, 2.0])},
            return_outcome=True,
        )

        expected_checksum = hashlib.sha256(b"matlab-payload").hexdigest()
        sidecar_path = tmp_path / "recording.mat.provenance.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert outcome.success is True
        assert outcome.path == output_path
        assert outcome.checksum_sha256 == expected_checksum
        assert outcome.provenance_path == sidecar_path
        assert sidecar["artifact"]["checksum_sha256"] == expected_checksum
        assert sidecar["artifact"]["path"] == str(output_path)

    def test_matlab_export_preserves_bool_api_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_savemat(path: str, *_args: Any, **_kwargs: Any) -> None:
            Path(path).write_bytes(b"matlab-payload")

        monkeypatch.setattr(export_module, "SCIPY_AVAILABLE", True)
        monkeypatch.setattr(export_module, "savemat", fake_savemat)

        result = export_module.export_to_matlab(
            str(tmp_path / "recording.mat"),
            {"positions": np.array([1.0, 2.0])},
        )

        assert isinstance(result, bool)

    def test_hdf5_export_can_return_structured_outcome_with_sidecar_checksum(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeGroup:
            def __init__(self) -> None:
                self.attrs: dict[str, Any] = {}

            def create_dataset(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        class FakeFile(FakeGroup):
            def __init__(self, path: str, _mode: str) -> None:
                super().__init__()
                self.path = Path(path)

            def __enter__(self) -> FakeFile:
                self.path.write_bytes(b"hdf5-payload")
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def create_group(self, _name: str) -> FakeGroup:
                return FakeGroup()

        class FakeH5py:
            File = FakeFile

        output_path = tmp_path / "recording.h5"
        monkeypatch.setattr(export_module, "H5PY_AVAILABLE", True)
        monkeypatch.setattr(export_module, "h5py", FakeH5py, raising=False)

        outcome = export_module.export_to_hdf5(
            str(output_path),
            {"positions": np.array([1.0, 2.0]), "club": "driver"},
            return_outcome=True,
        )

        expected_checksum = hashlib.sha256(b"hdf5-payload").hexdigest()
        sidecar_path = tmp_path / "recording.h5.provenance.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert outcome.success is True
        assert outcome.path == output_path
        assert outcome.checksum_sha256 == expected_checksum
        assert outcome.provenance_path == sidecar_path
        assert sidecar["artifact"]["checksum_sha256"] == expected_checksum
