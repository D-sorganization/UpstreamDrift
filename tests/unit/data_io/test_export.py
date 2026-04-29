"""Tests for src.shared.python.data_io.export (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.data_io.export import (
    CaptureExportData,
    get_available_export_formats,
)

pytestmark = pytest.mark.unit


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
            assert info["extension"].startswith(
                "."
            ), f"Extension for '{name}' should start with '.'"

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
