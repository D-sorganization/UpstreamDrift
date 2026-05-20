"""Tests for output path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.data_io._format_handlers import OutputFormat
from src.shared.python.data_io._path_utils import (
    create_output_structure,
    fast_dir_scan,
    resolve_base_path,
    sanitize_filename,
)


def test_sanitize_filename_strips_format_suffix():
    assert sanitize_filename("data_1.csv", OutputFormat.CSV) == "data_1"


def test_sanitize_filename_adds_timestamp_when_no_digit():
    out = sanitize_filename("simulation", OutputFormat.CSV)
    assert out.startswith("simulation_")
    # appended timestamp should contain digits
    assert any(c.isdigit() for c in out)


def test_sanitize_filename_test_mode_no_timestamp():
    # "test_" prefix bypasses timestamp appending
    assert sanitize_filename("test_data", OutputFormat.CSV) == "test_data"


def test_sanitize_filename_with_digit_no_timestamp():
    assert sanitize_filename("run_42", OutputFormat.JSON) == "run_42"


def test_sanitize_filename_outputformat_string():
    out = sanitize_filename("OutputFormat.CSV", OutputFormat.CSV)
    assert out == "test_format"


def test_fast_dir_scan_finds_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    (sub / "c.txt").write_text("c")

    results = sorted(p.name for p in fast_dir_scan(tmp_path))
    assert results == ["a.txt", "b.txt", "c.txt"]


def test_fast_dir_scan_respects_max_depth(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.txt").write_text("x")
    (tmp_path / "shallow.txt").write_text("s")

    shallow_only = list(fast_dir_scan(tmp_path, max_depth=0))
    names = [p.name for p in shallow_only]
    assert "shallow.txt" in names
    assert "deep.txt" not in names


def test_fast_dir_scan_missing_directory(tmp_path: Path):
    results = list(fast_dir_scan(tmp_path / "does_not_exist"))
    assert results == []


def test_create_output_structure(tmp_path: Path):
    dirs = {
        "simulations": tmp_path / "simulations",
        "analysis": tmp_path / "analysis",
        "exports": tmp_path / "exports",
        "reports": tmp_path / "reports",
        "cache": tmp_path / "cache",
    }
    create_output_structure(dirs)

    for d in dirs.values():
        assert d.is_dir()

    for engine in ["mujoco", "drake", "pinocchio", "matlab"]:
        assert (dirs["simulations"] / engine).is_dir()

    for at in ["biomechanics", "trajectories", "optimization", "comparisons"]:
        assert (dirs["analysis"] / at).is_dir()

    for et in ["videos", "images", "data", "c3d"]:
        assert (dirs["exports"] / et).is_dir()

    for rt in ["pdf", "html", "presentations"]:
        assert (dirs["reports"] / rt).is_dir()

    for ct in ["models", "computations", "temp"]:
        assert (dirs["cache"] / ct).is_dir()


def test_resolve_base_path_with_explicit_path(tmp_path: Path):
    target = tmp_path / "myout"
    result = resolve_base_path(target)
    assert result == target
    assert result.is_dir()


def test_resolve_base_path_string_input(tmp_path: Path):
    target = tmp_path / "as_str"
    result = resolve_base_path(str(target))
    assert result == target
    assert result.is_dir()


def test_resolve_base_path_none_returns_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve_base_path(None)
    assert isinstance(result, Path)
    assert result.is_dir()
