"""Tests for src.shared.python.data_io.provenance (Issues #1949, #1744)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from src.shared.python.data_io.provenance import (
    ProvenanceInfo,
    add_provenance_header_file,
    add_provenance_to_csv,
)

# ---------------------------------------------------------------------------
# ProvenanceInfo dataclass structure
# ---------------------------------------------------------------------------


class TestProvenanceInfoDefaults:
    def test_software_name_default(self) -> None:
        pi = ProvenanceInfo(
            timestamp_utc="2026-01-01T00:00:00Z",
            timestamp_local="2026-01-01T00:00:00+00:00",
        )
        assert isinstance(pi.software_name, str)

    def test_software_version_default(self) -> None:
        pi = ProvenanceInfo(
            timestamp_utc="2026-01-01T00:00:00Z",
            timestamp_local="2026-01-01T00:00:00+00:00",
        )
        assert isinstance(pi.software_version, str)

    def test_git_commit_sha_default_none(self) -> None:
        pi = ProvenanceInfo(
            timestamp_utc="2026-01-01T00:00:00Z",
            timestamp_local="2026-01-01T00:00:00+00:00",
        )
        assert pi.git_commit_sha is None or isinstance(pi.git_commit_sha, str)

    def test_parameters_default_empty(self) -> None:
        pi = ProvenanceInfo(
            timestamp_utc="2026-01-01T00:00:00Z",
            timestamp_local="2026-01-01T00:00:00+00:00",
        )
        assert pi.parameters == {}

    def test_frozen_mutation_raises(self) -> None:
        pi = ProvenanceInfo(
            timestamp_utc="2026-01-01T00:00:00Z",
            timestamp_local="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises((AttributeError, TypeError)):
            pi.software_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProvenanceInfo.capture
# ---------------------------------------------------------------------------


class TestProvenanceInfoCapture:
    def test_returns_provenance_info(self) -> None:
        result = ProvenanceInfo.capture()
        assert isinstance(result, ProvenanceInfo)

    def test_timestamp_utc_non_empty(self) -> None:
        result = ProvenanceInfo.capture()
        assert len(result.timestamp_utc) > 0

    def test_timestamp_local_non_empty(self) -> None:
        result = ProvenanceInfo.capture()
        assert len(result.timestamp_local) > 0

    def test_timestamp_utc_iso_format(self) -> None:
        result = ProvenanceInfo.capture()
        # Should end with 'Z' (UTC)
        assert result.timestamp_utc.endswith("Z")

    def test_python_version_captured(self) -> None:
        result = ProvenanceInfo.capture()
        assert result.python_version is not None
        assert "." in result.python_version

    def test_numpy_version_captured(self) -> None:
        result = ProvenanceInfo.capture()
        assert result.numpy_version is not None

    def test_parameters_stored(self) -> None:
        result = ProvenanceInfo.capture(parameters={"dt": 0.001, "steps": 100})
        assert result.parameters["dt"] == 0.001
        assert result.parameters["steps"] == 100

    def test_model_path_stored_when_exists(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.xml"
        model_file.write_text("<model/>")
        result = ProvenanceInfo.capture(model_path=model_file)
        assert result.model_file_path is not None
        assert result.model_file_hash is not None

    def test_model_hash_is_hex(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.xml"
        model_file.write_text("<model/>")
        result = ProvenanceInfo.capture(model_path=model_file)
        # SHA256 hex digest is 64 chars
        assert result.model_file_hash is not None
        assert all(c in "0123456789abcdef" for c in result.model_file_hash)

    def test_nonexistent_model_path_graceful(self) -> None:
        # Should not raise for a nonexistent path
        result = ProvenanceInfo.capture(model_path="/nonexistent/model.xml")
        assert result.model_file_hash is None


# ---------------------------------------------------------------------------
# ProvenanceInfo._hash_file
# ---------------------------------------------------------------------------


class TestProvenanceInfoHashFile:
    def test_hash_is_hex_string(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        h = ProvenanceInfo._hash_file(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 = 64 hex chars

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert ProvenanceInfo._hash_file(f1) != ProvenanceInfo._hash_file(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"identical")
        f2.write_bytes(b"identical")
        assert ProvenanceInfo._hash_file(f1) == ProvenanceInfo._hash_file(f2)


# ---------------------------------------------------------------------------
# ProvenanceInfo.to_header_lines
# ---------------------------------------------------------------------------


class TestProvenanceInfoHeaderLines:
    def test_provenance_returns_list(self) -> None:
        pi = ProvenanceInfo.capture()
        lines = pi.to_header_lines()
        assert isinstance(lines, list)

    def test_lines_start_with_hash(self) -> None:
        pi = ProvenanceInfo.capture()
        lines = pi.to_header_lines()
        assert len(lines) > 0
        for line in lines:
            assert line.startswith("#")

    def test_timestamp_in_header(self) -> None:
        pi = ProvenanceInfo.capture()
        combined = "\n".join(pi.to_header_lines())
        assert pi.timestamp_utc in combined


# ---------------------------------------------------------------------------
# add_provenance_header_file
# ---------------------------------------------------------------------------


class TestAddProvenanceHeaderFile:
    def test_writes_to_file(self) -> None:
        buf = io.StringIO()
        pi = ProvenanceInfo.capture()
        add_provenance_header_file(buf, pi)
        content = buf.getvalue()
        assert len(content) > 0

    def test_content_has_hash_lines(self) -> None:
        buf = io.StringIO()
        pi = ProvenanceInfo.capture()
        add_provenance_header_file(buf, pi)
        lines = buf.getvalue().splitlines()
        assert any(line.startswith("#") for line in lines)


# ---------------------------------------------------------------------------
# add_provenance_to_csv
# ---------------------------------------------------------------------------


class TestAddProvenanceToCSV:
    def test_prepends_header_to_existing_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")
        pi = ProvenanceInfo.capture()
        add_provenance_to_csv(csv_file, provenance=pi)
        content = csv_file.read_text()
        # Original data should still be present
        assert "x,y" in content

    def test_header_lines_added(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("a,b\n1,2\n")
        add_provenance_to_csv(csv_file)
        content = csv_file.read_text()
        assert "#" in content

    def test_returns_provenance_info(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("val\n42\n")
        result = add_provenance_to_csv(csv_file)
        assert isinstance(result, ProvenanceInfo)
