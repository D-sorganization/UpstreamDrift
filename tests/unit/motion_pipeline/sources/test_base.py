"""Tests for the MocapSourceAdapter ABC and SourceMetadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.base import (
    AdapterContractError,
    MocapSourceAdapter,
    SourceMetadata,
)


def test_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MocapSourceAdapter()  # type: ignore[abstract]


def test_subclass_without_overrides_fails() -> None:
    class Incomplete(MocapSourceAdapter):
        format_name = "incomplete"
        file_extensions = (".x",)

        # Missing supports/metadata/load on purpose

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_source_metadata_required_fields() -> None:
    md = SourceMetadata(
        format_name="x",
        fps=30.0,
        frame_count=10,
        unit_system="meters",
    )
    assert md.format_name == "x"
    assert md.keypoint_schema is None


def test_source_metadata_rejects_zero_fps() -> None:
    with pytest.raises(ValueError):
        SourceMetadata(format_name="x", fps=0.0, frame_count=1, unit_system="meters")


class _Stub(MocapSourceAdapter):
    format_name = "stub"
    file_extensions = (".stub",)

    @classmethod
    def supports(cls, path: Path) -> bool:  # noqa: ARG003
        return True

    def metadata(self, path: Path) -> SourceMetadata:  # noqa: ARG002
        return SourceMetadata(
            format_name=self.format_name,
            fps=30.0,
            frame_count=0,
            unit_system="meters",
        )

    def load(self, path: Path, calibration=None):  # noqa: ARG002
        # Return an object that lacks frames -> postcondition failure.
        class _Empty:
            frames = []

        return _Empty()  # type: ignore[return-value]


def test_load_checked_rejects_empty_payload(tmp_path: Path) -> None:
    p = tmp_path / "x.stub"
    p.write_text("dummy")
    with pytest.raises(AdapterContractError):
        _Stub().load_checked(p)
