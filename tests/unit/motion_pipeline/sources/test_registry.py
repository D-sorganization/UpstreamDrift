"""Tests for the source-adapter registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import (
    detect_format,
    list_formats,
    register_adapter,
    registered_adapters,
    unregister_adapter,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
    UnsupportedFormatError,
)


class _ClaimsAll(MocapSourceAdapter):
    format_name = "_claims_all"
    file_extensions = (".unique_test_ext",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        return path.suffix == ".unique_test_ext"

    def metadata(self, path: Path) -> SourceMetadata:
        return SourceMetadata(
            format_name=self.format_name,
            fps=30.0,
            frame_count=1,
            unit_system="meters",
        )

    def load(self, path: Path, calibration=None):  # noqa: ARG002
        raise NotImplementedError


@pytest.fixture
def fake_registered() -> _ClaimsAll:
    register_adapter(_ClaimsAll)
    yield _ClaimsAll
    unregister_adapter(_ClaimsAll)


def test_detect_format_finds_registered(fake_registered, tmp_path: Path) -> None:
    p = tmp_path / "f.unique_test_ext"
    p.write_text("")
    cls = detect_format(p)
    assert cls is fake_registered


def test_detect_format_raises_on_unknown(tmp_path: Path) -> None:
    p = tmp_path / "nope.totally_unknown_extension_xyz"
    p.write_text("not a real file")
    with pytest.raises(UnsupportedFormatError):
        detect_format(p)


def test_list_formats_includes_known() -> None:
    fmts = list_formats()
    assert "bvh" in fmts
    assert "trc" in fmts
    assert "opensim_sto_mot" in fmts


def test_first_registered_wins(tmp_path: Path) -> None:
    """Two adapters claiming the same path: first registered wins."""

    class _A(MocapSourceAdapter):
        format_name = "_first"
        file_extensions = (".dup_test",)

        @classmethod
        def supports(cls, path: Path) -> bool:
            return path.suffix == ".dup_test"

        def metadata(self, path: Path) -> SourceMetadata:
            return SourceMetadata(
                format_name="_first", fps=30.0, frame_count=1, unit_system="meters"
            )

        def load(self, path: Path, calibration=None):  # noqa: ARG002
            raise NotImplementedError

    class _B(_A):
        format_name = "_second"

    register_adapter(_A)
    register_adapter(_B)
    try:
        p = tmp_path / "x.dup_test"
        p.write_text("")
        assert detect_format(p) is _A
    finally:
        unregister_adapter(_A)
        unregister_adapter(_B)


def test_registered_adapters_snapshot_is_tuple() -> None:
    snap = registered_adapters()
    assert isinstance(snap, tuple)
    assert all(isinstance(c, type) for c in snap)
