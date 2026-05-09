"""Round-trip tests for the OpenSim ``.osim`` engine adapter."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from anthropometrics import ADAPTER_REGISTRY, EngineAdapter, SubjectAnthropometrics
from anthropometrics.engine_adapters import OpenSimAdapter

from ._assertions import assert_subjects_equal


def test_opensim_adapter_satisfies_protocol() -> None:
    assert isinstance(OpenSimAdapter(), EngineAdapter)


def test_opensim_in_registry() -> None:
    assert "opensim" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["opensim"].engine_name == "opensim"


def test_opensim_round_trip_preserves_subject(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = OpenSimAdapter()
    out = tmp_path / "subject.osim"
    adapter.export(sixteen_segment_subject, out)
    assert out.exists()
    restored = adapter.import_back(out)
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_opensim_export_emits_body_elements(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    out = tmp_path / "subject.osim"
    OpenSimAdapter().export(sixteen_segment_subject, out)
    root = ET.parse(out).getroot()
    bodies = root.findall("./Model/BodySet/objects/Body")
    assert len(bodies) == 16


def test_opensim_export_rejects_non_subject(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        OpenSimAdapter().export("nope", tmp_path / "x.osim")  # type: ignore[arg-type]


def test_opensim_wrong_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.osim"
    bad.write_text('<?xml version="1.0"?><NotOpenSim/>', encoding="utf-8")
    with pytest.raises(ValueError, match="OpenSimDocument"):
        OpenSimAdapter().import_back(bad)


def test_opensim_missing_metadata_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.osim"
    bad.write_text(
        '<?xml version="1.0"?><OpenSimDocument><Model name="x"/></OpenSimDocument>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="UDSubjectMetadata"):
        OpenSimAdapter().import_back(bad)


def test_opensim_optional_opensim_wheel_loadable(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """If the opensim wheel is installed, the produced model must parse."""
    osim = pytest.importorskip("opensim")
    out = tmp_path / "subject.osim"
    OpenSimAdapter().export(sixteen_segment_subject, out)
    osim.Model(str(out))
