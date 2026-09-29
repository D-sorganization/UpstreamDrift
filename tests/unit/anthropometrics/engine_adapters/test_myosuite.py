"""Round-trip tests for the MyoSuite (URDF + MJCF) engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthropometrics import ADAPTER_REGISTRY, EngineAdapter, SubjectAnthropometrics
from anthropometrics.engine_adapters import MyoSuiteAdapter

from ._assertions import assert_subjects_equal


def test_myosuite_adapter_satisfies_protocol() -> None:
    assert isinstance(MyoSuiteAdapter(), EngineAdapter)


def test_myosuite_in_registry() -> None:
    assert "myosuite" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["myosuite"].engine_name == "myosuite"


def test_myosuite_export_emits_both_urdf_and_mjcf(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    base = tmp_path / "subject"
    MyoSuiteAdapter().export(sixteen_segment_subject, base)
    assert base.with_suffix(".urdf").exists()
    assert base.with_suffix(".xml").exists()


def test_myosuite_round_trip_via_urdf(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = MyoSuiteAdapter()
    base = tmp_path / "subject"
    adapter.export(sixteen_segment_subject, base)
    restored = adapter.import_back(base.with_suffix(".urdf"))
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_myosuite_round_trip_via_mjcf(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = MyoSuiteAdapter()
    base = tmp_path / "subject"
    adapter.export(sixteen_segment_subject, base)
    restored = adapter.import_back(base.with_suffix(".xml"))
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_myosuite_unknown_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".urdf"):
        MyoSuiteAdapter().import_back(tmp_path / "subject.txt")


def test_myosuite_export_rejects_non_subject(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        MyoSuiteAdapter().export("nope", tmp_path / "subject")  # type: ignore[arg-type]


def test_myosuite_mjcf_missing_sidecar_raises(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    base = tmp_path / "subject"
    MyoSuiteAdapter().export(sixteen_segment_subject, base)
    # Delete the sidecar JSON.
    sidecar = base.with_suffix(".xml.meta.json")
    sidecar.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        MyoSuiteAdapter().import_back(base.with_suffix(".xml"))


def test_myosuite_urdf_missing_metadata_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.urdf"
    bad.write_text(
        '<?xml version="1.0"?><robot name="x"><link name="a"/></robot>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ud:metadata"):
        MyoSuiteAdapter().import_back(bad)


def test_myosuite_mjcf_wrong_root_raises(tmp_path: Path) -> None:
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text('<?xml version="1.0"?><not_mujoco/>', encoding="utf-8")
    sidecar = tmp_path / "bad.xml.meta.json"
    sidecar.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="<mujoco>"):
        MyoSuiteAdapter().import_back(bad_xml)


def test_myosuite_mjcf_missing_worldbody_raises(tmp_path: Path) -> None:
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text('<?xml version="1.0"?><mujoco model="x"/>', encoding="utf-8")
    sidecar = tmp_path / "bad.xml.meta.json"
    sidecar.write_text(
        '{"subject_id":"x","height_m":1.0,"mass_kg":1.0,'
        '"sex":"M","source_method":"x","age_years":null,"segments":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="<worldbody>"):
        MyoSuiteAdapter().import_back(bad_xml)


def test_myosuite_mjcf_sidecar_orphan_segment_raises(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """Sidecar listing a segment absent from MJCF raises a clear error."""
    import json

    base = tmp_path / "subject"
    MyoSuiteAdapter().export(sixteen_segment_subject, base)
    sidecar_path = base.with_suffix(".xml.meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["segments"][0]["name"] = "ghost_segment"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="ghost_segment"):
        MyoSuiteAdapter().import_back(base.with_suffix(".xml"))


def test_myosuite_optional_mujoco_wheel_loadable(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """If MuJoCo is installed, the produced MJCF must parse."""
    mujoco = pytest.importorskip("mujoco")
    base = tmp_path / "subject"
    MyoSuiteAdapter().export(sixteen_segment_subject, base)
    mujoco.MjModel.from_xml_path(str(base.with_suffix(".xml")))
