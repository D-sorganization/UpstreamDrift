"""Round-trip tests for the Drake URDF engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthropometrics import ADAPTER_REGISTRY, EngineAdapter, SubjectAnthropometrics
from anthropometrics.engine_adapters import DrakeAdapter

from ._assertions import assert_subjects_equal


def test_drake_adapter_satisfies_protocol() -> None:
    assert isinstance(DrakeAdapter(), EngineAdapter)


def test_drake_in_registry() -> None:
    assert "drake" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["drake"].engine_name == "drake"


def test_drake_round_trip_preserves_subject(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = DrakeAdapter()
    out = tmp_path / "subject.urdf"
    adapter.export(sixteen_segment_subject, out)
    assert out.exists() and out.stat().st_size > 0
    restored = adapter.import_back(out)
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_drake_export_rejects_non_subject(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        DrakeAdapter().export("not a subject", tmp_path / "x.urdf")  # type: ignore[arg-type]


def test_drake_wrong_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.urdf"
    bad.write_text('<?xml version="1.0"?><not_robot/>', encoding="utf-8")
    with pytest.raises(ValueError, match="<robot>"):
        DrakeAdapter().import_back(bad)


def test_drake_missing_metadata_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.urdf"
    bad.write_text(
        '<?xml version="1.0"?><robot name="x"><link name="y"/></robot>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ud:metadata"):
        DrakeAdapter().import_back(bad)


def test_drake_optional_drake_wheel_loadable(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """If pydrake is installed, the produced URDF must parse."""
    pydrake = pytest.importorskip("pydrake.multibody.parsing")
    from pydrake.multibody.plant import MultibodyPlant  # type: ignore

    out = tmp_path / "subject.urdf"
    DrakeAdapter().export(sixteen_segment_subject, out)
    plant = MultibodyPlant(time_step=0.0)
    parser = pydrake.Parser(plant)
    parser.AddModels(str(out))
