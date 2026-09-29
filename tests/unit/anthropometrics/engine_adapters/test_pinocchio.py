"""Round-trip tests for the Pinocchio URDF engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthropometrics import ADAPTER_REGISTRY, EngineAdapter, SubjectAnthropometrics
from anthropometrics.engine_adapters import PinocchioAdapter

from ._assertions import assert_subjects_equal


def test_pinocchio_adapter_satisfies_protocol() -> None:
    assert isinstance(PinocchioAdapter(), EngineAdapter)


def test_pinocchio_in_registry() -> None:
    assert "pinocchio" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["pinocchio"].engine_name == "pinocchio"


def test_pinocchio_round_trip_preserves_subject(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = PinocchioAdapter()
    out = tmp_path / "subject.urdf"
    adapter.export(sixteen_segment_subject, out)
    assert out.exists()
    restored = adapter.import_back(out)
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_pinocchio_export_rejects_non_subject(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        PinocchioAdapter().export(  # type: ignore[arg-type]
            "not a subject", tmp_path / "x.urdf"
        )


def test_pinocchio_optional_pinocchio_wheel_loadable(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """If the pinocchio wheel is installed, the URDF must build a model."""
    pin = pytest.importorskip("pinocchio")
    out = tmp_path / "subject.urdf"
    PinocchioAdapter().export(sixteen_segment_subject, out)
    pin.buildModelFromUrdf(str(out))
