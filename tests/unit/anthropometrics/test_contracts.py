"""Unit tests for the anthropometrics canonical data model + Protocols.

Covers every validation rule on :class:`SegmentProperties` and
:class:`SubjectAnthropometrics` plus a runtime-checkable
isinstance assertion for each Protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anthropometrics import (
    EngineAdapter,
    Estimator,
    Reader,
    SegmentProperties,
    Sex,
    SubjectAnthropometrics,
    Writer,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders.                                                        #
# --------------------------------------------------------------------------- #
def _diag_inertia(ix: float, iy: float, iz: float) -> np.ndarray:
    """Return a diagonal inertia tensor with the given principal moments."""
    return np.diag([ix, iy, iz]).astype(float)


def _make_segment(**overrides: Any) -> SegmentProperties:
    """Return a default-valid :class:`SegmentProperties`, with overrides."""
    defaults: dict[str, Any] = {
        "name": "upper_arm_left",
        "body_part_id": "upper_arm",
        "length_m": 0.30,
        "proximal_marker": "L_SHO",
        "distal_marker": "L_ELB",
        "mass_kg": 2.0,
        "com_xyz_m": np.array([0.15, 0.0, 0.0]),
        "inertia_tensor": _diag_inertia(0.02, 0.02, 0.005),
        "source_method": "de_leva",
        "source_subject_height_m": 1.80,
        "source_subject_mass_kg": 75.0,
    }
    defaults.update(overrides)
    return SegmentProperties(**defaults)


def _make_subject(**overrides: Any) -> SubjectAnthropometrics:
    """Return a default-valid :class:`SubjectAnthropometrics`."""
    seg = _make_segment()
    defaults: dict[str, Any] = {
        "subject_id": "SUBJ001",
        "height_m": 1.80,
        "mass_kg": 75.0,
        "segments": ((seg.name, seg),),
        "source_method": "de_leva",
    }
    defaults.update(overrides)
    return SubjectAnthropometrics(**defaults)


# --------------------------------------------------------------------------- #
# Happy paths.                                                                #
# --------------------------------------------------------------------------- #
def test_segment_properties_happy_path_constructs_and_freezes() -> None:
    seg = _make_segment()
    assert seg.name == "upper_arm_left"
    assert seg.length_m == pytest.approx(0.30)
    # frozen dataclass: assignment must raise.
    with pytest.raises(AttributeError):
        seg.length_m = 0.5  # type: ignore[misc]
    # com_xyz_m + inertia_tensor coerced to ndarray.
    assert isinstance(seg.com_xyz_m, np.ndarray)
    assert isinstance(seg.inertia_tensor, np.ndarray)
    assert seg.inertia_tensor.shape == (3, 3)


def test_subject_anthropometrics_happy_path() -> None:
    subj = _make_subject()
    assert subj.subject_id == "SUBJ001"
    assert subj.sex == Sex.UNSPECIFIED.value
    assert len(subj.segments) == 1


# --------------------------------------------------------------------------- #
# Mass / length scalar validation.                                            #
# --------------------------------------------------------------------------- #
def test_segment_zero_mass_rejected() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        _make_segment(mass_kg=0.0)


def test_segment_negative_length_rejected() -> None:
    with pytest.raises(ValueError, match="length_m"):
        _make_segment(length_m=-0.1)


def test_segment_non_finite_mass_rejected() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        _make_segment(mass_kg=float("nan"))


# --------------------------------------------------------------------------- #
# String identifiers.                                                         #
# --------------------------------------------------------------------------- #
def test_segment_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        _make_segment(name="")


def test_segment_blank_body_part_id_rejected() -> None:
    with pytest.raises(ValueError, match="body_part_id"):
        _make_segment(body_part_id="   ")


def test_segment_empty_source_method_rejected() -> None:
    with pytest.raises(ValueError, match="source_method"):
        _make_segment(source_method="")


def test_segment_empty_proximal_marker_rejected() -> None:
    with pytest.raises(ValueError, match="proximal_marker"):
        _make_segment(proximal_marker="")


def test_segment_optional_markers_may_be_none() -> None:
    seg = _make_segment(proximal_marker=None, distal_marker=None)
    assert seg.proximal_marker is None
    assert seg.distal_marker is None


# --------------------------------------------------------------------------- #
# Center-of-mass validation.                                                  #
# --------------------------------------------------------------------------- #
def test_segment_com_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="com_xyz_m"):
        _make_segment(com_xyz_m=np.zeros(2))


def test_segment_com_outside_bounds_rejected() -> None:
    # length_m=0.30, so |com| must be <= 0.60.
    with pytest.raises(ValueError, match="com_xyz_m"):
        _make_segment(com_xyz_m=np.array([1.0, 0.0, 0.0]))


def test_segment_com_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="com_xyz_m"):
        _make_segment(com_xyz_m=np.array([np.inf, 0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Inertia tensor invariants.                                                  #
# --------------------------------------------------------------------------- #
def test_inertia_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="inertia_tensor"):
        _make_segment(inertia_tensor=np.eye(2))


def test_inertia_asymmetric_rejected() -> None:
    asym = np.array(
        [[1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=float,
    )
    with pytest.raises(ValueError, match="symmetric"):
        _make_segment(inertia_tensor=asym)


def test_inertia_negative_eigenvalue_rejected() -> None:
    # Symmetric but with one negative eigenvalue.
    tensor = _diag_inertia(1.0, 1.0, -0.5)
    with pytest.raises(ValueError, match="positive-definite"):
        _make_segment(inertia_tensor=tensor)


def test_inertia_zero_eigenvalue_rejected() -> None:
    tensor = _diag_inertia(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive-definite"):
        _make_segment(inertia_tensor=tensor)


def test_inertia_triangle_inequality_violation_rejected() -> None:
    # eigenvalues (1, 1, 3): 1 + 1 < 3 violates triangle inequality.
    tensor = _diag_inertia(1.0, 1.0, 3.0)
    with pytest.raises(ValueError, match="triangle inequality"):
        _make_segment(inertia_tensor=tensor)


def test_inertia_triangle_inequality_message_names_failing_pair() -> None:
    tensor = _diag_inertia(1.0, 1.0, 3.0)
    with pytest.raises(ValueError) as info:
        _make_segment(inertia_tensor=tensor)
    msg = str(info.value)
    assert "Ix+Iy >= Iz" in msg


def test_inertia_off_diagonal_symmetric_accepted() -> None:
    tensor = np.array(
        [[0.020, 0.001, 0.0], [0.001, 0.020, 0.0], [0.0, 0.0, 0.005]],
        dtype=float,
    )
    seg = _make_segment(inertia_tensor=tensor)
    assert seg.inertia_tensor[0, 1] == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# SubjectAnthropometrics validation.                                          #
# --------------------------------------------------------------------------- #
def test_subject_empty_id_rejected() -> None:
    with pytest.raises(ValueError, match="subject_id"):
        _make_subject(subject_id="")


def test_subject_zero_height_rejected() -> None:
    with pytest.raises(ValueError, match="height_m"):
        _make_subject(height_m=0.0)


def test_subject_negative_mass_rejected() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        _make_subject(mass_kg=-1.0)


def test_subject_segments_not_tuple_rejected() -> None:
    seg = _make_segment()
    with pytest.raises(ValueError, match="segments"):
        _make_subject(segments=[(seg.name, seg)])  # type: ignore[arg-type]


def test_subject_empty_segments_rejected() -> None:
    with pytest.raises(ValueError, match="segments"):
        _make_subject(segments=())


def test_subject_duplicate_segment_names_rejected() -> None:
    seg = _make_segment()
    with pytest.raises(ValueError, match="duplicate"):
        _make_subject(segments=((seg.name, seg), (seg.name, seg)))


def test_subject_segment_entry_must_be_pair() -> None:
    seg = _make_segment()
    with pytest.raises(ValueError, match="pairs"):
        _make_subject(segments=((seg,),))  # type: ignore[arg-type]


def test_subject_segment_value_must_be_segment_properties() -> None:
    with pytest.raises(ValueError, match="SegmentProperties"):
        _make_subject(segments=(("upper_arm_left", "not-a-segment"),))  # type: ignore[arg-type]


def test_subject_invalid_sex_rejected() -> None:
    with pytest.raises(ValueError, match="sex"):
        _make_subject(sex="other")


def test_subject_negative_age_rejected() -> None:
    with pytest.raises(ValueError, match="age_years"):
        _make_subject(age_years=-5.0)


def test_subject_age_none_accepted() -> None:
    subj = _make_subject(age_years=None)
    assert subj.age_years is None


def test_subject_empty_source_method_rejected() -> None:
    with pytest.raises(ValueError, match="source_method"):
        _make_subject(source_method="")


# --------------------------------------------------------------------------- #
# Protocol runtime-checkable conformance.                                     #
# --------------------------------------------------------------------------- #
class _StubEstimator:
    def estimate(
        self,
        *,
        subject_id: str,
        height_m: float,
        mass_kg: float,
        sex: str = "unspecified",
        age_years: float | None = None,
    ) -> SubjectAnthropometrics:
        return _make_subject(
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            sex=sex,
            age_years=age_years,
        )


class _StubReader:
    def read(self, path: Path) -> SubjectAnthropometrics:
        del path
        return _make_subject()


class _StubWriter:
    def write(self, anthro: SubjectAnthropometrics, path: Path) -> None:
        del anthro, path


class _StubEngineAdapter:
    def to_engine_segment(self, props: SegmentProperties) -> object:
        return {"mass": props.mass_kg, "length": props.length_m}


def test_estimator_protocol_satisfied_by_stub() -> None:
    assert isinstance(_StubEstimator(), Estimator)


def test_reader_protocol_satisfied_by_stub() -> None:
    assert isinstance(_StubReader(), Reader)


def test_writer_protocol_satisfied_by_stub() -> None:
    assert isinstance(_StubWriter(), Writer)


def test_engine_adapter_protocol_satisfied_by_stub() -> None:
    assert isinstance(_StubEngineAdapter(), EngineAdapter)


def test_protocol_rejects_non_conforming_object() -> None:
    class _NotAnEstimator:
        pass

    assert not isinstance(_NotAnEstimator(), Estimator)
