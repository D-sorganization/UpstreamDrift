"""Unit tests for :class:`anthropometrics.estimators.DempsterEstimator`.

Validates that the JSON-backed Dempster (1955) estimator
reproduces the published numerical values within ``1e-3``,
satisfies the :class:`Estimator` Protocol, and enforces all
DbC preconditions on subject inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthropometrics import Estimator, SubjectAnthropometrics
from anthropometrics.estimators import DempsterEstimator
from anthropometrics.estimators.from_dempster import _DEFAULT_RATIO_FILE


PUBLISHED_TOL = 1e-3


@pytest.fixture(scope="module")
def dempster_table() -> dict:
    """Parsed Dempster ratio JSON for direct lookups."""
    with _DEFAULT_RATIO_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Validation against published Dempster (1955) numbers.                       #
# --------------------------------------------------------------------------- #
def test_published_head_mass_ratio(dempster_table: dict) -> None:
    """Dempster head mass ratio is 0.0810."""
    assert dempster_table["segments"]["head"]["mass_ratio"] == pytest.approx(
        0.0810, abs=PUBLISHED_TOL
    )


def test_published_thigh_mass_ratio(dempster_table: dict) -> None:
    """Dempster thigh mass ratio is 0.1000."""
    assert dempster_table["segments"]["thigh"]["mass_ratio"] == pytest.approx(
        0.1000, abs=PUBLISHED_TOL
    )


def test_published_forearm_com_proximal_ratio(dempster_table: dict) -> None:
    """Dempster forearm CoM proximal ratio is 0.430."""
    assert dempster_table["segments"]["forearm"]["com_proximal_ratio"] == pytest.approx(
        0.430, abs=PUBLISHED_TOL
    )


def test_estimator_method_name_matches_table() -> None:
    """The ``method_name`` attribute must come from the JSON 'method' key."""
    est = DempsterEstimator()
    assert est.method_name == "dempster_1955"


def test_estimator_exposes_citation() -> None:
    """The ``citation`` property must come from the JSON 'citation' key."""
    est = DempsterEstimator()
    assert "Dempster" in est.citation
    assert "1955" in est.citation


# --------------------------------------------------------------------------- #
# Smoke: full subject build + mass closure within 1%.                         #
# --------------------------------------------------------------------------- #
def test_smoke_full_subject_returns_subject_anthropometrics() -> None:
    """1.83 m / 82 kg subject produces a valid SubjectAnthropometrics."""
    est = DempsterEstimator()
    subject = est.estimate(
        subject_id="dempster-smoke",
        height_m=1.83,
        mass_kg=82.0,
    )
    assert isinstance(subject, SubjectAnthropometrics)
    assert subject.source_method == "dempster_1955"
    assert len(subject.segments) >= 8


def test_smoke_mass_closes_within_one_percent() -> None:
    """Sum of segment masses equals total subject mass within 1%."""
    est = DempsterEstimator()
    mass_kg = 82.0
    subject = est.estimate(
        subject_id="closure",
        height_m=1.83,
        mass_kg=mass_kg,
    )
    total = sum(props.mass_kg for _, props in subject.segments)
    assert abs(total - mass_kg) / mass_kg <= 0.01


def test_estimator_protocol_isinstance_check() -> None:
    """The estimator must satisfy the runtime-checkable Estimator Protocol."""
    assert isinstance(DempsterEstimator(), Estimator)


def test_inertia_tensors_are_diagonal_and_valid() -> None:
    """Every segment's inertia tensor is diagonal and physically valid.

    Validation is enforced by :class:`SegmentProperties`
    construction; the assertion here is that we successfully
    build a complete subject without raising.
    """
    est = DempsterEstimator()
    subject = est.estimate(
        subject_id="inertia",
        height_m=1.78,
        mass_kg=75.0,
    )
    for _name, props in subject.segments:
        # Off-diagonal elements must be zero (regression-based
        # estimate uses principal-axis gyration radii only).
        tensor = props.inertia_tensor
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert tensor[i, j] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Design by Contract.                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_height", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_height_raises_value_error(bad_height: float) -> None:
    """``height_m`` must be a positive finite number."""
    est = DempsterEstimator()
    with pytest.raises(ValueError, match="height_m"):
        est.estimate(subject_id="x", height_m=bad_height, mass_kg=70.0)


@pytest.mark.parametrize("bad_mass", [0.0, -50.0, float("nan"), float("inf")])
def test_invalid_mass_raises_value_error(bad_mass: float) -> None:
    """``mass_kg`` must be a positive finite number."""
    est = DempsterEstimator()
    with pytest.raises(ValueError, match="mass_kg"):
        est.estimate(subject_id="x", height_m=1.80, mass_kg=bad_mass)


def test_invalid_subject_id_raises_value_error() -> None:
    """``subject_id`` must be a non-empty string."""
    est = DempsterEstimator()
    with pytest.raises(ValueError, match="subject_id"):
        est.estimate(subject_id="", height_m=1.80, mass_kg=70.0)


def test_invalid_sex_raises_value_error() -> None:
    """``sex`` must be one of M/F/unspecified."""
    est = DempsterEstimator()
    with pytest.raises(ValueError, match="sex"):
        est.estimate(subject_id="x", height_m=1.80, mass_kg=70.0, sex="badcode")


# --------------------------------------------------------------------------- #
# JSON loader: malformed input handling.                                      #
# --------------------------------------------------------------------------- #
def test_missing_ratio_file_raises_file_not_found(tmp_path: Path) -> None:
    """Constructing with a non-existent ratio file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DempsterEstimator(ratio_file=tmp_path / "does-not-exist.json")


def test_malformed_ratio_file_raises_value_error(tmp_path: Path) -> None:
    """Constructing with a malformed JSON raises ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"method": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        DempsterEstimator(ratio_file=bad)


def test_malformed_segments_field_raises_value_error(tmp_path: Path) -> None:
    """Empty segments dict raises ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "x",
                "citation": "y",
                "segments": {},
                "segment_name_map": {"a": "b"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="segments"):
        DempsterEstimator(ratio_file=bad)


def test_non_object_json_root_raises_value_error(tmp_path: Path) -> None:
    """JSON root must be an object, not a list."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        DempsterEstimator(ratio_file=bad)


def test_segments_field_wrong_type_raises_value_error(tmp_path: Path) -> None:
    """The 'segments' field must be a mapping."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "x",
                "citation": "y",
                "segments": ["not", "a", "dict"],
                "segment_name_map": {"a": "b"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="segments"):
        DempsterEstimator(ratio_file=bad)


def test_empty_name_map_raises_value_error(tmp_path: Path) -> None:
    """An empty segment_name_map raises ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "x",
                "citation": "y",
                "segments": {
                    "head": {
                        "mass_ratio": 0.1,
                        "length_ratio": 0.1,
                        "com_proximal_ratio": 0.5,
                        "gyration_radii": {
                            "sagittal": 0.3,
                            "transverse": 0.3,
                            "longitudinal": 0.2,
                        },
                    }
                },
                "segment_name_map": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="segment_name_map"):
        DempsterEstimator(ratio_file=bad)


def test_gyration_radii_wrong_type_raises_value_error(tmp_path: Path) -> None:
    """gyration_radii must be a mapping, not a list."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "x",
                "citation": "y",
                "segments": {
                    "head": {
                        "mass_ratio": 0.1,
                        "length_ratio": 0.1,
                        "com_proximal_ratio": 0.5,
                        "gyration_radii": [0.3, 0.3, 0.2],
                    }
                },
                "segment_name_map": {"head": "head"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gyration_radii"):
        DempsterEstimator(ratio_file=bad)


def test_name_map_unknown_class_raises_value_error(tmp_path: Path) -> None:
    """A name map referencing an undefined class id raises ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "x",
                "citation": "y",
                "segments": {
                    "head": {
                        "mass_ratio": 0.1,
                        "length_ratio": 0.1,
                        "com_proximal_ratio": 0.5,
                        "gyration_radii": {
                            "sagittal": 0.3,
                            "transverse": 0.3,
                            "longitudinal": 0.2,
                        },
                    }
                },
                "segment_name_map": {"head": "head", "torso": "trunk"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown classes"):
        DempsterEstimator(ratio_file=bad)
