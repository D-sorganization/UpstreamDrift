"""Unit tests for :class:`anthropometrics.estimators.ZatsiorskyEstimator`.

Validates that the JSON-backed Zatsiorsky-Seluyanov (1985)
estimator reproduces the published numerical values within
``1e-3``, satisfies the :class:`Estimator` Protocol, and enforces
all DbC preconditions on subject inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthropometrics import Estimator, SubjectAnthropometrics
from anthropometrics.estimators import ZatsiorskyEstimator
from anthropometrics.estimators.from_zatsiorsky import _DEFAULT_RATIO_FILE


PUBLISHED_TOL = 1e-3


@pytest.fixture(scope="module")
def zatsiorsky_table() -> dict:
    """Parsed Zatsiorsky-Seluyanov ratio JSON for direct lookups."""
    with _DEFAULT_RATIO_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Validation against published Zatsiorsky-Seluyanov (1985) numbers.           #
# --------------------------------------------------------------------------- #
def test_published_thigh_mass_ratio(zatsiorsky_table: dict) -> None:
    """Zatsiorsky-Seluyanov thigh mass ratio is 0.1416."""
    assert zatsiorsky_table["segments"]["thigh"]["mass_ratio"] == pytest.approx(
        0.1416, abs=PUBLISHED_TOL
    )


def test_published_trunk_mass_ratio(zatsiorsky_table: dict) -> None:
    """Zatsiorsky-Seluyanov trunk mass ratio is 0.4346."""
    assert zatsiorsky_table["segments"]["trunk"]["mass_ratio"] == pytest.approx(
        0.4346, abs=PUBLISHED_TOL
    )


def test_published_head_mass_ratio(zatsiorsky_table: dict) -> None:
    """Zatsiorsky-Seluyanov head mass ratio is 0.0694."""
    assert zatsiorsky_table["segments"]["head"]["mass_ratio"] == pytest.approx(
        0.0694, abs=PUBLISHED_TOL
    )


def test_estimator_method_name_matches_table() -> None:
    """The ``method_name`` attribute must come from the JSON 'method' key."""
    est = ZatsiorskyEstimator()
    assert est.method_name == "zatsiorsky_seluyanov_1985"


def test_estimator_exposes_citation() -> None:
    """The ``citation`` property must come from the JSON 'citation' key."""
    est = ZatsiorskyEstimator()
    assert "Zatsiorsky" in est.citation
    assert "1985" in est.citation


# --------------------------------------------------------------------------- #
# Smoke: full subject build + mass closure within 1%.                         #
# --------------------------------------------------------------------------- #
def test_zatsiorsky_smoke_full_subject_returns_subject_anthropometrics() -> None:
    """1.83 m / 82 kg subject produces a valid SubjectAnthropometrics."""
    est = ZatsiorskyEstimator()
    subject = est.estimate(
        subject_id="zat-smoke",
        height_m=1.83,
        mass_kg=82.0,
    )
    assert isinstance(subject, SubjectAnthropometrics)
    assert subject.source_method == "zatsiorsky_seluyanov_1985"
    assert len(subject.segments) >= 8


def test_zatsiorsky_smoke_mass_closes_within_one_percent() -> None:
    """Sum of segment masses equals total subject mass within 1%."""
    est = ZatsiorskyEstimator()
    mass_kg = 82.0
    subject = est.estimate(
        subject_id="closure",
        height_m=1.83,
        mass_kg=mass_kg,
    )
    total = sum(props.mass_kg for _, props in subject.segments)
    assert abs(total - mass_kg) / mass_kg <= 0.01


def test_zatsiorsky_estimator_protocol_isinstance_check() -> None:
    """The estimator must satisfy the runtime-checkable Estimator Protocol."""
    assert isinstance(ZatsiorskyEstimator(), Estimator)


def test_inertia_tensors_are_diagonal_and_valid() -> None:
    """Every segment's inertia tensor is diagonal and physically valid."""
    est = ZatsiorskyEstimator()
    subject = est.estimate(
        subject_id="inertia",
        height_m=1.78,
        mass_kg=75.0,
    )
    for _name, props in subject.segments:
        tensor = props.inertia_tensor
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert tensor[i, j] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Design by Contract.                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_height", [0.0, -1.0, float("nan"), float("inf")])
def test_zatsiorsky_invalid_height_raises_value_error(bad_height: float) -> None:
    """``height_m`` must be a positive finite number."""
    est = ZatsiorskyEstimator()
    with pytest.raises(ValueError, match="height_m"):
        est.estimate(subject_id="x", height_m=bad_height, mass_kg=70.0)


@pytest.mark.parametrize("bad_mass", [0.0, -50.0, float("nan"), float("inf")])
def test_zatsiorsky_invalid_mass_raises_value_error(bad_mass: float) -> None:
    """``mass_kg`` must be a positive finite number."""
    est = ZatsiorskyEstimator()
    with pytest.raises(ValueError, match="mass_kg"):
        est.estimate(subject_id="x", height_m=1.80, mass_kg=bad_mass)


def test_zatsiorsky_invalid_subject_id_raises_value_error() -> None:
    """``subject_id`` must be a non-empty string."""
    est = ZatsiorskyEstimator()
    with pytest.raises(ValueError, match="subject_id"):
        est.estimate(subject_id="", height_m=1.80, mass_kg=70.0)


def test_invalid_age_raises_value_error() -> None:
    """``age_years`` if provided must be non-negative."""
    est = ZatsiorskyEstimator()
    with pytest.raises(ValueError, match="age_years"):
        est.estimate(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            age_years=-1.0,
        )


# --------------------------------------------------------------------------- #
# Custom ratio file — supports calibration overrides without code changes.    #
# --------------------------------------------------------------------------- #
def test_custom_ratio_file_overrides_default(tmp_path: Path) -> None:
    """A user can pass a custom ratio file via the constructor."""
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "method": "custom_method",
                "citation": "test fixture",
                "segments": {
                    "torso": {
                        "mass_ratio": 1.0,
                        "length_ratio": 0.5,
                        "com_proximal_ratio": 0.5,
                        "gyration_radii": {
                            "sagittal": 0.3,
                            "transverse": 0.3,
                            "longitudinal": 0.2,
                        },
                    }
                },
                "segment_name_map": {"torso": "torso"},
            }
        ),
        encoding="utf-8",
    )
    est = ZatsiorskyEstimator(ratio_file=custom)
    assert est.method_name == "custom_method"
    subject = est.estimate(subject_id="x", height_m=1.80, mass_kg=70.0)
    assert len(subject.segments) == 1
    assert subject.segments[0][0] == "torso"
    assert subject.segments[0][1].mass_kg == pytest.approx(70.0, rel=1e-9)
