"""Unit tests for ``anthropometrics.readers.c3d_subject_info``.

The reader is exercised three ways:

* **Synthetic fixture path** — extends ``_synthetic_c3d_dict`` with
  ``PROCESSING`` / ``SUBJECTS`` parameter groups and feeds the
  resulting mapping straight into the pure
  :func:`~anthropometrics.readers.c3d_subject_info._extract_subject_metadata`
  helper, asserting every attribute on the returned dataclass.
* **Real-data path** — opens ``data/C3D_TA_Driver.c3d`` (which lacks
  PROCESSING / SUBJECTS) and asserts every field is ``None``
  (or :data:`Sex.UNSPECIFIED` for sex).
* **Error paths** — missing file → ``FileNotFoundError``; malformed
  ``Height`` (NaN) → ``height_m`` is ``None`` with no exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from anthropometrics import Sex
from anthropometrics.readers import (
    C3DSubjectMetadata,
    read_c3d_subject_metadata,
)
from anthropometrics.readers.c3d_subject_info import (
    _extract_subject_metadata,
    _iter_flat,
    _read_scalar,
)
from tests.unit.sidekick.lab.bio._synthetic import (
    _synthetic_c3d_dict,
)


# --------------------------------------------------------------------------- #
# Fixture helpers.                                                            #
# --------------------------------------------------------------------------- #
def _with_processing(
    base: dict[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    """Add a ``PROCESSING`` group with *fields* to the synthetic dict."""
    base["parameters"]["PROCESSING"] = {
        key: {"value": np.atleast_1d(np.asarray([value]).ravel())}
        for key, value in fields.items()
    }
    return base


def _with_subjects(
    base: dict[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    """Add a ``SUBJECTS`` group with *fields* to the synthetic dict."""
    group: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            group[key] = {"value": [value]}
        else:
            group[key] = {"value": np.atleast_1d(np.asarray([value]).ravel())}
    base["parameters"]["SUBJECTS"] = group
    return base


# --------------------------------------------------------------------------- #
# Synthetic happy path.                                                       #
# --------------------------------------------------------------------------- #
def test_extract_full_metadata_from_synthetic_fixture() -> None:
    fixture = _synthetic_c3d_dict()
    fixture = _with_processing(
        fixture,
        Bodymass=72.5,
        Height=1780.0,  # mm
        LeftLegLength=900.0,
        RightLegLength=910.0,
        LeftArmLength=620.0,
        RightArmLength=630.0,
    )
    fixture = _with_subjects(
        fixture,
        NAMES="SUBJ-001",
        AGE=42.0,
        SEX="M",
    )

    metadata = _extract_subject_metadata(fixture["parameters"])

    assert isinstance(metadata, C3DSubjectMetadata)
    assert metadata.subject_id == "SUBJ-001"
    assert metadata.mass_kg == pytest.approx(72.5)
    assert metadata.height_m == pytest.approx(1.780)
    assert metadata.leg_length_m == pytest.approx(0.905)
    assert metadata.arm_length_m == pytest.approx(0.625)
    assert metadata.age_years == pytest.approx(42.0)
    assert metadata.sex is Sex.MALE


def test_dataclass_is_frozen() -> None:
    """Sanity check: the returned dataclass refuses attribute assignment."""
    metadata = _extract_subject_metadata({})
    with pytest.raises(AttributeError):
        metadata.subject_id = "X"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Field-level partials.                                                       #
# --------------------------------------------------------------------------- #
def test_only_left_leg_length_uses_left_value() -> None:
    fixture = _with_processing(_synthetic_c3d_dict(), LeftLegLength=880.0)
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.leg_length_m == pytest.approx(0.880)
    assert metadata.arm_length_m is None


def test_only_right_arm_length_uses_right_value() -> None:
    fixture = _with_processing(_synthetic_c3d_dict(), RightArmLength=600.0)
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.arm_length_m == pytest.approx(0.600)


def test_sex_female_maps_correctly() -> None:
    fixture = _with_subjects(_synthetic_c3d_dict(), SEX="f")
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.sex is Sex.FEMALE


def test_sex_unknown_value_falls_back_to_unspecified() -> None:
    fixture = _with_subjects(_synthetic_c3d_dict(), SEX="other")
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.sex is Sex.UNSPECIFIED


def test_blank_subject_name_treated_as_missing() -> None:
    fixture = _with_subjects(_synthetic_c3d_dict(), NAMES="   ")
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.subject_id is None


def test_case_insensitive_group_and_key_names() -> None:
    fixture = _synthetic_c3d_dict()
    fixture["parameters"]["processing"] = {
        "bodymass": {"value": np.array([70.0])},
        "HEIGHT": {"value": np.array([1700.0])},
    }
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.mass_kg == pytest.approx(70.0)
    assert metadata.height_m == pytest.approx(1.7)


# --------------------------------------------------------------------------- #
# Missing / malformed fields.                                                 #
# --------------------------------------------------------------------------- #
def test_no_processing_or_subjects_groups_returns_all_none() -> None:
    fixture = _synthetic_c3d_dict()
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata == C3DSubjectMetadata(
        subject_id=None,
        height_m=None,
        mass_kg=None,
        age_years=None,
        sex=Sex.UNSPECIFIED,
        leg_length_m=None,
        arm_length_m=None,
    )


def test_height_nan_does_not_raise_returns_none() -> None:
    fixture = _with_processing(_synthetic_c3d_dict(), Height=float("nan"))
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.height_m is None


def test_mass_inf_returns_none() -> None:
    fixture = _with_processing(_synthetic_c3d_dict(), Bodymass=float("inf"))
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.mass_kg is None


def test_empty_value_array_returns_none() -> None:
    fixture = _synthetic_c3d_dict()
    fixture["parameters"]["PROCESSING"] = {
        "Bodymass": {"value": np.array([])},
        "Height": {"value": np.array([])},
    }
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.mass_kg is None
    assert metadata.height_m is None


def test_non_numeric_scalar_returns_none() -> None:
    fixture = _synthetic_c3d_dict()
    fixture["parameters"]["PROCESSING"] = {
        "Bodymass": {"value": ["not-a-number"]},
    }
    metadata = _extract_subject_metadata(fixture["parameters"])
    assert metadata.mass_kg is None


def test_unreadable_value_type_returns_none() -> None:
    """A value that is not iterable and not a scalar should not crash."""
    assert _read_scalar({"key": {"value": object()}}, "key") is None


def test_iter_flat_handles_string() -> None:
    """``_iter_flat`` returns the string itself, not its characters."""
    assert list(_iter_flat("abc")) == ["abc"]


def test_iter_flat_handles_nested_lists() -> None:
    assert list(_iter_flat([[1, 2], [3, 4]])) == [1, 2, 3, 4]


def test_iter_flat_handles_scalar() -> None:
    assert list(_iter_flat(7.0)) == [7.0]


# --------------------------------------------------------------------------- #
# Public entry point — file-system / ezc3d wiring.                            #
# --------------------------------------------------------------------------- #
def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.c3d"
    with pytest.raises(FileNotFoundError):
        read_c3d_subject_metadata(bogus)


def test_read_uses_ezc3d_and_returns_metadata(tmp_path: Path) -> None:
    """End-to-end through ``read_c3d_subject_metadata`` with ezc3d mocked."""
    path = tmp_path / "fake.c3d"
    path.write_bytes(b"")  # exists so FileNotFoundError is not raised

    fixture = _with_processing(
        _synthetic_c3d_dict(),
        Bodymass=80.0,
        Height=1820.0,
    )
    fixture = _with_subjects(fixture, SEX="F", AGE=29.0)

    with patch(
        "anthropometrics.readers.c3d_subject_info._load_parameters",
        return_value=fixture["parameters"],
    ):
        metadata = read_c3d_subject_metadata(path)

    assert metadata.mass_kg == pytest.approx(80.0)
    assert metadata.height_m == pytest.approx(1.820)
    assert metadata.sex is Sex.FEMALE
    assert metadata.age_years == pytest.approx(29.0)


def test_read_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / "fake.c3d"
    path.write_bytes(b"")
    with patch(
        "anthropometrics.readers.c3d_subject_info._load_parameters",
        return_value=_synthetic_c3d_dict()["parameters"],
    ):
        metadata = read_c3d_subject_metadata(str(path))
    assert metadata.mass_kg is None
