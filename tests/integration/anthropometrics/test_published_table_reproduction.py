"""Issue #4819 — reproduce every published ratio in every estimator table.

Each of the three estimators (de Leva 1996, Dempster 1955,
Zatsiorsky-Seluyanov 1985) ships with a published ratio table. The
existing :mod:`test_estimator_validation` exercise spot-checks a few
segments. This file is the **comprehensive sweep** — it walks every
class in every published table and asserts the four canonical ratios
``(mass_ratio, length_ratio, com_proximal_ratio, gyration_radii)`` are
recovered from the materialised :class:`SegmentProperties` within
``1e-3`` of the source-of-truth constants.

Key implementation note: all three estimators normalise the per-segment
mass ratio so the per-subject sum equals exactly the input
``mass_kg``. The published mass ratio for one class is therefore
recovered as ``mass_kg_segment / (mass_kg_subject * mass_scale)`` where
``mass_scale = 1 / sum(raw_mass_ratios over segment_name_map)``. This
file applies that correction inline so the integration check matches
the exact published numbers, not the renormalised numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from anthropometrics import SubjectAnthropometrics
from anthropometrics.estimators import (
    DeLevaEstimator,
    DempsterEstimator,
    ZatsiorskyEstimator,
)
from humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    _SEGMENT_NAME_MAP,
)

# 1e-3 absolute tolerance per the issue acceptance criterion.
_PUBLISHED_TOL: float = 1.0e-3

_RATIOS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "shared"
    / "python"
    / "anthropometrics"
    / "estimators"
    / "ratios"
)
_DEMPSTER_JSON = _RATIOS_DIR / "dempster_1955.json"
_ZATSIORSKY_JSON = _RATIOS_DIR / "zatsiorsky_seluyanov_1985.json"


@dataclass(frozen=True)
class _PublishedRatios:
    """Published source-of-truth ratios for one class in one table."""

    mass_ratio: float
    length_ratio: float
    com_proximal_ratio: float
    gyration_sagittal: float
    gyration_transverse: float
    gyration_longitudinal: float


def _load_json_published(
    path: Path,
) -> tuple[dict[str, _PublishedRatios], dict[str, str]]:
    """Return ``(class_id -> ratios, anatomical -> class_id)`` from a JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    classes: dict[str, _PublishedRatios] = {}
    for class_id, raw in data["segments"].items():
        gyr = raw["gyration_radii"]
        classes[class_id] = _PublishedRatios(
            mass_ratio=float(raw["mass_ratio"]),
            length_ratio=float(raw["length_ratio"]),
            com_proximal_ratio=float(raw["com_proximal_ratio"]),
            gyration_sagittal=float(gyr["sagittal"]),
            gyration_transverse=float(gyr["transverse"]),
            gyration_longitudinal=float(gyr["longitudinal"]),
        )
    name_map = {str(k): str(v) for k, v in data["segment_name_map"].items()}
    return classes, name_map


def _de_leva_published(sex: str) -> tuple[dict[str, _PublishedRatios], dict[str, str]]:
    """Return ``(class_id -> ratios, anatomical -> class_id)`` for de Leva."""
    table = DE_LEVA_DATA.female if sex == "F" else DE_LEVA_DATA.male
    classes: dict[str, _PublishedRatios] = {}
    for class_id, seg in table.items():
        classes[class_id] = _PublishedRatios(
            mass_ratio=seg.mass_ratio,
            length_ratio=seg.length_ratio,
            com_proximal_ratio=seg.com_proximal_ratio,
            gyration_sagittal=seg.gyration_sagittal,
            gyration_transverse=seg.gyration_transverse,
            gyration_longitudinal=seg.gyration_longitudinal,
        )
    return classes, dict(_SEGMENT_NAME_MAP)


@dataclass(frozen=True)
class _Spec:
    """One estimator scenario: estimator factory + published table + canonical subject."""

    label: str
    factory: Callable[[], object]
    height_m: float
    mass_kg: float
    sex: str
    classes: Mapping[str, _PublishedRatios]
    name_map: Mapping[str, str]


def _build_specs() -> list[_Spec]:
    """Return one :class:`_Spec` per published table covered by the issue."""
    de_leva_male_classes, de_leva_male_map = _de_leva_published("M")
    de_leva_female_classes, de_leva_female_map = _de_leva_published("F")
    dempster_classes, dempster_map = _load_json_published(_DEMPSTER_JSON)
    zats_classes, zats_map = _load_json_published(_ZATSIORSKY_JSON)
    return [
        _Spec(
            label="de_leva-male",
            factory=DeLevaEstimator,
            height_m=1.78,
            mass_kg=75.0,
            sex="M",
            classes=de_leva_male_classes,
            name_map=de_leva_male_map,
        ),
        _Spec(
            label="de_leva-female",
            factory=DeLevaEstimator,
            height_m=1.65,
            mass_kg=60.0,
            sex="F",
            classes=de_leva_female_classes,
            name_map=de_leva_female_map,
        ),
        _Spec(
            label="dempster",
            factory=DempsterEstimator,
            height_m=1.78,
            mass_kg=75.0,
            sex="M",
            classes=dempster_classes,
            name_map=dempster_map,
        ),
        _Spec(
            label="zatsiorsky-male",
            factory=ZatsiorskyEstimator,
            height_m=1.78,
            mass_kg=75.0,
            sex="M",
            classes=zats_classes,
            name_map=zats_map,
        ),
    ]


_SPECS: list[_Spec] = _build_specs()


def _mass_scale(spec: _Spec) -> float:
    """Return the mass-renormalisation factor applied by the estimator.

    The estimator scales every mass_ratio by ``1 / sum(raw)`` over the
    *segment_name_map* keys (not the unique class set) so per-subject
    mass closure is exact. To recover the original published ratio we
    divide the materialised ratio back through the same factor.
    """
    raw_sum = sum(
        spec.classes[class_id].mass_ratio for class_id in spec.name_map.values()
    )
    if raw_sum <= 0:
        raise AssertionError(f"raw_ratio_sum is non-positive for {spec.label}")
    return 1.0 / raw_sum


_PARAMS: list[tuple[str, str, _Spec, _PublishedRatios]] = []
for _spec in _SPECS:
    # Sweep every (anatomical, class_id) pair. One published row may be
    # exercised under several anatomical names (e.g. left/right); each
    # gets its own parametrise id so failures pinpoint exactly which
    # name disagreed.
    for _anat_name, _class_id in _spec.name_map.items():
        _PARAMS.append((_spec.label, _anat_name, _spec, _spec.classes[_class_id]))


def _id(param: tuple[str, str, _Spec, _PublishedRatios]) -> str:
    spec_label, anat_name, _spec, _ratios = param
    return f"{spec_label}::{anat_name}"


@pytest.fixture(scope="module")
def _records() -> dict[str, SubjectAnthropometrics]:
    """One materialised :class:`SubjectAnthropometrics` per spec, cached."""
    cache: dict[str, SubjectAnthropometrics] = {}
    for spec in _SPECS:
        estimator = spec.factory()
        cache[spec.label] = estimator.estimate(  # type: ignore[attr-defined]
            subject_id=f"published-{spec.label}",
            height_m=spec.height_m,
            mass_kg=spec.mass_kg,
            sex=spec.sex,
        )
    return cache


@pytest.mark.parametrize(
    ("spec_label", "anatomical", "spec", "published"),
    _PARAMS,
    ids=[_id(p) for p in _PARAMS],
)
def test_published_segment_reproduces_within_1e_minus_3(
    spec_label: str,
    anatomical: str,
    spec: _Spec,
    published: _PublishedRatios,
    _records: dict[str, SubjectAnthropometrics],
) -> None:
    """Recover every (mass, length, CoM, gyration) ratio at 1e-3 tolerance.

    The four published quantities are recovered from the
    :class:`SegmentProperties` as follows:

    * ``length_ratio = length_m / height_m``
    * ``com_proximal_ratio = com_xyz_m[2] / length_m``
    * ``gyration_i = sqrt(I_ii / mass) / length_m``
    * ``mass_ratio = mass_kg / (subject_mass_kg * mass_scale)``
      (mass_scale undoes the per-subject mass normalisation in the
      estimator driver — see module docstring).
    """
    record = _records[spec_label]
    seg = dict(record.segments)[anatomical]

    # length_ratio
    assert seg.length_m / spec.height_m == pytest.approx(
        published.length_ratio, abs=_PUBLISHED_TOL
    )
    # com_proximal_ratio (only the longitudinal axis carries CoM offset)
    assert seg.com_xyz_m[2] / seg.length_m == pytest.approx(
        published.com_proximal_ratio, abs=_PUBLISHED_TOL
    )
    # mass_ratio (after un-normalising)
    mass_scale = _mass_scale(spec)
    raw_mass_ratio = seg.mass_kg / (spec.mass_kg * mass_scale)
    assert raw_mass_ratio == pytest.approx(published.mass_ratio, abs=_PUBLISHED_TOL)
    # gyration radii: sqrt(I / m) / L. The estimator places ix on the
    # sagittal axis, iy on transverse, iz on longitudinal (see
    # ``_segment_properties_from_ratios``), and the inertia tensor is
    # diagonal in this frame.
    inertia = np.asarray(seg.inertia_tensor)
    gyr_sag = float(np.sqrt(inertia[0, 0] / seg.mass_kg) / seg.length_m)
    gyr_tr = float(np.sqrt(inertia[1, 1] / seg.mass_kg) / seg.length_m)
    gyr_long = float(np.sqrt(inertia[2, 2] / seg.mass_kg) / seg.length_m)
    assert gyr_sag == pytest.approx(published.gyration_sagittal, abs=_PUBLISHED_TOL)
    assert gyr_tr == pytest.approx(published.gyration_transverse, abs=_PUBLISHED_TOL)
    assert gyr_long == pytest.approx(
        published.gyration_longitudinal, abs=_PUBLISHED_TOL
    )


@pytest.mark.parametrize("spec_label", [s.label for s in _SPECS])
def test_subject_mass_closure_to_one_percent(
    spec_label: str, _records: dict[str, SubjectAnthropometrics]
) -> None:
    """Sum of segment masses equals subject mass within 1 percent."""
    record = _records[spec_label]
    total = sum(float(p.mass_kg) for _, p in record.segments)
    assert total == pytest.approx(record.mass_kg, rel=1.0e-2)


@pytest.mark.parametrize("spec_label", [s.label for s in _SPECS])
def test_every_segment_inertia_is_positive_definite(
    spec_label: str, _records: dict[str, SubjectAnthropometrics]
) -> None:
    """Every materialised inertia tensor has strictly positive eigenvalues."""
    record = _records[spec_label]
    for name, props in record.segments:
        eigs = np.linalg.eigvalsh(np.asarray(props.inertia_tensor))
        assert np.all(eigs > 0), f"non-PD inertia on {spec_label}/{name}: {eigs}"
