from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anthropometrics import (
    SegmentProperties,
    SubjectAnthropometrics,
    build_inertia_priors_from_subject,
    load_addbiomechanics_inertia_priors,
    save_inertia_priors,
)


def _subject() -> SubjectAnthropometrics:
    pelvis = SegmentProperties(
        name="pelvis",
        body_part_id="pelvis",
        length_m=0.30,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=10.0,
        com_xyz_m=np.array([0.02, 0.0, 0.0]),
        inertia_tensor=np.array(
            [
                [0.110, 0.004, -0.002],
                [0.004, 0.120, 0.003],
                [-0.002, 0.003, 0.100],
            ]
        ),
        source_method="addbiomechanics_force_plate",
        source_subject_height_m=1.80,
        source_subject_mass_kg=76.0,
    )
    thigh = SegmentProperties(
        name="thigh_r",
        body_part_id="thigh_r",
        length_m=0.44,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=8.5,
        com_xyz_m=np.array([0.0, -0.20, 0.0]),
        inertia_tensor=np.diag([0.070, 0.068, 0.020]),
        source_method="addbiomechanics_force_plate",
        source_subject_height_m=1.80,
        source_subject_mass_kg=76.0,
    )
    return SubjectAnthropometrics(
        subject_id="athlete-7",
        height_m=1.80,
        mass_kg=76.0,
        segments=(("pelvis", pelvis), ("thigh_r", thigh)),
        source_method="addbiomechanics_force_plate",
        sex="M",
    )


def test_builds_deterministic_bounded_estimator_specs() -> None:
    priors = build_inertia_priors_from_subject(
        _subject(),
        source_session_id="addbio-session-42",
        correction_fraction=0.05,
        prior_scale_fraction=0.01,
    )

    assert [param.name for param in priors.parameters[:6]] == [
        "theta_prior.pelvis.inertia.ixx",
        "theta_prior.pelvis.inertia.iyy",
        "theta_prior.pelvis.inertia.izz",
        "theta_prior.pelvis.inertia.ixy",
        "theta_prior.pelvis.inertia.ixz",
        "theta_prior.pelvis.inertia.iyz",
    ]
    ixx = priors.parameters[0]
    assert ixx.prior == pytest.approx(0.110)
    assert ixx.lower == pytest.approx(0.1045)
    assert ixx.upper == pytest.approx(0.1155)
    assert ixx.prior_scale == pytest.approx(0.00110)

    block_payload = priors.to_estimator_parameter_block_payload()
    assert block_payload["parameters"][0] == {
        "name": "theta_prior.pelvis.inertia.ixx",
        "initial": pytest.approx(0.110),
        "kind": "inertia",
        "lower": pytest.approx(0.1045),
        "upper": pytest.approx(0.1155),
        "prior": pytest.approx(0.110),
        "prior_scale": pytest.approx(0.00110),
        "locked": False,
    }


def test_prior_set_serializes_and_loads_without_drift(tmp_path: Path) -> None:
    priors = build_inertia_priors_from_subject(
        _subject(),
        source_session_id="addbio-session-42",
    )
    path = tmp_path / "priors.json"

    save_inertia_priors(priors, path)
    loaded = load_addbiomechanics_inertia_priors(path)

    assert loaded.to_dict() == priors.to_dict()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["parameters"][0]["source_session_id"] == "addbio-session-42"


def test_loads_addbiomechanics_scaled_segment_export(tmp_path: Path) -> None:
    payload = {
        "source": "AddBiomechanics",
        "session_id": "forceplate-2026-05-31",
        "subject": {
            "subject_id": "athlete-9",
            "height_m": 1.74,
            "mass_kg": 68.0,
            "sex": "F",
        },
        "segments": [
            {
                "name": "torso",
                "body_part_id": "torso",
                "length_m": 0.55,
                "mass_kg": 22.0,
                "com_xyz_m": [0.0, 0.12, 0.0],
                "inertia_tensor": [
                    [0.40, 0.01, 0.00],
                    [0.01, 0.35, 0.02],
                    [0.00, 0.02, 0.30],
                ],
            }
        ],
    }
    path = tmp_path / "addbio.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    priors = load_addbiomechanics_inertia_priors(
        path,
        correction_fraction=0.10,
        prior_scale_fraction=0.02,
    )

    assert priors.subject_id == "athlete-9"
    assert priors.source_session_id == "forceplate-2026-05-31"
    assert len(priors.parameters) == 6
    assert priors.parameters[1].name == "theta_prior.torso.inertia.iyy"
    assert priors.parameters[1].prior == pytest.approx(0.35)


def test_rejects_non_physical_addbiomechanics_inertia(tmp_path: Path) -> None:
    payload = {
        "session_id": "bad-forceplate",
        "subject": {"subject_id": "bad", "height_m": 1.80, "mass_kg": 75.0},
        "segments": [
            {
                "name": "pelvis",
                "mass_kg": 10.0,
                "inertia_tensor": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -0.1],
                ],
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="positive-definite"):
        load_addbiomechanics_inertia_priors(path)
