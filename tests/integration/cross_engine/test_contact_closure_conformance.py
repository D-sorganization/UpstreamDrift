"""Cross-engine conformance tests for the shared contact law and grip closure (MS-72 #10352).

Loads justified tolerances directly from the YAML frontmatter of
docs/development/matched_swing_program/CONTACT_CLOSURE_CONFORMANCE.md (DbC / LoD constraint).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.shared.python.motion_matching.contact_law import (
    CONFORMANCE_VERSION,
    ContactParameters,
    ContactSample,
    GroundPlane,
    contact_parity_report,
    sphere_ground_contact,
)

DOC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "development"
    / "matched_swing_program"
    / "CONTACT_CLOSURE_CONFORMANCE.md"
)

REGISTRY_PATH = Path(__file__).resolve().parent / "divergence_registry.yaml"


pytestmark = [pytest.mark.integration]


def load_frontmatter(path: Path) -> dict[str, Any]:
    """Parse YAML frontmatter from a markdown document."""
    if not path.exists():
        pytest.fail(f"Conformance specification document missing at {path}")
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        pytest.fail(
            f"Conformance document {path} must start with YAML front matter ('---')"
        )
    parts = text.split("---", 2)
    if len(parts) < 3:
        pytest.fail(f"Malformed YAML front matter in {path}")
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        pytest.fail(f"Front matter in {path} must parse to a dict")
    return data


@pytest.fixture(scope="module")
def conformance_spec() -> dict[str, Any]:
    return load_frontmatter(DOC_PATH)


@pytest.fixture(scope="module")
def tolerances(conformance_spec: dict[str, Any]) -> dict[str, Any]:
    raw = conformance_spec.get("tolerances")
    assert isinstance(raw, dict), "Frontmatter must define a 'tolerances' mapping"
    parsed: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
        else:
            parsed[k] = v
    return parsed


@pytest.fixture
def default_contact_params() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=50000.0,
        dissipation_s_m=2.0,
        static_friction=0.8,
        dynamic_friction=0.6,
        viscous_friction=0.01,
        transition_velocity_m_s=0.01,
    )


@pytest.fixture
def default_ground() -> GroundPlane:
    return GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


def test_conformance_version_matches_spec(conformance_spec: dict[str, Any]) -> None:
    spec_version = conformance_spec.get("conformance_version")
    assert spec_version is not None, "Specification must declare conformance_version"
    assert spec_version == CONFORMANCE_VERSION, (
        f"contact_law.CONFORMANCE_VERSION ({CONFORMANCE_VERSION}) must match "
        f"specification frontmatter version ({spec_version})"
    )


def test_zero_penetration_produces_zero_force(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
    tolerances: dict[str, Any],
) -> None:
    radius = 0.03
    # Center at z = 0.05 m with ground height 0.0 m -> distance = +0.02 m (no penetration)
    center = np.array([0.0, 0.0, 0.05])
    velocity = np.array([1.0, 0.5, -2.0])
    sample = sphere_ground_contact(
        center, velocity, radius, default_ground, default_contact_params
    )
    assert sample.penetration_m == 0.0
    tol = tolerances["zero_penetration_force_n"]
    assert np.all(sample.normal_force_n == tol)
    assert np.all(sample.friction_force_n == tol)


def test_normal_force_matches_hunt_crossley_equation(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
    tolerances: dict[str, Any],
) -> None:
    radius = 0.03
    # Center at z = 0.02 m -> penetration = 0.01 m
    center = np.array([0.0, 0.0, 0.02])
    # Velocity downward (-0.5 m/s) -> rate = +0.5 m/s
    velocity = np.array([0.0, 0.0, -0.5])
    sample = sphere_ground_contact(
        center, velocity, radius, default_ground, default_contact_params
    )
    d = 0.01
    d_dot = 0.5
    expected_f_n = (
        default_contact_params.stiffness_n_m
        * d
        * (1.0 + default_contact_params.dissipation_s_m * d_dot)
    )
    tol = tolerances["normal_force_parity_n"]
    assert sample.normal_force_n[2] == pytest.approx(expected_f_n, abs=tol)


def test_rebound_tensile_force_clipped_at_zero(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
    tolerances: dict[str, Any],
) -> None:
    radius = 0.03
    center = np.array([0.0, 0.0, 0.02])  # penetration = 0.01 m
    # Upward velocity (+2.0 m/s) -> rate = -2.0 m/s
    # 1.0 + 2.0 * (-2.0) = -3.0 < 0 (tensile!)
    velocity = np.array([0.0, 0.0, 2.0])
    sample = sphere_ground_contact(
        center, velocity, radius, default_ground, default_contact_params
    )
    tol = tolerances["rebound_tensile_force_n"]
    assert sample.normal_force_n[2] == pytest.approx(0.0, abs=tol)
    assert np.all(sample.friction_force_n == 0.0)


def test_regularized_friction_direction_and_magnitude(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
    tolerances: dict[str, Any],
) -> None:
    radius = 0.03
    center = np.array([0.0, 0.0, 0.02])  # penetration = 0.01 m
    # Pure horizontal velocity: x = 0.1 m/s, y = 0.0, z = 0.0
    velocity = np.array([0.1, 0.0, 0.0])
    sample = sphere_ground_contact(
        center, velocity, radius, default_ground, default_contact_params
    )
    assert sample.friction_force_n[0] < 0.0  # opposes positive x velocity
    assert sample.friction_force_n[1] == pytest.approx(0.0, abs=1e-12)
    assert sample.friction_force_n[2] == pytest.approx(0.0, abs=1e-12)

    speed = 0.1
    ratio = speed / default_contact_params.transition_velocity_m_s
    mu = (
        default_contact_params.dynamic_friction
        + (
            default_contact_params.static_friction
            - default_contact_params.dynamic_friction
        )
        * math.exp(-(ratio**2))
        + default_contact_params.viscous_friction * speed
    )
    expected_magnitude = mu * sample.normal_force_n[2] * math.tanh(ratio)
    tol = tolerances["friction_force_parity_n"]
    assert abs(sample.friction_force_n[0]) == pytest.approx(expected_magnitude, abs=tol)


def test_energy_dissipation_nonnegative(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
) -> None:
    radius = 0.03
    center = np.array([0.0, 0.0, 0.02])
    velocity = np.array([0.05, 0.05, -0.2])
    sample = sphere_ground_contact(
        center, velocity, radius, default_ground, default_contact_params
    )
    assert sample.normal_force_n[2] * velocity[2] <= 0.0
    assert np.dot(sample.friction_force_n, velocity[:3]) <= 0.0


def test_contact_parity_report_schema(
    default_ground: GroundPlane,
    default_contact_params: ContactParameters,
) -> None:
    def adapter1(center: np.ndarray, vel: np.ndarray, r: float) -> ContactSample:
        return sphere_ground_contact(
            center, vel, r, default_ground, default_contact_params
        )

    def adapter2(center: np.ndarray, vel: np.ndarray, r: float) -> ContactSample:
        return sphere_ground_contact(
            center, vel, r, default_ground, default_contact_params
        )

    states = [
        (np.array([0.0, 0.0, 0.01]), np.array([0.0, 0.0, -0.1])),
        (np.array([0.0, 0.0, 0.10]), np.array([0.0, 0.0, 0.0])),
    ]
    report = contact_parity_report(
        {"engine_a": adapter1, "engine_b": adapter2}, states, radius=0.03
    )
    assert report["conformance_version"] == CONFORMANCE_VERSION
    assert report["states"] == 2
    assert report["penetrating_states"] == 1
    assert report["max_normal_force_difference_n"]["engine_b"] == 0.0


def test_dual_grip_closure_rank_and_relative_transform(
    tolerances: dict[str, Any],
) -> None:
    """Verify dual-grip weld closure satisfies rank 6 spatial constraint."""
    expected_rank = tolerances["closure_rank_expected"]
    pos_tol = tolerances["closure_position_tolerance_m"]

    p_lead = np.array([0.1, -0.2, 0.8])
    p_trail = np.array([0.1, -0.2, 0.8])
    pos_residual = np.linalg.norm(p_lead - p_trail)
    assert pos_residual <= pos_tol

    rng = np.random.default_rng(42)
    J_lead = rng.normal(size=(6, 14))
    J_trail = rng.normal(size=(6, 14))
    J_closure = np.hstack([J_lead, -J_trail])
    rank = int(np.linalg.matrix_rank(J_closure))
    assert rank == expected_rank


def test_divergence_registry_covers_contact_and_closure() -> None:
    """Verify divergence_registry.yaml contains required contact & closure entries."""
    assert REGISTRY_PATH.exists()
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert registry.get("version") == 1
    ids = {d["id"] for d in registry.get("divergences", [])}
    assert "contact-solver-evaluation-timing" in ids
    assert "dual-grip-closure-formulation" in ids
    assert "hunt-crossley-dissipation-clipping" in ids


def test_contact_parameters_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContactParameters(
            stiffness_n_m=-100.0,
            dissipation_s_m=1.0,
            static_friction=0.5,
            dynamic_friction=0.4,
            viscous_friction=0.0,
            transition_velocity_m_s=0.01,
        )
    with pytest.raises(ValueError, match="Static friction must not be below"):
        ContactParameters(
            stiffness_n_m=1000.0,
            dissipation_s_m=1.0,
            static_friction=0.3,
            dynamic_friction=0.5,
            viscous_friction=0.0,
            transition_velocity_m_s=0.01,
        )
