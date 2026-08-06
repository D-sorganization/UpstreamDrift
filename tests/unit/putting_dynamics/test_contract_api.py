"""Public-facade contract test for putting_dynamics (#8345).

Pins the package's importable surface: downstream consumers (the P1
R3F visualization API and any future Tools promotion via the vendor
pattern) rely on these names resolving from the package root.
"""

from __future__ import annotations

import pytest

import src.shared.python.putting_dynamics as pd

pytestmark = pytest.mark.unit

_EXPECTED_EXPORTS = {
    # constants
    "DEFAULT_SLIDING_MU",
    "DEFAULT_STATIC_MU",
    "DT_S",
    "GROUND_RESTITUTION",
    "HOLE_RADIUS_M",
    "LOFT_SWEEP_MAX_DEG",
    "LOFT_SWEEP_MIN_DEG",
    "STIMP_RELEASE_SPEED_MPS",
    # types
    "BallState",
    "CollisionReport",
    "FrictionField",
    "FrictionParams",
    "HeightField",
    "Mode",
    "PuttResult",
    "PutterState",
    "SurfaceSpec",
    "TrajectorySample",
    # functions
    "ball_kinetic_energy_j",
    "bumpy_friction_field",
    "bumpy_height_field",
    "capture_speed_mps",
    "effective_head_mass",
    "energy_balance_error_j",
    "grain_factor",
    "is_static_hold",
    "m_to_feet",
    "m_to_yards",
    "mps_to_mph",
    "rolling_mu",
    "rolling_mu_to_stimp",
    "simulate_ball",
    "simulate_strike",
    "sliding_mu",
    "stimp_to_rolling_mu",
    "strike",
    "sweep_dynamic_loft",
}


def test_facade_exports_are_pinned() -> None:
    assert set(pd.__all__) == _EXPECTED_EXPORTS


def test_every_export_resolves() -> None:
    for name in pd.__all__:
        assert getattr(pd, name) is not None


def test_collision_report_fields_needed_by_p1_visualization() -> None:
    # The P1 scene consumes these exact fields (epic #8345 P1/P3).
    fields = {
        "ball_speed_mps",
        "launch_angle_deg",
        "horizontal_speed_mps",
        "vertical_speed_mps",
        "spin_rad_s",
        "effective_loft_deg",
        "putter_dv_mps",
        "impulse_n_s",
        "contact_time_proxy_s",
        "kinetic_energy_loss_j",
        "face_twist_rad_s",
        "twist_moment_n_m_s",
        "attachment_impulse_n_s",
        "attachment_moment_n_m_s",
    }
    assert fields <= set(pd.CollisionReport.__dataclass_fields__)


def test_ground_restitution_matches_contact_rs_default() -> None:
    # rust_core/upstream-physics/src/contact.rs: cor = 0.78.
    assert pd.GROUND_RESTITUTION == 0.78
