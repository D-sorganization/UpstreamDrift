"""Per-hand ``grip_force_limit_n`` field, validation and fail-closed flag (HRG-HX4, #13761).

``grip_force_limit_n`` is a force in newtons. It is not
``GripPose.grip_strength_deg``, which is a visual roll. Per the option (c)
ruling the cap is only ever enforced in the Python bushing law; MuJoCo pad
contact (and every engine without enforcement) must raise on a finite limit.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.grip_contact.capability_inventory import (
    FORCE_LIMIT_MECHANISMS,
    load_grip_capability_inventory,
)
from src.shared.python.grip_contact.grip_force_limit import UNLIMITED
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

pytestmark = pytest.mark.unit

ENGINES = (
    "mujoco",
    "drake",
    "opensim",
    "pinocchio",
    "simscape",
    "jaxsim",
    "myosuite",
)


def test_grip_pose_exposes_per_hand_grip_force_limit_default_unlimited() -> None:
    for hand in Hand:
        assert DEFAULT_GRIP_POSE.grip_force_limit_n(hand) == UNLIMITED
    assert UNLIMITED == "unlimited"
    pose = GripPose(lead_grip_force_limit_n=300.0, trail_grip_force_limit_n=250.0)
    assert pose.grip_force_limit_n(Hand.LEAD) == 300.0
    assert pose.grip_force_limit_n(Hand.TRAIL) == 250.0


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, math.nan, math.inf, "fast", "", True, None, [5.0]]
)
@pytest.mark.parametrize(
    "field", ["lead_grip_force_limit_n", "trail_grip_force_limit_n"]
)
def test_grip_force_limit_rejects_nonpositive_values(field: str, bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        GripPose(**{field: bad})  # type: ignore[arg-type]


def test_finite_grip_limit_raises_on_engine_without_enforcement() -> None:
    inventory = load_grip_capability_inventory()
    report = inventory.force_limit_enforcement_report()
    assert report["default"] == UNLIMITED
    engines = report["engines"]
    assert set(ENGINES) <= set(engines)
    bushing_enforcing_engines = {"mujoco", "pinocchio"}
    for engine in ENGINES:
        if engine in bushing_enforcing_engines:
            assert engines[engine] == {"pad_contact": False, "weld_bushing": True}
            # weld_bushing is enforceable, does not raise
            inventory.require_grip_force_limit_enforceable(
                engine, "weld_bushing", 100.0
            )
            # pad_contact is not enforceable, raises
            with pytest.raises(ValueError, match=engine):
                inventory.require_grip_force_limit_enforceable(
                    engine, "pad_contact", 100.0
                )
        else:
            assert engines[engine] == dict.fromkeys(FORCE_LIMIT_MECHANISMS, False)
            for mechanism in FORCE_LIMIT_MECHANISMS:
                with pytest.raises(ValueError, match=engine):
                    inventory.require_grip_force_limit_enforceable(
                        engine, mechanism, 100.0
                    )
    with pytest.raises(ValueError, match="unknown_engine"):
        inventory.require_grip_force_limit_enforceable(
            "unknown_engine", "pad_contact", 1.0
        )


def test_mujoco_club_retained_below_limit_slips_above_limit() -> None:
    """HRG-HX4 (#13761): below limit club is retained, above limit force is capped (slip)."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed")

    from pathlib import Path
    import json
    import numpy as np
    from src.engines.physics_engines.mujoco.python.grip_bushing import (
        probe_bushing_forces,
    )
    from src.shared.python.grip_contact import load_coordinate_swing

    root = Path(__file__).resolve().parents[3]
    models = root / "docs/development/full_body_models"
    fixtures = root / "tests/fixtures/club_face"
    spec_bytes = (models / "full_body_spec_anthro_driver.json").read_bytes()
    names = json.loads(spec_bytes)["coordinate_order"]
    swing = load_coordinate_swing(
        fixtures / "swing_q_driver.npz",
        fixtures / "address_poses.json",
        "driver",
        names,
    )

    cap_n = 150.0
    capped_pose = GripPose(
        lead_grip_force_limit_n=cap_n,
        trail_grip_force_limit_n=cap_n,
    )

    # 1. Deflection below limit: 0.1 mm along x produces ~100 N (< 150 N)
    delta_below = np.array([1e-4, 0.0, 0.0])
    probe_free_below = probe_bushing_forces(spec_bytes, swing, delta_below)
    probe_capped_below = probe_bushing_forces(
        spec_bytes, swing, delta_below, grip_pose=capped_pose
    )
    norm_free_below = float(np.linalg.norm(probe_free_below.force_n["L"]))
    norm_capped_below = float(np.linalg.norm(probe_capped_below.force_n["L"]))
    assert norm_free_below < cap_n
    # Retained: bit-identical or approx to uncapped force
    np.testing.assert_allclose(
        probe_capped_below.force_n["L"], probe_free_below.force_n["L"], rtol=1e-12
    )
    assert norm_capped_below == pytest.approx(norm_free_below, rel=1e-12)

    # 2. Deflection above limit: 0.25 mm along x produces ~250 N (> 150 N)
    delta_above = np.array([2.5e-4, 0.0, 0.0])
    probe_free_above = probe_bushing_forces(spec_bytes, swing, delta_above)
    probe_capped_above = probe_bushing_forces(
        spec_bytes, swing, delta_above, grip_pose=capped_pose
    )
    norm_free_above = float(np.linalg.norm(probe_free_above.force_n["L"]))
    norm_capped_above = float(np.linalg.norm(probe_capped_above.force_n["L"]))
    assert norm_free_above > cap_n
    # Slip: capped force magnitude is exactly cap_n, direction preserved
    assert norm_capped_above == pytest.approx(cap_n, rel=1e-9)
    np.testing.assert_allclose(
        probe_capped_above.force_n["L"] / cap_n,
        probe_free_above.force_n["L"] / norm_free_above,
        rtol=1e-9,
    )
    # Total engine force on club in qfrc_passive reflects the cap
    assert np.linalg.norm(probe_capped_above.engine_total_force_n) < np.linalg.norm(
        probe_free_above.engine_total_force_n
    )


def test_default_unlimited_leaves_grip_pose_and_engines_unchanged() -> None:
    """Regression: the explicit ``unlimited`` default changes nothing."""
    explicit = GripPose(
        lead_grip_force_limit_n=UNLIMITED, trail_grip_force_limit_n=UNLIMITED
    )
    assert explicit == DEFAULT_GRIP_POSE == GripPose()
    for hand in Hand:
        assert explicit.distance_below_butt_m(hand) == (
            DEFAULT_GRIP_POSE.distance_below_butt_m(hand)
        )
    inventory = load_grip_capability_inventory()
    for engine in (*ENGINES, "unknown_engine"):
        for mechanism in FORCE_LIMIT_MECHANISMS:
            inventory.require_grip_force_limit_enforceable(engine, mechanism, UNLIMITED)
