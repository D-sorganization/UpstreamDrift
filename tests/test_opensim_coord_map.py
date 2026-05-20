"""Tests for the OpenSim ↔ Simscape coordinate-mapping helper.

These tests are deliberately pure-Python and do not import the OpenSim
SWIG wrapper. The single test that exercises a live OpenSim model is
gated behind ``pytest.mark.requires_opensim``.

Acceptance criteria from issue #4114:

* ``to_simscape ∘ from_simscape == identity`` within 1e-12 absolute
  tolerance for 100 random poses.
* Hand-verified neutral-pose mapping is exact.
* Tests run on every CI lane (no OpenSim dependency for the core helper).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching.coord_map import (
    OPENSIM_COORD_ORDER,
    OPENSIM_NEUTRAL_POSE,
    OPENSIM_SIGN_CONVENTION,
    OPENSIM_TO_SIMSCAPE,
    SIMSCAPE_COORD_ORDER,
    frame_y_up_to_z_up,
    frame_z_up_to_y_up,
    from_simscape,
    quat_canonical_to_eigen,
    quat_eigen_to_canonical,
    to_simscape,
)

# Project the requires_opensim marker — registered locally below so the
# test still loads on bare CI lanes that don't have ``opensim`` installed.
requires_opensim = pytest.mark.requires_opensim


# ---------------------------------------------------------------------------
# Sanity tests — table integrity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_opensim_order_has_39_unique_entries():
    assert len(OPENSIM_COORD_ORDER) == 39
    assert len(set(OPENSIM_COORD_ORDER)) == 39, "no duplicate OpenSim coords"


@pytest.mark.unit
def test_simscape_order_has_25_unique_entries():
    assert len(SIMSCAPE_COORD_ORDER) == 25
    assert len(set(SIMSCAPE_COORD_ORDER)) == 25, "no duplicate Simscape coords"


@pytest.mark.unit
def test_mapping_table_uses_only_known_names():
    for os_name, sim_name in OPENSIM_TO_SIMSCAPE.items():
        assert (
            os_name in OPENSIM_COORD_ORDER
        ), f"{os_name!r} not a known OpenSim coordinate"
        assert (
            sim_name in SIMSCAPE_COORD_ORDER
        ), f"{sim_name!r} not a known Simscape coordinate"


@pytest.mark.unit
def test_sign_convention_only_uses_unit_signs():
    for k, v in OPENSIM_SIGN_CONVENTION.items():
        assert v in (-1.0, +1.0), f"{k}: sign {v} is not ±1.0"


@pytest.mark.unit
def test_simscape_targets_are_unique():
    """Each Simscape coordinate must be mapped from at most one OpenSim coord."""
    targets = list(OPENSIM_TO_SIMSCAPE.values())
    assert len(targets) == len(set(targets)), "duplicate Simscape targets in mapping"


@pytest.mark.unit
def test_neutral_pose_is_length_39_and_finite():
    assert OPENSIM_NEUTRAL_POSE.shape == (39,)
    assert np.all(np.isfinite(OPENSIM_NEUTRAL_POSE))


# ---------------------------------------------------------------------------
# Round-trip tests — to_simscape ∘ from_simscape == identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_round_trip_identity_zero_pose():
    q_sim = np.zeros(25, dtype=np.float64)
    q_round = to_simscape(from_simscape(q_sim))
    np.testing.assert_allclose(q_round, q_sim, atol=1e-12)


@pytest.mark.unit
def test_round_trip_identity_random_poses():
    """Random Simscape poses round-trip exactly through OpenSim and back.

    Per issue #4114, this is the headline acceptance criterion:
    ``to_simscape ∘ from_simscape == identity`` within 1e-12 absolute
    tolerance for 100 random poses.

    The four scapulothoracic Simscape coordinates have no OpenSim
    counterpart in Rajagopal2015 — they are zeroed before the round trip
    so the identity check is meaningful (they cannot be preserved by any
    mapping that goes through a 39-D OpenSim coordinate vector).
    """
    rng = np.random.default_rng(seed=42)
    scap_idx = [
        SIMSCAPE_COORD_ORDER.index(c)
        for c in ("l_scap.rx", "l_scap.ry", "r_scap.rx", "r_scap.ry")
    ]
    for trial in range(100):
        q_sim = rng.standard_normal(25)
        q_sim[scap_idx] = 0.0  # scapula DOFs are unmapped by construction
        q_round = to_simscape(from_simscape(q_sim))
        np.testing.assert_allclose(q_round, q_sim, atol=1e-12, err_msg=f"trial {trial}")


@pytest.mark.unit
def test_round_trip_identity_extreme_values():
    """Round trip survives values far outside any plausible physiological range.

    Only mapped DOFs are exercised; the scapulothoracic DOFs (positions
    9, 10, 17, 18) stay zero because they have no OpenSim counterpart.
    """
    q_sim = np.array(
        [1e6, -1e6, 1e3, np.pi, -np.pi, 100.0] + [0.0] * 19,
        dtype=np.float64,
    )
    q_round = to_simscape(from_simscape(q_sim))
    np.testing.assert_allclose(q_round, q_sim, atol=1e-9)


# ---------------------------------------------------------------------------
# Hand-verified neutral-pose mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_neutral_opensim_pose_maps_to_zero_simscape():
    """Zero OpenSim pose → zero Simscape pose (modulo unmapped DOFs).

    Hand-verification: the Rajagopal2015 default state is the all-zeros
    coordinate vector; every Simscape coordinate that has an OpenSim
    counterpart must therefore also be zero. Coordinates without an
    OpenSim counterpart (the four scapulothoracic DOFs) are filled with
    zero by the projection.
    """
    q_os = np.zeros(39, dtype=np.float64)
    q_sim = to_simscape(q_os)
    np.testing.assert_array_equal(q_sim, np.zeros(25))


@pytest.mark.unit
def test_neutral_simscape_pose_maps_to_neutral_opensim():
    """Zero Simscape pose → OpenSim neutral pose."""
    q_sim = np.zeros(25, dtype=np.float64)
    q_os = from_simscape(q_sim)
    np.testing.assert_array_equal(q_os, OPENSIM_NEUTRAL_POSE)


@pytest.mark.unit
def test_pelvis_translation_y_up_to_z_up_swap():
    """OpenSim pelvis_ty (Y-up vertical) maps onto Simscape hip.tz (Z-up vertical)."""
    q_os = np.zeros(39, dtype=np.float64)
    pelvis_ty_idx = OPENSIM_COORD_ORDER.index("pelvis_ty")
    q_os[pelvis_ty_idx] = 0.95  # 0.95 m vertical — typical pelvis height
    q_sim = to_simscape(q_os)
    hip_tz_idx = SIMSCAPE_COORD_ORDER.index("hip.tz")
    assert q_sim[hip_tz_idx] == pytest.approx(0.95)
    # And no other Simscape DOF should be touched
    expected = np.zeros(25)
    expected[hip_tz_idx] = 0.95
    np.testing.assert_array_equal(q_sim, expected)


@pytest.mark.unit
def test_lumbar_block_maps_to_spine_torso():
    """Lumbar 3-DOF block → spine.rx + spine.ry + torso.rz."""
    q_os = np.zeros(39, dtype=np.float64)
    q_os[OPENSIM_COORD_ORDER.index("lumbar_extension")] = 0.10
    q_os[OPENSIM_COORD_ORDER.index("lumbar_bending")] = 0.05
    q_os[OPENSIM_COORD_ORDER.index("lumbar_rotation")] = 0.20
    q_sim = to_simscape(q_os)
    assert q_sim[SIMSCAPE_COORD_ORDER.index("spine.rx")] == pytest.approx(0.10)
    assert q_sim[SIMSCAPE_COORD_ORDER.index("spine.ry")] == pytest.approx(0.05)
    assert q_sim[SIMSCAPE_COORD_ORDER.index("torso.rz")] == pytest.approx(0.20)


@pytest.mark.unit
def test_left_wrist_dev_sign_flips():
    """``wrist_dev_l`` is documented as +1 vs Simscape (right side flips, left agrees)."""
    q_os = np.zeros(39, dtype=np.float64)
    q_os[OPENSIM_COORD_ORDER.index("wrist_dev_l")] = 0.30
    q_sim = to_simscape(q_os)
    expected_sign = OPENSIM_SIGN_CONVENTION["wrist_dev_l"]
    assert q_sim[SIMSCAPE_COORD_ORDER.index("l_wrist.ry")] == pytest.approx(
        expected_sign * 0.30
    )


@pytest.mark.unit
def test_right_wrist_dev_sign_flips():
    """``wrist_dev_r`` is documented as -1 vs Simscape."""
    q_os = np.zeros(39, dtype=np.float64)
    q_os[OPENSIM_COORD_ORDER.index("wrist_dev_r")] = 0.30
    q_sim = to_simscape(q_os)
    assert OPENSIM_SIGN_CONVENTION["wrist_dev_r"] == -1.0
    assert q_sim[SIMSCAPE_COORD_ORDER.index("r_wrist.ry")] == pytest.approx(-0.30)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_simscape_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expects a"):
        to_simscape(np.zeros(38))


@pytest.mark.unit
def test_to_simscape_rejects_nan():
    bad = np.zeros(39)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        to_simscape(bad)


@pytest.mark.unit
def test_to_simscape_rejects_inf():
    bad = np.zeros(39)
    bad[5] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        to_simscape(bad)


@pytest.mark.unit
def test_from_simscape_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expects a"):
        from_simscape(np.zeros(24))


@pytest.mark.unit
def test_from_simscape_rejects_nan():
    bad = np.zeros(25)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        from_simscape(bad)


@pytest.mark.unit
def test_to_simscape_accepts_list_input():
    q_sim = to_simscape([0.0] * 39)
    assert q_sim.shape == (25,)
    assert q_sim.dtype == np.float64


@pytest.mark.unit
def test_unmapped_opensim_coords_are_dropped():
    """Setting only leg DOFs in OpenSim must produce zero Simscape pose."""
    q_os = np.zeros(39, dtype=np.float64)
    leg_coords = [
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "knee_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
        "mtp_angle_l",
        "pro_sup_r",
        "pro_sup_l",
    ]
    for c in leg_coords:
        q_os[OPENSIM_COORD_ORDER.index(c)] = 0.5
    q_sim = to_simscape(q_os)
    np.testing.assert_array_equal(q_sim, np.zeros(25))


@pytest.mark.unit
def test_unmapped_simscape_coords_get_dropped_from_round_trip():
    """The four scapulothoracic DOFs should NOT survive a Simscape→OpenSim→Simscape trip."""
    q_sim = np.zeros(25, dtype=np.float64)
    for c in ("l_scap.rx", "l_scap.ry", "r_scap.rx", "r_scap.ry"):
        q_sim[SIMSCAPE_COORD_ORDER.index(c)] = 0.42
    q_round = to_simscape(from_simscape(q_sim))
    # Scapula DOFs are dropped on the way to OpenSim, so come back as 0.
    for c in ("l_scap.rx", "l_scap.ry", "r_scap.rx", "r_scap.ry"):
        assert q_round[SIMSCAPE_COORD_ORDER.index(c)] == 0.0


# ---------------------------------------------------------------------------
# Quaternion ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quat_eigen_to_canonical_basic():
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])
    q_wxyz = quat_eigen_to_canonical(q_xyzw)
    np.testing.assert_array_equal(q_wxyz, np.array([0.9, 0.1, 0.2, 0.3]))


@pytest.mark.unit
def test_quat_round_trip():
    rng = np.random.default_rng(0)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    np.testing.assert_allclose(
        quat_canonical_to_eigen(quat_eigen_to_canonical(q)), q, atol=1e-15
    )


@pytest.mark.unit
def test_quat_batch_shape_preserved():
    q_xyzw = np.zeros((5, 4))
    q_xyzw[:, 3] = 1.0  # identity quaternions
    q_wxyz = quat_eigen_to_canonical(q_xyzw)
    assert q_wxyz.shape == (5, 4)
    np.testing.assert_array_equal(q_wxyz[:, 0], 1.0)
    np.testing.assert_array_equal(q_wxyz[:, 1:], 0.0)


@pytest.mark.unit
def test_quat_eigen_rejects_wrong_last_axis():
    with pytest.raises(ValueError, match="last axis"):
        quat_eigen_to_canonical(np.zeros(3))


# ---------------------------------------------------------------------------
# Frame Y-up ↔ Z-up
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_frame_y_up_to_z_up_basic():
    # OpenSim +Y (up) should land on Simscape +Z (up).
    v_yup = np.array([0.0, 1.0, 0.0])
    v_zup = frame_y_up_to_z_up(v_yup)
    np.testing.assert_allclose(v_zup, np.array([0.0, 0.0, 1.0]))
    # OpenSim +Z (right) → Simscape +Y (right).
    v_yup = np.array([0.0, 0.0, 1.0])
    v_zup = frame_y_up_to_z_up(v_yup)
    np.testing.assert_allclose(v_zup, np.array([0.0, 1.0, 0.0]))
    # OpenSim +X (anterior) stays as Simscape +X (target line).
    v_yup = np.array([1.0, 0.0, 0.0])
    v_zup = frame_y_up_to_z_up(v_yup)
    np.testing.assert_allclose(v_zup, np.array([1.0, 0.0, 0.0]))


@pytest.mark.unit
def test_frame_round_trip():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((10, 3))
    v_round = frame_z_up_to_y_up(frame_y_up_to_z_up(v))
    np.testing.assert_allclose(v_round, v, atol=1e-15)


@pytest.mark.unit
def test_frame_rejects_wrong_last_axis():
    with pytest.raises(ValueError, match="last axis"):
        frame_y_up_to_z_up(np.zeros(2))


# ---------------------------------------------------------------------------
# Live-OpenSim validation (gated behind requires_opensim)
# ---------------------------------------------------------------------------
