"""Independent array-level recomputation of representative headline claims.

These tests intentionally do not read a summary JSON's ``passed`` booleans or
headline scalars.  They derive the registered quantities from the committed
CSV/NPZ arrays so a false self-report cannot satisfy the release gate (#8918).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


@pytest.mark.scientific
def test_planar_wscg_negative_couple_is_recomputed_from_raw_table() -> None:
    rows = np.genfromtxt(DATA / "wscg_two_hand_raw/ztcf.csv", delimiter=",", names=True)

    couple = rows["equivalent_midpoint_couple_local_source_z_nm"]
    lead_command = rows["lead_command_torque_nm"]
    trail_command = rows["trail_command_torque_nm"]

    assert float(np.min(couple)) == pytest.approx(-19.6304815091102)
    assert np.all(lead_command == 0.0)
    assert np.all(trail_command == 0.0)


@pytest.mark.scientific
def test_spatial_killswitch_duration_and_nondegenerate_parity_are_recomputed() -> None:
    with np.load(DATA / "spatial_forward_contact_study.npz") as arrays:
        durations = []
        couples = []
        for engine in ("mujoco", "pinocchio"):
            time = arrays[f"{engine}_killswitch_time"]
            couple = arrays[f"{engine}_killswitch_swing_normal_couple"]
            post_cut_negative = (time >= 0.18) & (couple < 0.0)
            step = float(np.median(np.diff(time)))
            durations.append(float(np.count_nonzero(post_cut_negative)) * step)
            couples.append(couple)

    assert durations == pytest.approx([0.0375, 0.0375])
    relative_rms = np.sqrt(np.mean(np.square(couples[0] - couples[1]))) / max(
        np.sqrt(np.mean(np.square(couples[0]))), 1.0e-12
    )
    assert relative_rms < 0.002
    assert not np.array_equal(couples[0], couples[1])


@pytest.mark.scientific
def test_articulated_shaft_match_counts_and_mixed_signs_are_recomputed() -> None:
    with np.load(DATA / "articulated_shaft_atlas.npz") as arrays:
        matched = np.asarray(arrays["matched_load_work"], dtype=bool)
        speed_difference = np.asarray(
            arrays["matched_final_speed_difference_m_s"], dtype=float
        )

    retained = speed_difference[matched]
    assert matched.size == 384
    assert int(np.count_nonzero(matched)) == 126
    assert int(np.count_nonzero(retained < 0.0)) == 82
    assert int(np.count_nonzero(retained > 0.0)) == 44
    assert int(np.count_nonzero(retained == 0.0)) == 0


@pytest.mark.scientific
def test_articulated_ground_primary_and_posthoc_counts_are_recomputed() -> None:
    with np.load(DATA / "articulated_ground_atlas.npz") as arrays:
        matched = np.asarray(arrays["matched"], dtype=bool)
        load_error = np.asarray(arrays["load_match_relative_error"], dtype=float)
        total_work = np.asarray(arrays["primary_terminal_total_work"], dtype=float)
        ground_work = np.asarray(arrays["primary_terminal_ground_work"], dtype=float)
        speed = np.asarray(arrays["primary_final_speed"], dtype=float)

    fixed, coupled = 0, 3
    nonground_work = total_work - ground_work
    left = nonground_work[:, coupled]
    right = nonground_work[:, fixed]
    scale = np.maximum(1.0e-6, 0.5 * (np.abs(left) + np.abs(right)))
    nonground_error = np.abs(left - right) / scale
    alternative = (load_error <= 0.05) & (nonground_error <= 0.05)
    speed_delta = speed[:, coupled] - speed[:, fixed]
    retained = speed_delta[alternative]

    assert matched.size == 384
    assert int(np.count_nonzero(matched)) == 0
    assert int(np.count_nonzero(alternative)) == 60
    assert int(np.count_nonzero(retained < 0.0)) == 40
    assert int(np.count_nonzero(retained > 0.0)) == 20
