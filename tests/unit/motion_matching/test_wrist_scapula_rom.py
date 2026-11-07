"""SHOW-A19 (#14795): evidence-based wrist and scapula range of motion characterisation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.motion_matching import range_of_motion
from src.shared.python.motion_matching.wrist_scapula_rom import (
    WRIST_SCAPULA_ROM,
    wrist_scapula_range_rad,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)

WRIST_SCAPULA_COORDINATES = {
    "LWInputX",
    "RWInputX",
    "LWInputY",
    "RWInputY",
    "LScapInputX",
    "RScapInputX",
    "LScapInputY",
    "RScapInputY",
}


def test_every_wrist_and_scapula_axis_has_a_cited_range() -> None:
    assert set(WRIST_SCAPULA_ROM) == WRIST_SCAPULA_COORDINATES
    for rom in WRIST_SCAPULA_ROM.values():
        assert rom.lower_deg < rom.upper_deg
        assert rom.sources and all(len(s) > 20 for s in rom.sources)
        assert rom.motion
        if rom.capture_observed is not None:
            c_lo, c_hi = rom.capture_observed
            assert rom.lower_deg <= c_lo, f"{rom.coordinate}: {rom.lower_deg} > {c_lo}"
            assert rom.upper_deg >= c_hi, f"{rom.coordinate}: {rom.upper_deg} < {c_hi}"
        if rom.margin_deg is not None:
            assert rom.margin_deg > 0.0


def test_range_of_motion_table_reads_the_cited_wrist_scapula_ranges() -> None:
    for name, rom in WRIST_SCAPULA_ROM.items():
        assert range_of_motion.HUMAN_RANGES_DEG[name] == (
            rom.lower_deg,
            rom.upper_deg,
        )
        assert range_of_motion.UPPER_RANGES_DEG[name] == (
            rom.lower_deg,
            rom.upper_deg,
        )


def test_spec_coordinate_ranges_match_cited_wrist_scapula_table() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name, rom in WRIST_SCAPULA_ROM.items():
        assert tuple(spec["coordinate_ranges_deg"][name]) == (
            rom.lower_deg,
            rom.upper_deg,
        )


def test_wrist_scapula_range_rad_is_precondition_checked() -> None:
    lo, hi = wrist_scapula_range_rad("RWInputX")
    assert lo == pytest.approx(math.radians(-60.0))
    assert hi == pytest.approx(math.radians(45.0))
    with pytest.raises(PreconditionError):
        wrist_scapula_range_rad("NotACoordinate")


def test_option_c_address_reachability_improvement() -> None:
    """Option (c) matched-V reachability under baseline vs evidence ranges (#14795)."""
    import importlib.util
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    grip_rot_path = (
        Path(__file__).resolve().parents[3]
        / "src/shared/python/motion_matching/execution/grip_rotation.py"
    )
    spec_mod = importlib.util.spec_from_file_location("grip_rot_mod", grip_rot_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    apply_grip_rotation = mod.apply_grip_rotation

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    poses_path = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/club_face/address_poses.json"
    )
    poses = json.loads(poses_path.read_text(encoding="utf-8"))["poses"]
    q0 = np.array(poses["driver"]["q_rad"], dtype=float)

    totals_c = {
        "L": [-85.5491, 43.3732, -26.1391],
        "R": [85.5491, -43.3732, 153.8609],
    }

    class TestRig:
        def __init__(self, spec_dict: dict) -> None:
            xml, self.meta = export_full_body_mjcf(
                json.dumps(spec_dict).encode(), visual=True
            )
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            self.order = self.meta["coordinate_order"]
            self.joint = [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
                for n in self.order
            ]

        def sites(self, q: np.ndarray):
            self.data.qpos[self.joint] = q
            mujoco.mj_forward(self.model, self.data)

            def get(name):
                i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                return (
                    self.data.site_xpos[i].copy(),
                    self.data.site_xmat[i].reshape(3, 3).copy(),
                )

            return (
                get(self.meta["frame_sites"]["LW"]),
                get("native_closure_a"),
                get("native_closure_b"),
            )

    old = TestRig(spec)
    new_spec, _ = apply_grip_rotation(spec, totals_c)
    new = TestRig(new_spec)

    (rp, rR), _, _ = old.sites(q0)

    arm_idx = [
        i
        for i, n in enumerate(new.order)
        if n[0] in "LR"
        and any(k in n for k in ("SInput", "EInput", "FInput", "WInput", "Scap"))
    ]
    arm_names = [new.order[i] for i in arm_idx]

    def solve(bounds_dict: dict[str, tuple[float, float]]):
        lb = np.array(
            [math.radians(bounds_dict.get(n, (-180, 180))[0]) for n in arm_names]
        )
        ub = np.array(
            [math.radians(bounds_dict.get(n, (-180, 180))[1]) for n in arm_names]
        )

        def resid(x):
            row = q0.copy()
            row[arm_idx] = x
            (p, R), a, b = new.sites(row)
            return np.concatenate(
                [
                    (p - rp) * 100,
                    Rot.from_matrix(rR.T @ R).as_rotvec() * 17.2,
                    (a[0] - b[0]) * 100,
                    Rot.from_matrix(a[1].T @ b[1]).as_rotvec() * 17.2,
                ]
            )

        sol = least_squares(
            resid, np.clip(q0[arm_idx], lb, ub), bounds=(lb, ub), xtol=1e-12, ftol=1e-12
        )
        row_opt = q0.copy()
        row_opt[arm_idx] = sol.x
        (p, _), _, _ = new.sites(row_opt)
        err_mm = np.linalg.norm(p - rp) * 1000
        deg_vals = dict(zip(arm_names, np.degrees(sol.x), strict=True))
        return err_mm, deg_vals

    legacy_bounds = {
        **range_of_motion.HUMAN_RANGES_DEG,
        "LWInputX": (-40.0, 25.0),
        "RWInputX": (-40.0, 25.0),
        "LWInputY": (-70.0, 70.0),
        "RWInputY": (-70.0, 70.0),
        "LScapInputX": (-10.0, 30.0),
        "RScapInputX": (-10.0, 30.0),
        "LScapInputY": (-40.0, 40.0),
        "RScapInputY": (-40.0, 40.0),
    }

    err_legacy, vals_legacy = solve(legacy_bounds)
    assert err_legacy == pytest.approx(26.74, abs=0.1)
    assert vals_legacy["RWInputX"] == pytest.approx(25.0, abs=0.1)  # bound hi
    assert vals_legacy["RWInputY"] == pytest.approx(-70.0, abs=0.1)  # bound lo

    err_evidence, vals_evidence = solve(range_of_motion.HUMAN_RANGES_DEG)
    assert err_evidence < err_legacy  # Error strictly reduced
    assert err_evidence == pytest.approx(18.76, abs=0.1)
    # RWInputX is now free (inside [-60, 45]), sitting at ~43.9 deg
    assert -60.0 < vals_evidence["RWInputX"] < 45.0
