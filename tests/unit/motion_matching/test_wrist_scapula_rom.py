"""SHOW-A19 (#14795): evidence-based wrist and scapula range of motion characterisation."""

from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"

WRIST_SCAPULA_COORDINATES = {
    "LWInputX", "RWInputX", "LWInputY", "RWInputY",
    "LScapInputX", "RScapInputX", "LScapInputY", "RScapInputY",
}  # fmt: skip


def test_every_wrist_and_scapula_axis_has_a_cited_range() -> None:
    assert set(WRIST_SCAPULA_ROM) == WRIST_SCAPULA_COORDINATES
    for rom in WRIST_SCAPULA_ROM.values():
        assert rom.lower_deg < rom.upper_deg and rom.sources and rom.motion
        assert all(len(s) > 20 for s in rom.sources)
        if rom.capture_observed is not None:
            c_lo, c_hi = rom.capture_observed
            assert rom.lower_deg <= c_lo and rom.upper_deg >= c_hi
        if rom.margin_deg is not None:
            assert rom.margin_deg > 0.0


def test_range_of_motion_table_reads_the_cited_wrist_scapula_ranges() -> None:
    for name, rom in WRIST_SCAPULA_ROM.items():
        assert range_of_motion.HUMAN_RANGES_DEG[name] == (rom.lower_deg, rom.upper_deg)
        assert range_of_motion.UPPER_RANGES_DEG[name] == (rom.lower_deg, rom.upper_deg)


def test_spec_coordinate_ranges_match_cited_wrist_scapula_table() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name, rom in WRIST_SCAPULA_ROM.items():
        assert tuple(spec["coordinate_ranges_deg"][name]) == (
            rom.lower_deg,
            rom.upper_deg,
        )


def test_wrist_scapula_range_rad_is_precondition_checked() -> None:
    rad = wrist_scapula_range_rad
    assert rad("RWInputX") == (pytest.approx(-1.04719755), pytest.approx(0.78539816))
    with pytest.raises(PreconditionError):
        rad("NotACoordinate")


def test_option_c_address_reachability_improvement() -> None:
    """Option (c) matched-V reachability under baseline vs evidence ranges (#14795)."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.grip_rotation import (
        apply_grip_rotation,
    )

    spec = json.loads(SPEC.read_text("utf-8"))
    poses_raw = (ROOT / "tests/fixtures/club_face/address_poses.json").read_text(
        "utf-8"
    )
    q0 = np.array(json.loads(poses_raw)["poses"]["driver"]["q_rad"], dtype=float)
    totals_c = {"L": [-85.5491, 43.3732, -26.1391], "R": [85.5491, -43.3732, 153.8609]}

    def make_rig(s: dict):
        xml, meta = export_full_body_mjcf(json.dumps(s).encode(), visual=True)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        j = [
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in meta["coordinate_order"]
        ]
        lw = meta["frame_sites"]["LW"]
        s_idx = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in (lw, "native_closure_a", "native_closure_b")
        ]

        def sites(q: np.ndarray):
            d.qpos[j] = q
            mujoco.mj_forward(m, d)
            return [
                (d.site_xpos[i].copy(), d.site_xmat[i].reshape(3, 3).copy())
                for i in s_idx
            ]

        return meta["coordinate_order"], sites

    order, old_sites = make_rig(spec)
    new_spec, _ = apply_grip_rotation(spec, totals_c)
    _, new_sites = make_rig(new_spec)
    (rp, rR), _, _ = old_sites(q0)

    arm_k = ("SInput", "EInput", "FInput", "WInput", "Scap")
    arm_names = [n for n in order if n[0] in "LR" and any(k in n for k in arm_k)]
    arm_idx = [order.index(n) for n in arm_names]

    def solve(bounds_dict: dict[str, tuple[float, float]]):
        deg = [bounds_dict.get(n, (-180, 180)) for n in arm_names]
        lb, ub = np.radians([b[0] for b in deg]), np.radians([b[1] for b in deg])

        def resid(x: np.ndarray) -> np.ndarray:
            row = q0.copy()
            row[arm_idx] = x
            (p, R), a, b = new_sites(row)
            d_p, d_R = (p - rp) * 100, Rot.from_matrix(rR.T @ R).as_rotvec() * 17.2
            d_a = (a[0] - b[0]) * 100
            d_b = Rot.from_matrix(a[1].T @ b[1]).as_rotvec() * 17.2
            return np.concatenate([d_p, d_R, d_a, d_b])

        sol = least_squares(
            resid, np.clip(q0[arm_idx], lb, ub), bounds=(lb, ub), xtol=1e-12, ftol=1e-12
        )
        row_opt = q0.copy()
        row_opt[arm_idx] = sol.x
        (p, _), _, _ = new_sites(row_opt)
        return float(np.linalg.norm(p - rp) * 1000), dict(
            zip(arm_names, np.degrees(sol.x), strict=True)
        )

    legacy = dict(range_of_motion.HUMAN_RANGES_DEG)
    for k, v in [
        ("WInputX", (-40.0, 25.0)), ("WInputY", (-70.0, 70.0)),
        ("ScapInputX", (-10.0, 30.0)), ("ScapInputY", (-40.0, 40.0)),
    ]:  # fmt: skip
        legacy[f"L{k}"] = legacy[f"R{k}"] = v

    err_legacy, vals_legacy = solve(legacy)
    assert err_legacy == pytest.approx(26.74, abs=0.1)
    assert vals_legacy["RWInputX"] == pytest.approx(25.0, abs=0.1)
    assert vals_legacy["RWInputY"] == pytest.approx(-70.0, abs=0.1)

    err_evidence, vals_evidence = solve(range_of_motion.HUMAN_RANGES_DEG)
    assert err_evidence < err_legacy and err_evidence == pytest.approx(18.76, abs=0.1)
    assert -60.0 < vals_evidence["RWInputX"] < 45.0
