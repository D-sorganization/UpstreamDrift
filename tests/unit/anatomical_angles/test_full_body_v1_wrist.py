"""HRG-E2 (#13169): wrist and forearm angles of the full-body-v1 model.

The adapter turns engine body rotations (any engine) into Wu wrist and ISB
elbow frames through a neutral-pose calibration declared in the HRG-00 model
basis. The synthetic tests build the arm chain from the pinned sign-map axes;
the MuJoCo test checks the same signs on the compiled fixture spec.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.anatomical_angles import OBSERVABILITY
from src.shared.python.anatomical_angles import full_body_v1 as fb
from src.shared.python.motion_matching import coordinate_sign_map as sign_map

pytestmark = pytest.mark.unit

RECEIPT = "receipts/fit/receipt.json"
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
NATIVE_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence"
    / "native_geometry_spec_9967.json"
)
THETA = np.radians(20.0)


def _axis_rot(coordinate: str, angle: float) -> np.ndarray:
    axis = np.asarray(sign_map.sign_for(coordinate).axis_model, dtype=float)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _synthetic(
    side: str,
    q: dict[str, float],
    body_offsets: dict[str, np.ndarray] | None = None,
    pelvis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """World rotations of the arm bodies built from the sign-map axes.

    At ``q = 0`` every body is aligned with the pelvis; ``body_offsets`` adds
    an arbitrary constant body-local frame, ``pelvis`` a world placement.
    """
    chain = fb.side_chain(side)
    p = np.eye(3) if pelvis is None else pelvis
    humerus = p
    forearm = humerus @ _axis_rot(chain.elbow, q.get(chain.elbow, 0.0))
    forearm = forearm @ _axis_rot(chain.forearm_axial, q.get(chain.forearm_axial, 0.0))
    hand = forearm @ _axis_rot(chain.wrist_x, q.get(chain.wrist_x, 0.0))
    hand = hand @ _axis_rot(chain.wrist_y, q.get(chain.wrist_y, 0.0))
    out = {chain.humerus: humerus, chain.forearm_body: forearm, chain.hand: hand}
    offsets = body_offsets or {}
    return {k: v @ offsets.get(k, np.eye(3)) for k, v in out.items()}


def _angles(side: str, q: dict[str, float], **kw: Any) -> fb.WristForearm:
    neutral = _synthetic(side, {}, **kw)
    posed = _synthetic(side, q, **kw)
    cal = fb.calibrate_neutral(side, neutral, pelvis_rotation=kw.get("pelvis"))
    series = {k: v[None] for k, v in posed.items()}
    return fb.full_body_v1_wrist_forearm(cal, series, receipt_path=RECEIPT)


def test_chain_bodies_come_from_the_sign_map() -> None:
    left, right = fb.side_chain("L"), fb.side_chain("R")
    assert left.hand == sign_map.sign_for("LWInputX").body == "Clubface Vector"
    assert left.forearm_body == sign_map.sign_for("LWInputX").parent_body
    assert left.humerus == sign_map.sign_for("LEInput").parent_body == "LUpperArm"
    assert right.hand == "RHandStandoff"
    assert right.forearm_body == "RLowerForearm"
    with pytest.raises(ValueError, match="side"):
        fb.side_chain("X")


def test_neutral_pose_is_zero_for_every_component() -> None:
    for side in ("L", "R"):
        res = _angles(side, {})
        for arr in (
            res.wrist.flexion_rad,
            res.wrist.ulnar_deviation_rad,
            res.wrist.axial_rotation_rad,
            res.forearm.flexion_rad,
            res.forearm.pronation_rad,
        ):
            assert np.allclose(arr, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("coordinate", "field", "expected"),
    [
        # HRG-00: LWInputY + = left wrist extension; RWInputY + = right flexion.
        ("LWInputY", "flexion_rad", -THETA),
        ("RWInputY", "flexion_rad", THETA),
        # *WInputX + = radial deviation (cock), ulnar negative, both sides.
        ("LWInputX", "ulnar_deviation_rad", -THETA),
        ("RWInputX", "ulnar_deviation_rad", -THETA),
    ],
)
def test_wrist_signs_match_the_hrg00_sign_map(
    coordinate: str, field: str, expected: float
) -> None:
    side = coordinate[0]
    w = _angles(side, {coordinate: THETA}).wrist
    assert getattr(w, field)[0] == pytest.approx(expected, abs=1e-9)
    others = {"flexion_rad", "ulnar_deviation_rad", "axial_rotation_rad"} - {field}
    for other in others:
        assert getattr(w, other)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("coordinate", "expected"), [("RFInput", THETA), ("LFInput", -THETA)]
)
def test_forearm_pronation_is_separate_and_signed(
    coordinate: str, expected: float
) -> None:
    res = _angles(coordinate[0], {coordinate: THETA})
    assert res.forearm.pronation_rad[0] == pytest.approx(expected, abs=1e-9)
    # Pronation turns the hand with the forearm: the wrist itself is neutral.
    assert res.wrist.axial_rotation_rad[0] == pytest.approx(0.0, abs=1e-9)
    assert res.wrist.flexion_rad[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("coordinate", ["LEInput", "REInput"])
def test_elbow_extension_is_negative_isb_flexion(coordinate: str) -> None:
    res = _angles(coordinate[0], {coordinate: THETA})
    assert res.forearm.flexion_rad[0] == pytest.approx(-THETA, abs=1e-9)
    assert res.forearm.pronation_rad[0] == pytest.approx(0.0, abs=1e-9)


def test_mirrored_motion_gives_identical_anatomical_values() -> None:
    # Mirror across the sagittal plane: the shared +z wrist-Y axis and -x
    # forearm axis flip sense on the left, the -y cock axis does not.
    left = _angles(
        "L", {"LWInputY": -0.3, "LWInputX": 0.2, "LFInput": -0.4, "LEInput": -0.5}
    )
    right = _angles(
        "R", {"RWInputY": 0.3, "RWInputX": 0.2, "RFInput": 0.4, "REInput": -0.5}
    )
    for a, b in (
        (left.wrist.flexion_rad, right.wrist.flexion_rad),
        (left.wrist.ulnar_deviation_rad, right.wrist.ulnar_deviation_rad),
        (left.wrist.axial_rotation_rad, right.wrist.axial_rotation_rad),
        (left.forearm.pronation_rad, right.forearm.pronation_rad),
        (left.forearm.flexion_rad, right.forearm.flexion_rad),
    ):
        assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert right.wrist.flexion_rad[0] > 0.0
    assert right.forearm.pronation_rad[0] > 0.0


def test_engine_body_frames_and_world_placement_do_not_change_angles() -> None:
    rng = np.random.default_rng(13169)
    q = {"RWInputY": 0.3, "RWInputX": -0.2, "RFInput": 0.4, "REInput": -0.7}
    chain = fb.side_chain("R")
    offsets = {
        b: Rotation.random(random_state=rng).as_matrix()
        for b in (chain.humerus, chain.forearm_body, chain.hand)
    }
    pelvis = Rotation.random(random_state=rng).as_matrix()
    plain = _angles("R", q)
    moved = _angles("R", q, body_offsets=offsets, pelvis=pelvis)
    for a, b in (
        (plain.wrist.flexion_rad, moved.wrist.flexion_rad),
        (plain.wrist.ulnar_deviation_rad, moved.wrist.ulnar_deviation_rad),
        (plain.wrist.axial_rotation_rad, moved.wrist.axial_rotation_rad),
        (plain.forearm.pronation_rad, moved.forearm.pronation_rad),
    ):
        assert np.allclose(a, b, atol=1e-10)


def test_series_carry_observability_receipt_and_calibration_id() -> None:
    res = _angles("L", {"LWInputY": THETA})
    assert res.wrist.observability == OBSERVABILITY == "fitted_grip_rotation"
    assert res.wrist.receipt_path == RECEIPT
    assert res.forearm.receipt_path == RECEIPT
    assert all(c.startswith(fb.CALIBRATION_ID) for c in res.wrist.calibration)


def test_missing_or_misshaped_bodies_raise() -> None:
    neutral = _synthetic("R", {})
    cal = fb.calibrate_neutral("R", neutral)
    with pytest.raises(KeyError, match="RHandStandoff"):
        fb.calibrate_neutral(
            "R", {k: v for k, v in neutral.items() if k != "RHandStandoff"}
        )
    bad = {k: np.broadcast_to(v, (2, 3, 3)) for k, v in neutral.items()}
    bad["RHandStandoff"] = neutral["RHandStandoff"][None]
    with pytest.raises(ValueError, match="shape"):
        fb.full_body_v1_wrist_forearm(cal, bad, receipt_path=RECEIPT)
    with pytest.raises(ValueError, match="receipt"):
        fb.full_body_v1_wrist_forearm(
            cal, {k: v[None] for k, v in neutral.items()}, receipt_path=""
        )


def test_only_the_rolled_wrist_family_is_accepted() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    native = json.loads(NATIVE_PATH.read_text(encoding="utf-8"))
    fb.check_spec_family(spec, native)  # anthropometric, rolled: accepted
    with pytest.raises(ValueError, match="WRIST_ROLL"):
        fb.check_spec_family(native, native)


# --- compiled MuJoCo model ---------------------------------------------------


def _mujoco_rotations(
    model: Any, data: Any, q_by_name: dict[str, float], bodies: list[str]
) -> dict[str, np.ndarray]:
    import mujoco

    qpos = np.zeros(model.nq)
    for name, value in q_by_name.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[model.jnt_qposadr[joint]] = value
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    out = {}
    for b in range(model.nbody):
        short = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b).rsplit("/", 1)[-1]
        if short in bodies:
            out[short] = data.xmat[b].reshape(3, 3).copy()
    out["__pelvis__"] = data.xmat[1].reshape(3, 3).copy()
    return out


@pytest.mark.parametrize(
    ("coordinate", "group", "field", "expected"),
    [
        ("LWInputY", "wrist", "flexion_rad", -THETA),
        ("RWInputY", "wrist", "flexion_rad", THETA),
        ("LWInputX", "wrist", "ulnar_deviation_rad", -THETA),
        ("RWInputX", "wrist", "ulnar_deviation_rad", -THETA),
        ("LFInput", "forearm", "pronation_rad", -THETA),
        ("RFInput", "forearm", "pronation_rad", THETA),
        ("LEInput", "forearm", "flexion_rad", -THETA),
        ("REInput", "forearm", "flexion_rad", -THETA),
    ],
)
def test_compiled_fixture_matches_the_declared_calibration(
    coordinate: str, group: str, field: str, expected: float
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(SPEC_PATH.read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    side = coordinate[0]
    chain = fb.side_chain(side)
    bodies = [chain.humerus, chain.forearm_body, chain.hand]
    neutral = _mujoco_rotations(model, data, {}, bodies)
    pelvis = neutral.pop("__pelvis__")
    cal = fb.calibrate_neutral(side, neutral, pelvis_rotation=pelvis)
    posed = _mujoco_rotations(model, data, {coordinate: THETA}, bodies)
    posed.pop("__pelvis__")
    res = fb.full_body_v1_wrist_forearm(
        cal, {k: v[None] for k, v in posed.items()}, receipt_path=RECEIPT
    )
    assert getattr(getattr(res, group), field)[0] == pytest.approx(expected, abs=1e-6)


# --- private best fit (report only) -------------------------------------------


def test_best_fit_wrist_forearm_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print best-fit wrist/forearm extremes when ``HRG_BEST_FIT_BUNDLE`` is set."""
    location = os.environ.get("HRG_BEST_FIT_BUNDLE")
    if not location or not Path(location).is_file():
        pytest.skip("HRG_BEST_FIT_BUNDLE is not set to a bundle file")
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    with np.load(location, allow_pickle=False) as bundle:
        spec_bytes = bytes(bundle["spec"])
        q = np.asarray(bundle["reference_q"], dtype=float)
    names = list(json.loads(spec_bytes.decode("utf-8"))["coordinate_order"])
    xml, _ = export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with capsys.disabled():
        print(f"\nHRG-E2 best-fit wrist/forearm report ({len(q)} frames, degrees)")
        for side in ("L", "R"):
            chain = fb.side_chain(side)
            bodies = [chain.humerus, chain.forearm_body, chain.hand]
            neutral = _mujoco_rotations(model, data, {}, bodies)
            cal = fb.calibrate_neutral(
                side, neutral, pelvis_rotation=neutral.pop("__pelvis__")
            )
            frames = [
                _mujoco_rotations(
                    model, data, dict(zip(names, row, strict=True)), bodies
                )
                for row in q
            ]
            series = {b: np.stack([f[b] for f in frames]) for b in bodies}
            res = fb.full_body_v1_wrist_forearm(cal, series, receipt_path=location)
            for label, arr in (
                ("wrist flexion", res.wrist.flexion_deg),
                ("ulnar deviation", res.wrist.ulnar_deviation_deg),
                ("wrist axial", res.wrist.axial_rotation_deg),
                ("forearm pronation", res.forearm.pronation_deg),
            ):
                print(
                    f"  {side} {label:18s} address {arr[0]:7.1f}"
                    f"  min {np.nanmin(arr):7.1f}  max {np.nanmax(arr):7.1f}"
                )
    assert q.shape[1] == len(names)
