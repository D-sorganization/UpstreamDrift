"""Static tibial-torsion foot-yaw offset in the shared spec (R3b, #12789)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import tibial_torsion as tt
from src.shared.python.motion_matching.identifiability import (
    body_poses_from_coordinates,
)

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[3]
_SPECS = _ROOT / "docs/development/full_body_models"
_CANONICAL = ("full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json")


def _load(name: str) -> dict:
    return json.loads((_SPECS / name).read_text(encoding="utf-8"))


def _generic(name: str) -> dict:
    """The committed document with its torsion offset undone (generic leg)."""
    document = _load(name)
    record = document.pop(tt.TIBIAL_TORSION_KEY, None)
    if record is None:
        return document
    for side in tt.SIDES:
        joint = next(j for j in document["joints"] if j["name"] == f"ankle_{side}")
        base = np.asarray(joint["parent_to_base"], dtype=float)
        axis = tt._shank_axis(document, side)
        sign = tt._lateral_sign(document, side)
        rot = tt._rotation(axis, -sign * math.radians(record[side]))
        base[:3, :3] = rot @ base[:3, :3]
        joint["parent_to_base"] = base.tolist()
    return document


def _foot_yaw_deg(document: dict) -> dict[str, float]:
    """Zero-pose calcn -> toes yaw in the spec world, positive = toes out."""
    poses = body_poses_from_coordinates(document, {})
    out = {}
    for side, lateral in (("r", -1.0), ("l", 1.0)):  # spec world: +y is left
        axis = poses[f"toes_{side}"][:3, 3] - poses[f"calcn_{side}"][:3, 3]
        out[side] = math.degrees(math.atan2(lateral * axis[1], axis[0]))
    return out


# --------------------------------------------------------------- constants


def test_default_puts_the_generic_leg_inside_the_documented_adult_range() -> None:
    low, high = tt.ADULT_TIBIAL_TORSION_RANGE_DEG
    total = tt.GENERIC_LEG_TORSION_DEG + tt.DEFAULT_TIBIAL_TORSION_OFFSET_DEG
    assert low <= total <= high
    assert (0.0, high - tt.GENERIC_LEG_TORSION_DEG) == tt.OFFSET_RANGE_DEG
    assert "doi:10.1007/s004020050459" in tt.TIBIAL_TORSION_SOURCE


@pytest.mark.parametrize("name", _CANONICAL)
@pytest.mark.parametrize("side", tt.SIDES)
def test_generic_leg_torsion_constant_is_measured_from_the_spec(
    name: str, side: str
) -> None:
    measured = tt.malleolar_torsion_deg(_generic(name), side)
    assert measured == pytest.approx(tt.GENERIC_LEG_TORSION_DEG, abs=0.01)


# ------------------------------------------------------------ application


@pytest.mark.parametrize("name", _CANONICAL)
def test_zero_offset_is_bit_identical(name: str) -> None:
    generic = _generic(name)
    out = tt.apply_tibial_torsion(generic, {"r": 0.0, "l": 0.0})
    assert tt.TIBIAL_TORSION_KEY not in out
    assert json.dumps(out, sort_keys=True) == json.dumps(generic, sort_keys=True)


@pytest.mark.parametrize("name", _CANONICAL)
def test_offset_turns_each_foot_out_about_a_fixed_ankle_centre(name: str) -> None:
    generic = _generic(name)
    out = tt.apply_tibial_torsion(generic, {"r": 10.0, "l": 7.0})
    before, after = _foot_yaw_deg(generic), _foot_yaw_deg(out)
    assert after["r"] - before["r"] == pytest.approx(10.0, abs=0.3)
    assert after["l"] - before["l"] == pytest.approx(7.0, abs=0.3)
    for side, offset in (("r", 10.0), ("l", 7.0)):
        assert tt.malleolar_torsion_deg(out, side) == pytest.approx(
            tt.GENERIC_LEG_TORSION_DEG + offset, abs=0.01
        )
    p0 = body_poses_from_coordinates(generic, {})
    p1 = body_poses_from_coordinates(out, {})
    for side in tt.SIDES:
        np.testing.assert_allclose(
            p1[f"talus_{side}"][:3, 3], p0[f"talus_{side}"][:3, 3], atol=1e-12
        )


def test_only_the_ankle_base_frames_change_and_no_coordinate_is_added() -> None:
    generic = _generic(_CANONICAL[1])
    out = tt.apply_tibial_torsion(generic, {"r": 10.0, "l": 10.0})
    assert out["coordinate_order"] == generic["coordinate_order"]
    changed = [
        (a["name"], key)
        for a, b in zip(generic["joints"], out["joints"], strict=True)
        for key in a
        if a[key] != b[key]
    ]
    assert sorted(changed) == [
        ("ankle_l", "parent_to_base"),
        ("ankle_r", "parent_to_base"),
    ]
    assert out[tt.TIBIAL_TORSION_KEY] == {
        "r": 10.0,
        "l": 10.0,
        "source": tt.TIBIAL_TORSION_SOURCE,
    }
    assert tt.TIBIAL_TORSION_KEY not in generic  # input not mutated


# ------------------------------------------------------------------ DbC


@pytest.mark.parametrize(
    "offsets",
    [
        {"r": math.nan, "l": 0.0},
        {"r": math.inf, "l": 0.0},
        {"r": -1.0, "l": 0.0},
        {"r": 0.0, "l": tt.OFFSET_RANGE_DEG[1] + 0.1},
        {"r": 10.0},
        {"r": 10.0, "l": 10.0, "x": 1.0},
    ],
)
def test_apply_rejects_invalid_offsets(offsets: dict) -> None:
    with pytest.raises(ValueError):
        tt.apply_tibial_torsion(_generic(_CANONICAL[0]), offsets)


def test_apply_refuses_a_second_application() -> None:
    once = tt.apply_tibial_torsion(_generic(_CANONICAL[0]), {"r": 5.0, "l": 5.0})
    with pytest.raises(ValueError, match="already"):
        tt.apply_tibial_torsion(once, {"r": 5.0, "l": 5.0})


def test_validate_rejects_out_of_range_or_unsourced_records() -> None:
    document = tt.apply_tibial_torsion(_generic(_CANONICAL[0]), {"r": 10.0, "l": 10.0})
    tt.validate_tibial_torsion(document)
    tt.validate_tibial_torsion(_generic(_CANONICAL[0]))  # generic leg is valid
    bad = copy.deepcopy(document)
    bad[tt.TIBIAL_TORSION_KEY]["l"] = 80.0
    with pytest.raises(ValueError, match="offset"):
        tt.validate_tibial_torsion(bad)
    unsourced = copy.deepcopy(document)
    unsourced[tt.TIBIAL_TORSION_KEY]["source"] = " "
    with pytest.raises(ValueError, match="source"):
        tt.validate_tibial_torsion(unsourced)
    with pytest.raises(ValueError, match="side"):
        tt.malleolar_torsion_deg(document, "x")


# --------------------------------------------------- committed documents


@pytest.mark.parametrize("name", _CANONICAL)
def test_committed_specs_carry_the_default_offset(name: str) -> None:
    """The canonical driver and 7-iron documents are the shared source."""
    document = _load(name)
    record = document[tt.TIBIAL_TORSION_KEY]
    assert record == {
        "r": tt.DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
        "l": tt.DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
        "source": tt.TIBIAL_TORSION_SOURCE,
    }
    tt.validate_tibial_torsion(document)
    for side in tt.SIDES:
        assert tt.malleolar_torsion_deg(document, side) == pytest.approx(
            tt.GENERIC_LEG_TORSION_DEG + tt.DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
            abs=0.01,
        )
    assert "#12789" in document["provenance"]


def test_apply_script_is_idempotent_on_the_committed_layout(tmp_path: Path) -> None:
    from scripts.apply_tibial_torsion import apply_file

    path = tmp_path / "spec.json"
    path.write_bytes((_SPECS / _CANONICAL[1]).read_bytes())
    assert apply_file(path) is False  # already applied
    with pytest.raises(ValueError, match="already records"):
        apply_file(path, 5.0)


def test_builder_applies_the_default_and_zero_keeps_the_generic_leg() -> None:
    """The spec builder is the one place new documents get the offset."""
    from src.shared.python.motion_matching.execution.spec_builder import (
        compose_anthropometric_document,
    )

    generic = compose_anthropometric_document(tibial_torsion_offset_deg=0.0).document
    default = compose_anthropometric_document().document
    assert tt.TIBIAL_TORSION_KEY not in generic
    offset = tt.DEFAULT_TIBIAL_TORSION_OFFSET_DEG
    expected = tt.apply_tibial_torsion(generic, {"r": offset, "l": offset})
    assert json.dumps(default, sort_keys=True) == json.dumps(expected, sort_keys=True)


# ------------------------------------------------------------ cross-engine

_FEET = {
    f"{b}_{s}": (f"{b}_{s}", (0.0, 0.0, 0.0)) for b in ("calcn", "toes") for s in "rl"
}


def _engine_foot_yaw_deg(engine: str, document: dict) -> dict[str, float]:
    from src.shared.python.motion_matching.pipeline.plant import (
        EngineUnavailableError,
        get_plant,
    )

    try:
        plant = get_plant(engine, document)
    except EngineUnavailableError as exc:
        pytest.skip(f"{engine} unavailable: {exc}")
    try:
        poses = plant.frame_poses(_FEET, np.zeros(len(plant.coordinate_order)))
    except NotImplementedError as exc:  # e.g. OpenSim without native geometry
        pytest.skip(f"{engine} frame poses unavailable: {exc}")
    out = {}
    for side, lateral in (("r", -1.0), ("l", 1.0)):
        axis = poses[f"toes_{side}"][1] - poses[f"calcn_{side}"][1]
        out[side] = math.degrees(math.atan2(lateral * axis[1], axis[0]))
    return out


@pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio", "opensim"])
def test_every_engine_turns_the_feet_like_the_shared_kinematics(engine: str) -> None:
    """Engines consume the offset through the spec's ankle placement (DRY)."""
    generic = _generic(_CANONICAL[1])
    torsion = tt.apply_tibial_torsion(generic, {"r": 10.0, "l": 10.0})
    shared = {
        s: _foot_yaw_deg(torsion)[s] - _foot_yaw_deg(generic)[s] for s in tt.SIDES
    }
    before = _engine_foot_yaw_deg(engine, generic)
    after = _engine_foot_yaw_deg(engine, torsion)
    for side in tt.SIDES:
        assert after[side] - before[side] == pytest.approx(shared[side], abs=0.05)


# ---------------------------------------- canonical address (live, opt-in)

_ADDRESS_ARGS = {
    "driver": ["--static-seeds", "--fix-club-markers", "--capture", "driver"],
    "iron": [
        "--static-seeds",
        "--zmp-filter",
        "--fix-club-markers",
        "--capture",
        "iron",
    ],
}
_ADDRESS_SPEC = {"driver": _CANONICAL[0], "iron": _CANONICAL[1]}
#: Acceptance band of #12789 for the trail style target in ``style`` mode.
_STYLE_BAND_DEG = 3.0


@pytest.mark.slow
@pytest.mark.live_simulation
@pytest.mark.parametrize("capture", sorted(_ADDRESS_ARGS))
def test_canonical_address_reaches_the_trail_target_inside_the_hip_limit(
    capture: str, tmp_path: Path
) -> None:
    """#12789 acceptance 1: 20 +/- 3 deg trail toe-out, hip off its limit."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.foot_progression import (
        TRAIL_TOE_OUT_STYLE_DEG,
    )
    from src.shared.python.motion_matching.pipeline import cli
    from src.shared.python.motion_matching.pipeline.address_feet import (
        FootTargets,
        model_feet_deg,
    )

    spec = _SPECS / _ADDRESS_SPEC[capture]
    argv = [*_ADDRESS_ARGS[capture], "--spec", str(spec), "--out", str(tmp_path)]
    result = cli.calibrate_run(cli.build_parser().parse_args(argv)).cal_res
    probe = FootTargets(
        {"left": 0.0, "right": 0.0}, {"left": False, "right": False}, "probe"
    )
    feet = model_feet_deg(result.kin, result.address2.q, probe)
    assert feet["right"] == pytest.approx(TRAIL_TOE_OUT_STYLE_DEG, abs=_STYLE_BAND_DEG)
    names = list(result.kin.coordinate_order)
    hip = math.degrees(result.address2.q[names.index("hip_rotation_r")])
    low, high = _load(_ADDRESS_SPEC[capture])["coordinate_ranges_deg"]["hip_rotation_r"]
    assert low + 1.0 < hip < high - 1.0
