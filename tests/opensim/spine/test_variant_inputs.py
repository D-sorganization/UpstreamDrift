"""Movement variants as golf trunk inputs and model range checks (SPN-13b, #12767).

The swing is generated from an invented coupling law on the synthetic fixture
(as in ``test_drive.py``); every number is invented.  Pure adapter tests run
without OpenSim; the model-range tests load the fixture natively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    DriveOptions,
    drive_segmental_model,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    GolfTrunkInput,
)
from src.engines.physics_engines.opensim.python.spine.variant_inputs import (
    VariantGolfInput,
    check_model_ranges,
    event_indices,
    model_coordinate_ranges,
    variant_golf_input,
)
from src.shared.python.spine.coupling import LinearCouplingLaw
from src.shared.python.spine.golf_terms import evaluate
from src.shared.python.spine.levels import SpinalLevel, levels_between
from src.shared.python.spine.movement_variants import movement_variant
from tests.helpers.opensim_support import require_opensim

PHI = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
N = 41
EVENT_TIMES = {"address": 0.1, "top": 0.5, "impact": 0.75}


def _law() -> LinearCouplingLaw:
    mats = {}
    for k, lv in enumerate(LUMBAR):
        w = 0.1 + 0.05 * k
        mats[lv] = np.array(
            [[w, 0.03 * (k + 1), 0.0], [0.0, w, 0.02 * (k + 1)], [0.0, 0.0, w / 2]]
        )
    return LinearCouplingLaw("invented", "1", mats, PHI, "invented, SPN-13b test")


def _golf_input(events: dict[str, float] | None = None) -> GolfTrunkInput:
    t = np.linspace(0.0, 1.0, N)
    s = t / t[-1]
    angles = np.column_stack(
        [
            -0.2 + 0.1 * np.sin(np.pi * s),
            0.08 * np.sin(2.0 * np.pi * s),
            -0.5 * np.exp(-(((s - 0.5) / 0.2) ** 2)),
        ]
    )
    rots = Rotation.from_euler("ZXY", angles).as_matrix()
    return GolfTrunkInput(
        time_s=t,
        pelvis_to_thorax=rots,
        pelvis_pose_ground=np.tile(np.eye(4), (N, 1, 1)),
        other_coordinates={"arm_flex_r": np.linspace(0.0, 1.0, N)},
        events=EVENT_TIMES if events is None else events,
        source={"file": "synthetic.sto", "sha256": "a" * 64},
    )


# -------------------------------------------------------------- pure adapter


@pytest.mark.unit
def test_event_indices_are_nearest_samples() -> None:
    idx = event_indices(_golf_input())
    assert idx == {"address": 4, "top": 20, "impact": 30}


@pytest.mark.unit
def test_event_indices_reject_out_of_range_time() -> None:
    with pytest.raises(ValueError, match="outside"):
        event_indices(_golf_input({"address": -0.5, "top": 0.5, "impact": 0.75}))
    with pytest.raises(TypeError, match="GolfTrunkInput"):
        event_indices("swing")  # type: ignore[arg-type]


@pytest.mark.unit
def test_variant_golf_input_holds_other_coordinates_and_records_source() -> None:
    golf = _golf_input()
    out = variant_golf_input(
        golf, movement_variant("x_factor_top_scale"), handedness="right", parameter=1.2
    )
    assert isinstance(out, VariantGolfInput)
    new = out.golf_input
    assert isinstance(new, GolfTrunkInput)
    np.testing.assert_array_equal(new.time_s, golf.time_s)
    np.testing.assert_array_equal(new.pelvis_pose_ground, golf.pelvis_pose_ground)
    np.testing.assert_array_equal(
        new.other_coordinates["arm_flex_r"], golf.other_coordinates["arm_flex_r"]
    )
    assert dict(new.events) == dict(golf.events)
    assert new.source["file"] == "synthetic.sto"
    assert new.source["sha256"] == "a" * 64
    assert new.source["variant"] == out.result.distribution_source
    base = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": golf.pelvis_to_thorax},
        handedness="right",
    )
    after = evaluate(
        "x_factor_jcs_axial",
        {"pelvis_to_thorax_rotation": new.pelvis_to_thorax},
        handedness="right",
    )
    assert abs(after[20] - 1.2 * base[20]) <= 1e-6


@pytest.mark.unit
def test_variant_golf_input_contracts() -> None:
    var = movement_variant("x_factor_top_scale")
    with pytest.raises(TypeError, match="GolfTrunkInput"):
        variant_golf_input("swing", var, handedness="right")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="impact"):
        variant_golf_input(
            _golf_input({"address": 0.1, "top": 0.5}), var, handedness="right"
        )


# ------------------------------------------------------- native range check


@pytest.fixture(scope="module")
def fixture_model(tmp_path_factory) -> tuple[FixtureSpec, Path]:
    require_opensim()
    spec = FixtureSpec()
    path = write_fixture_osim(
        spec, tmp_path_factory.mktemp("variants") / "fixture.osim", coupling=_law()
    )
    return spec, path


def _coordinate_names(spec: FixtureSpec) -> list[str]:
    return [c for lv in fixture_level_names(spec) for c in lv.coordinates]


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_model_coordinate_ranges_read_natively(fixture_model) -> None:
    spec, path = fixture_model
    names = _coordinate_names(spec)
    ranges = model_coordinate_ranges(path, names)
    assert set(ranges) == set(names)
    for lo, hi in ranges.values():
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no_such_coordinate"):
        model_coordinate_ranges(path, ["no_such_coordinate"])


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_driven_variant_is_checked_against_model_ranges(fixture_model) -> None:
    spec, path = fixture_model
    out = variant_golf_input(
        _golf_input(),
        movement_variant("x_factor_top_scale"),
        handedness="right",
        parameter=1.2,
    )
    drive = drive_segmental_model(
        out.golf_input,
        model_path=path,
        model_id="fixture",
        law=_law(),
        options=DriveOptions(pelvis=FIXTURE_PELVIS),
    )
    assert drive.input_source["variant"] == out.result.distribution_source
    level_coords = {
        name: values
        for name, values in drive.coordinates.items()
        if name in set(_coordinate_names(spec))
    }
    check = check_model_ranges(path, level_coords)
    assert check.ok is True
    forced = dict(level_coords)
    first = _coordinate_names(spec)[0]
    forced[first] = np.full(N, 1.5)
    bad = check_model_ranges(path, forced)
    assert bad.ok is False
    assert {v.coordinate for v in bad.violations} == {first}
