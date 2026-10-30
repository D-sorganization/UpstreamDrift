"""HRG-Q1 limits: distinguish hard range gaps from report-only trunk ROM.

This row qualifies engine configuration, not physiological validity. Known
hard-limit gaps remain strict xfails; missing engines and the thorax-only
MyoSuite mapping are unavailable, never successful qualification cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import full_body_v2_geometry as geo
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
ENGINES = ("mujoco", "pinocchio", "drake", "opensim", "myosuite")


@pytest.fixture(scope="module")
def spec() -> dict:
    path = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    return geo.derive_landmark_v2_spec(
        json.loads(path.read_text(encoding="utf-8")),
        provenance="HRG-Q1 limits qualification",
    )


def _mujoco_limits(spec: dict) -> dict:
    mj = pytest.importorskip("mujoco")
    if not isinstance(getattr(mj, "MjModel", None), type):
        pytest.skip("MuJoCo stub has no MjModel")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(json.dumps(spec, sort_keys=True).encode())
    model = mj.MjModel.from_xml_string(xml)
    return {
        name: (model.joint(name).range.copy(), bool(model.joint(name).limited[0]))
        for name in spec["coordinate_order"]
    }


def _pinocchio_limits(spec: dict) -> dict:
    pin = pytest.importorskip("pinocchio")
    if not isinstance(getattr(pin, "Model", None), type):
        pytest.skip("Pinocchio stub has no Model")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(spec).model
    result = {}
    for name in spec["coordinate_order"]:
        index = model.joints[model.getJointId(name)].idx_q
        bounds = np.array(
            [model.lowerPositionLimit[index], model.upperPositionLimit[index]]
        )
        # Pinocchio represents an absent bound by +/- DBL_MAX, not infinity.
        bounded = np.all(np.abs(bounds) < np.finfo(float).max)
        result[name] = (bounds, bool(bounded))
    return result


def _drake_limits(spec: dict) -> dict:
    api = pytest.importorskip("pydrake.all")
    if not isinstance(getattr(api, "MultibodyPlant", None), type):
        pytest.skip("Drake stub has no MultibodyPlant")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    plant = FullBodyDrakeModel(spec).plant
    result = {}
    for name in spec["coordinate_order"]:
        joint = plant.GetJointByName(name)
        bounds = np.array(
            [joint.position_lower_limits()[0], joint.position_upper_limits()[0]]
        )
        result[name] = (bounds, bool(np.all(np.isfinite(bounds))))
    return result


def _opensim_limits(spec: dict, directory: Path) -> dict:
    osim = require_opensim()
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml, _ = export_full_body_osim(
        json.dumps(spec, sort_keys=True).encode(), club_geometry_ref=None
    )
    path = directory / "limits.osim"
    path.write_text(xml, encoding="utf-8")
    model = osim.Model(str(path))
    state = model.initSystem()
    coords = model.getCoordinateSet()
    return {
        name: (
            np.array([coords.get(name).getRangeMin(), coords.get(name).getRangeMax()]),
            bool(coords.get(name).getClamped(state)),
        )
        for name in spec["coordinate_order"]
    }


@pytest.fixture(scope="module", params=ENGINES)
def engine_limits(request, spec, tmp_path_factory):
    engine = request.param
    if engine == "myosuite":
        status = geo.EXPORT_STATUS[engine]
        assert status["status"] == "mapped_thorax_only"
        pytest.skip(f"MyoSuite has no v2 joint-limit export: {status['reason']}")
    if engine == "opensim":
        limits = _opensim_limits(spec, tmp_path_factory.mktemp("limits"))
    else:
        loaders = {
            "mujoco": _mujoco_limits,
            "pinocchio": _pinocchio_limits,
            "drake": _drake_limits,
        }
        limits = loaders[engine](spec)
    assert set(limits) == set(spec["coordinate_order"])
    return engine, limits


def test_trunk_soft_rom_is_not_enforced_as_hard_limits(spec, engine_limits):
    engine, limits = engine_limits
    assert spec["soft_limits_deg"]
    for name in spec["soft_limits_deg"]:
        assert name not in spec["coordinate_ranges_deg"]
        _, enforced = limits[name]
        assert not enforced, f"{engine} clamps report-only trunk coordinate {name}"


def test_joint_limits_match_spec_ranges(spec, engine_limits):
    engine, limits = engine_limits
    mismatches = []
    for name, degrees in spec["coordinate_ranges_deg"].items():
        expected = np.radians(degrees)
        actual, enforced = limits[name]
        if not enforced or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            mismatches.append(
                f"{name}: expected={expected.tolist()}, actual={actual.tolist()}, "
                f"enforced={enforced}"
            )
    assert not mismatches, (
        f"{engine}: {len(mismatches)}/{len(spec['coordinate_ranges_deg'])} "
        "hard ranges unqualified (radians):\n" + "\n".join(mismatches)
    )
