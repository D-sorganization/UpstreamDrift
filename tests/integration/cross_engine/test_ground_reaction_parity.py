"""Cross-engine ground-reaction parity (GCV-2, #11708).

Every engine stands the same 72 kg, two-footed stance (feet at +/-0.15 m on the
world Y axis, Z-up) and is judged only by its provider frame through the one
shared ground-reaction breakdown labels (``contact:grf_<left|right|net>``).

Quantities compared, all against the analytic statics of the shared scenario:

* net vertical GRF equals body weight (2 % relative, the contact-compliance
  bound used by ``test_force_overlay_parity.py`` for settled compliant contact);
* the two feet carry the load symmetrically (each within 5 % of half weight; the
  stance is left/right symmetric, the residual is contact-settling asymmetry);
* the net CoP lies on the ground plane at the mid-line between the feet
  (|y| < 1 cm, |x| < 5 cm);
* the left foot is on world +Y in every engine (the OpenSim Y-up mapping).

Pinocchio has no contact model: it is fed equilibrium ``ContactSample`` records
of the repo's shared contact law and reports the breakdown from them.  Simscape
is file-based and out of scope: it reports the ground reaction unavailable.
Rows for engines that are not installed skip with an explicit reason.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from _pytest.outcomes import Skipped

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
WEIGHT_RTOL = 0.02
FOOT_SPLIT_RTOL = 0.05
COP_Y_ATOL_M = 0.01
COP_X_ATOL_M = 0.05

_MODULES = {
    "mujoco": "tests/unit/engines/mujoco/test_ground_reaction_wiring.py",
    "drake": "tests/engines/drake/test_drake_ground_reaction.py",
    "pinocchio": "tests/unit/engines/pinocchio/test_pinocchio_ground_reaction.py",
    "opensim": "tests/unit/engines/opensim/test_opensim_ground_reaction.py",
}


@functools.cache
def _load(engine: str) -> ModuleType:
    # Executed once per engine: re-executing a module that imports pydrake or
    # pinocchio under a fresh name breaks their native submodule imports.
    path = ROOT / _MODULES[engine]
    spec = importlib.util.spec_from_file_location(f"_gcv2_{engine}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Skipped as exc:  # module-level importorskip of the engine
        pytest.skip(f"{engine} unavailable: {exc}")
    return module


def _labels(frame) -> dict:
    return {w.label: w for w in frame.wrenches}


def _mujoco(_tmp: Path):
    mod = _load("mujoco")  # skips before the engine-specific import below
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model, data = mod.build_stance()
    return _labels(MujocoForceTorqueSource(model).sample(data)), mod.WEIGHT_N


def _drake(_tmp: Path):
    mod = _load("drake")
    from src.engines.physics_engines.drake.python.drake_force_torque import (
        DrakeForceTorqueSource,
    )

    plant, diagram, pctx = mod.build_stance()
    frame = DrakeForceTorqueSource(plant, diagram).sample(pctx)
    return _labels(frame), mod.WEIGHT_N


def _opensim(tmp: Path):
    mod = _load("opensim")
    from src.engines.physics_engines.opensim.python.opensim_force_torque import (
        OpenSimForceTorqueSource,
    )

    model, state = mod.build_stance(tmp)
    return _labels(OpenSimForceTorqueSource(model).sample(state)), mod.WEIGHT_N


def _pinocchio(_tmp: Path):
    mod = _load("pinocchio")
    source, q = mod.build_stance()
    return _labels(mod._frame(source, q, mod._stance_samples())), mod.WEIGHT_N


_BUILDERS: dict[str, Callable[[Path], tuple[dict, float]]] = {
    "mujoco": _mujoco,
    "drake": _drake,
    "pinocchio": _pinocchio,
    "opensim": _opensim,
}
ENGINES = [
    pytest.param(name, marks=getattr(pytest.mark, f"requires_{name}"))
    for name in _BUILDERS
]


@pytest.fixture(params=ENGINES, scope="module")
def breakdown(request, tmp_path_factory):
    wrenches, weight = _BUILDERS[request.param](tmp_path_factory.mktemp("grf"))
    return request.param, wrenches, weight


def test_net_grf_equals_body_weight(breakdown) -> None:
    engine, w, weight = breakdown
    net = np.asarray(w["contact:grf_net"].force_n)
    assert net[2] == pytest.approx(weight, rel=WEIGHT_RTOL), engine


def test_feet_share_the_load_symmetrically(breakdown) -> None:
    engine, w, weight = breakdown
    for side in ("left", "right"):
        fz = w[f"contact:grf_{side}"].force_n[2]
        assert fz == pytest.approx(weight / 2, rel=FOOT_SPLIT_RTOL), (engine, side)


def test_net_cop_is_on_the_ground_between_the_feet(breakdown) -> None:
    engine, w, _ = breakdown
    x, y, z = w["contact:grf_net"].point_m
    assert z == pytest.approx(0.0, abs=1e-9), engine
    assert abs(y) < COP_Y_ATOL_M and abs(x) < COP_X_ATOL_M, (engine, x, y)


def test_left_foot_is_on_world_positive_y(breakdown) -> None:
    engine, w, _ = breakdown
    assert w["contact:grf_left"].point_m[1] > 0.0 > w["contact:grf_right"].point_m[1]


def test_free_and_com_moments_are_reported(breakdown) -> None:
    engine, w, _ = breakdown
    for label in ("contact:free_moment_net", "contact:moment_com_net"):
        assert label in w, (engine, label)


def test_simscape_reports_the_ground_reaction_unavailable() -> None:
    from src.engines.simscape import force_channels

    assert not hasattr(force_channels, "ground_reaction_wrenches"), (
        "Simscape is file-based and out of scope for GCV-2; if it gains a "
        "ground-reaction path it needs a parity row"
    )
