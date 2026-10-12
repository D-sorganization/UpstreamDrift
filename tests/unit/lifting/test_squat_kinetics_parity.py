"""Back-squat kinetics parity across engines: GRF and hand wrench (LIFT-7, #11747).

Every engine evaluates the same states of the committed back-squat same-input
bundle (LIFT-5): the shared contact law on its own kinematics gives the
ground-reaction breakdown (GCV-1: per foot and net force, CoP, free moment),
and the closure KKT multiplier gives the per-hand wrench on the grip (GCV-7/8).

Tolerances were fixed before measuring and are never loosened.  They are the
acceleration-parity level L1 of epic #11605 that the GCV-8 grip test already
uses, ``|dX| <= ATOL + RTOL * max|X|``, plus a CoP bound derived from it
(CoP = moment / Fz with Fz of several hundred newtons).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
from src.shared.python.lifting.kinetics_parity import (
    GRIP_ENGINES,
    KineticsSample,
    bundle_kinetics,
    kinetics_differences,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

RECORD = (
    Path(__file__).resolve().parents[3]
    / "docs/development/lifting/evidence/back_squat/kinetics_parity.json"
)
ATOL = 1e-6  # N or N m
RTOL = 1e-8
COP_ATOL_M = 1e-8
FRAMES = (0, 100, 200)  # window start, squat bottom, window end
ENGINES = ("drake", "pinocchio", "opensim")
_MODULE = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
}


def _real_package(engine: str) -> bool:
    try:
        module = importlib.import_module(_MODULE[engine])
    except (ImportError, OSError):
        return False
    if type(module).__module__ == "unittest.mock":
        return False
    if not getattr(module, "__file__", None):
        return False
    if engine == "pinocchio":
        return all(hasattr(module, n) for n in ("rnea", "buildModelFromXML", "Model"))
    if engine == "drake":
        import pydrake.all as drake_all

        return type(drake_all).__module__ != "unittest.mock" and hasattr(
            drake_all, "MultibodyPlant"
        )
    return True


def _require_real_engine(engine: str) -> None:
    if not _real_package(engine):
        pytest.skip(
            f"{_MODULE[engine]} is not installed (or is a test double)",
        )


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.fixture(scope="module")
def mujoco_samples(bundle) -> list[KineticsSample]:
    pytest.importorskip("mujoco")
    return bundle_kinetics("mujoco", bundle, FRAMES)


def test_committed_record_is_inside_the_tolerances(bundle) -> None:
    """The full-window record (every 20th state, all engines) meets the bounds."""
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["bundle_spec_sha256"] == bundle.spec_sha256
    assert record["indices"] == list(range(0, bundle.steps + 1, 20))
    assert len(record["source_commit"]) == 40
    values = record["mujoco_values"]
    grf_scale = max(np.abs(v["grf_net_n"]).max() for v in values)
    grip_scale = max(np.abs(v["grip_net_force_n"]).max() for v in values)
    engines = record["max_difference_from_mujoco"]
    assert set(engines) == {"drake", "pinocchio", "opensim"}
    for engine in ("drake", "pinocchio"):
        assert not engines[engine].get("unavailable"), engine
    for engine, diff in engines.items():
        if diff.get("unavailable"):
            assert diff["reason"], engine
            continue
        for key in ("left", "right", "net"):
            assert diff[f"grf_force_{key}_n"] <= ATOL + RTOL * grf_scale, engine
            assert diff[f"free_moment_{key}_nm"] <= ATOL + RTOL * grf_scale, engine
            assert diff[f"cop_{key}_m"] <= COP_ATOL_M, engine
        if engine in GRIP_ENGINES:
            for key in ("hand_left_force_n", "hand_right_force_n", "grip_net_force_n"):
                assert diff[key] <= ATOL + RTOL * grip_scale, (engine, key)
        else:
            assert diff["grip_net_force_n"] is None
            assert diff["grip_unavailable_reason"], engine


def test_grip_engines_are_the_kkt_engines() -> None:
    assert set(GRIP_ENGINES) == {"mujoco", "drake"}


def test_rejects_an_unknown_engine(bundle) -> None:
    with pytest.raises(ValueError, match="engine"):
        bundle_kinetics("bullet", bundle, FRAMES)


def test_rejects_indices_outside_the_bundle(bundle) -> None:
    with pytest.raises(ValueError, match="indices"):
        bundle_kinetics("mujoco", bundle, (0, bundle.steps + 1))
    with pytest.raises(ValueError, match="indices"):
        bundle_kinetics("mujoco", bundle, ())


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_mujoco_feet_carry_the_lifter(mujoco_samples, bundle) -> None:
    for sample in mujoco_samples:
        assert sample.time_s == pytest.approx(sample.index * bundle.dt_s)
        for foot in ("left", "right"):
            reaction = sample.per_foot[foot]
            assert reaction.in_contact
            assert reaction.cop_m is not None
        assert sample.net.force_n[2] > 0.0
        assert sample.grip.split_method == "constraint_multiplier"


@pytest.mark.integration
@pytest.mark.parametrize("engine", ENGINES)
def test_ground_reaction_matches_mujoco(mujoco_samples, bundle, engine) -> None:
    _require_real_engine(engine)
    for ref, other in zip(
        mujoco_samples, bundle_kinetics(engine, bundle, FRAMES), strict=True
    ):
        diff = kinetics_differences(ref, other)
        scale = float(np.abs(ref.net.force_n).max())
        bound = ATOL + RTOL * scale
        for key in ("left", "right", "net"):
            assert diff[f"grf_force_{key}_n"] <= bound, (engine, ref.index, diff)
            assert diff[f"free_moment_{key}_nm"] <= bound, (engine, ref.index, diff)
            assert diff[f"cop_{key}_m"] <= COP_ATOL_M, (engine, ref.index, diff)


@pytest.mark.integration
@pytest.mark.requires_drake
def test_drake_hand_wrench_matches_mujoco(mujoco_samples, bundle) -> None:
    _require_real_engine("drake")
    for ref, other in zip(
        mujoco_samples, bundle_kinetics("drake", bundle, FRAMES), strict=True
    ):
        diff = kinetics_differences(ref, other)
        scale = float(np.abs(np.asarray(ref.grip.net_force_n)).max())
        bound = ATOL + RTOL * scale
        for key in ("hand_left_force_n", "hand_right_force_n", "grip_net_force_n"):
            assert diff[key] <= bound, (ref.index, key, diff[key])
        assert diff["grip_couple_nm"] <= bound, (ref.index, diff["grip_couple_nm"])


@pytest.mark.integration
@pytest.mark.requires_pinocchio
def test_pinocchio_hand_wrench_is_unavailable_with_a_reason(bundle) -> None:
    _require_real_engine("pinocchio")
    (sample,) = bundle_kinetics("pinocchio", bundle, (0,))
    assert sample.grip.split_method == "unavailable"
    assert sample.grip.net_force_n is None
    assert "pinocchio" in sample.grip.unavailable_reason
    diff = kinetics_differences(sample, sample)
    assert diff["grip_net_force_n"] is None
