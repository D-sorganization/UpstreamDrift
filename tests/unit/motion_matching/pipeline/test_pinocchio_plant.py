"""Unit tests for Pinocchio MatchingPlant full lane (MS-14 #10333).

Native Pinocchio/Pink probes skip when the SDK is absent. Contract and
fail-closed receipt tests always run so Windows CI stays honest.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.plant import (
    EngineUnavailableError,
    MatchingPlant,
    get_plant,
)
from src.shared.python.motion_matching.pipeline.receipt_components import (
    ConstrainedIkReceipt,
)
from src.shared.python.motion_matching.pipeline.plants.pinocchio_lane_receipts import (
    CLOSURE_RESIDUAL_BUDGET_M,
    build_blocked_constrained_ik_receipt,
    build_blocked_pinocchio_lane_receipt,
    build_blocked_runtime_receipt,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_SPEC = _REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
_PLANT_SRC = (
    _REPO / "src/shared/python/motion_matching/pipeline/plants/pinocchio_plant.py"
)


def _require_real_pinocchio() -> None:
    try:
        import pinocchio as pin
    except ImportError as exc:
        pytest.skip(f"pinocchio not importable: {exc}")
    if type(pin).__module__ == "unittest.mock" or not hasattr(
        pin, "constraintDynamics"
    ):
        pytest.skip("pinocchio is mocked, not a real installation")


def _load_spec() -> dict[str, Any]:
    return json.loads(_SPEC.read_text(encoding="utf-8"))


def test_pinocchio_plant_module_has_no_top_level_sdk_import() -> None:
    """LoD: pinocchio/pink must stay inside lazy adapter paths."""
    tree = ast.parse(_PLANT_SRC.read_text(encoding="utf-8"))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    assert "pinocchio" not in top_imports
    assert "pink" not in top_imports


def test_get_plant_pinocchio_fails_closed_without_sdk() -> None:
    try:
        import pinocchio as pin
    except ImportError:
        pin = None
    real = (
        pin is not None
        and type(pin).__module__ != "unittest.mock"
        and hasattr(pin, "constraintDynamics")
    )
    if real:
        pytest.skip("pinocchio is installed; fail-closed path not exercised")
    with pytest.raises(EngineUnavailableError, match="pinocchio|SDK|Failed"):
        get_plant("pinocchio", _load_spec())


def test_blocked_lane_receipt_is_honest_and_not_accepted() -> None:
    receipt = build_blocked_pinocchio_lane_receipt(
        reason="pinocchio SDK unavailable on host",
        host="unit-test",
        repository_revision="deadbeef",
    )
    assert receipt["status"] == "blocked"
    assert receipt["accepted"] is False
    assert receipt["engine"] == "pinocchio"
    assert receipt["native_claims"] is False
    assert receipt["gates"] == []
    assert "G1" not in json.dumps(receipt)
    assert receipt["closure_residual_m"] is None


def test_blocked_pink_receipt_validates_schema_without_green_closure() -> None:
    payload = build_blocked_constrained_ik_receipt(
        reason="pink/pinocchio unavailable",
        model_name="anthro_driver",
        capture_name="driver",
    )
    receipt = ConstrainedIkReceipt.model_validate(payload)
    assert receipt.is_qualified is False
    assert receipt.all_frames_converged is False
    assert receipt.closure_residual_m is None
    assert receipt.qualification_state == "blocked_unavailable"


def test_constrained_ik_receipt_rejects_qualified_with_open_closure() -> None:
    with pytest.raises(ValueError, match="closure_residual_m"):
        ConstrainedIkReceipt.model_validate(
            {
                "backend_name": "pink",
                "solver": "quadprog",
                "model_name": "anthro_driver",
                "capture_name": "driver",
                "step_mode": "physical",
                "limit_policy": "enforce",
                "task_policy": "dual_grip_hard_equality",
                "time_semantics": "strict_physical_elapsed_dt",
                "frame_count": 30,
                "frame_success_count": 30,
                "all_frames_converged": True,
                "closure_residual_m": CLOSURE_RESIDUAL_BUDGET_M + 1e-3,
                "is_qualified": True,
                "qualification_state": "qualified",
            }
        )


def test_blocked_runtime_receipt_records_missing_components() -> None:
    receipt = build_blocked_runtime_receipt(
        missing=("pinocchio", "pink"),
        repository_revision="deadbeef",
    )
    assert receipt["status"] == "blocked"
    assert receipt["accepted"] is False
    assert set(receipt["missing_components"]) == {"pinocchio", "pink"}


def test_create_ik_rejects_unknown_backend_contract() -> None:
    from src.shared.python.motion_matching.pipeline.plants.pinocchio_plant import (
        PinocchioMatchingPlant,
    )

    plant = object.__new__(PinocchioMatchingPlant)
    plant.spec_dict = {"coordinate_order": ["a"]}
    with pytest.raises(ValueError, match="Unknown Pinocchio IK backend"):
        PinocchioMatchingPlant.create_ik(plant, {}, ik_backend="not-a-backend")


def test_pinocchio_plant_satisfies_matching_plant_protocol() -> None:
    _require_real_pinocchio()
    plant = get_plant("pinocchio", _load_spec())
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "pinocchio"
    assert len(plant.coordinate_order) in (41, 44)
    assert plant.ground_plane is not None
    assert len(plant.plant_sha) == 64


def test_pinocchio_plant_create_constrained_ik_pink() -> None:
    _require_real_pinocchio()
    pytest.importorskip("pink")
    plant = get_plant("pinocchio", _load_spec())
    backend = plant.create_constrained_ik()
    assert backend.backend_name.startswith("pink")


def test_pinocchio_plant_accelerations_and_derivatives() -> None:
    _require_real_pinocchio()
    plant = get_plant("pinocchio", _load_spec())
    coords = plant.coordinate_order
    q = dict.fromkeys(coords, 0.0)
    v = dict.fromkeys(coords, 0.0)
    tau = dict.fromkeys(coords, 0.0)
    acc = plant.accelerations(q, v, tau)
    assert len(acc) == len(coords)
    assert all(np.isfinite(val) for val in acc.values())
    deriv = plant.acceleration_derivatives(q, v, tau)
    assert deriv is not None
    contact = plant.contact_effort_derivatives(q, v)
    assert contact is not None


def test_pinocchio_plant_matches_mujoco_accelerations() -> None:
    """Shared contact law + weld: Pinocchio qdd within 1e-6 of MuJoCo at 3 states."""
    _require_real_pinocchio()
    pytest.importorskip("mujoco")
    spec = _load_spec()
    pin_plant = get_plant("pinocchio", spec)
    mj_plant = get_plant("mujoco", spec)
    shared = [c for c in pin_plant.coordinate_order if c in mj_plant.coordinate_order]
    assert len(shared) >= 30
    rng = np.random.default_rng(10333)
    for _ in range(3):
        q = {c: float(rng.normal(scale=0.02)) for c in shared}
        v = {c: float(rng.normal(scale=0.01)) for c in shared}
        tau = dict.fromkeys(shared, 0.0)
        # Pad missing coords with zeros for each plant.
        q_pin = {c: q.get(c, 0.0) for c in pin_plant.coordinate_order}
        v_pin = {c: v.get(c, 0.0) for c in pin_plant.coordinate_order}
        tau_pin = dict.fromkeys(pin_plant.coordinate_order, 0.0)
        q_mj = {c: q.get(c, 0.0) for c in mj_plant.coordinate_order}
        v_mj = {c: v.get(c, 0.0) for c in mj_plant.coordinate_order}
        tau_mj = dict.fromkeys(mj_plant.coordinate_order, 0.0)
        a_pin = pin_plant.accelerations(q_pin, v_pin, tau_pin)
        a_mj = mj_plant.accelerations(q_mj, v_mj, tau_mj)
        for name in shared:
            assert a_pin[name] == pytest.approx(a_mj[name], abs=1e-6), name


def test_pinocchio_matching_plant_alias_module() -> None:
    from src.engines.physics_engines.pinocchio.python.matching_plant import (
        PinocchioMatchingPlant as PinAlias,
    )
    from src.shared.python.motion_matching.pipeline.plants.pinocchio_plant import (
        PinocchioMatchingPlant,
    )

    assert PinAlias is PinocchioMatchingPlant


def test_resolve_fit_native_plant_reuses_matching_plant() -> None:
    """Fitter must consume registry MatchingPlant model (LoD: one plant)."""
    from types import SimpleNamespace

    from src.engines.physics_engines.pinocchio.python.full_body_fit import (
        resolve_fit_native_plant,
    )

    model = object()
    matching = SimpleNamespace(engine_name="pinocchio", model=model)
    assert resolve_fit_native_plant({"schema_version": 1}, matching) is model
    with pytest.raises(ValueError, match="pinocchio MatchingPlant"):
        resolve_fit_native_plant(
            {"schema_version": 1},
            SimpleNamespace(engine_name="mujoco", model=model),
        )


def test_resolve_fit_native_plant_builds_when_unbound() -> None:
    _require_real_pinocchio()
    from src.engines.physics_engines.pinocchio.python.full_body_fit import (
        resolve_fit_native_plant,
    )

    built = resolve_fit_native_plant(_load_spec(), matching_plant=None)
    assert built is not None
    assert hasattr(built, "accelerations")
