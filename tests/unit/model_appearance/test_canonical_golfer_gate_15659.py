"""Canonical golfer gate: refuse a non-current model before any simulation (#15659).

Panel Option A (#15648 follow-up): optimization, allocation and replay may only
run on the registry's current golfer model, and a Solution Bundle carries a
model lock split three ways -- ``plant_hash`` (dynamics), ``controller_hash``
(gains and spline basis) and ``solver_hash`` (integrator and step) -- so a
mismatch names the domain that moved.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import canonical_golfer_gate as gate
from src.shared.python.model_appearance.canonical_golfer_gate import (
    GolferModelLock,
    NonCanonicalGolferError,
    bundle_lock_status,
    controller_hash,
    lock_for_spec,
    plant_hash,
    require_canonical_golfer,
    require_locked_bundle,
    solver_hash,
    verify_lock,
)
from src.shared.python.model_appearance.golfer_model_versions import (
    current_golfer_model,
)
from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    SolutionBundleError,
    load_solution_bundle,
    save_solution_bundle,
)

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (REPO / "tests/fixtures/club_face/spec_driver.json").read_text(encoding="utf-8")
)
KP, KD = np.full(3, 200.0), np.full(3, 20.0)
KNOTS = np.array([0.0, 0.0, 0.0, 0.125, 0.25, 0.25, 0.25])
CONTROL = {"hold": "zoh"}
INTEGRATOR = {"name": "rk4", "dt": 0.05}


def _canonical_spec() -> dict[str, Any]:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    spec["spec_revision"] = {"version": 1, "configuration": "skinned_hand_wrists"}
    return spec


def _lock(spec: dict[str, Any] | None = None) -> GolferModelLock:
    return lock_for_spec(
        spec or _canonical_spec(),
        controller=controller_hash(KP, KD, KNOTS, CONTROL),
        solver=solver_hash(INTEGRATOR),
    )


# -- require_canonical_golfer -------------------------------------------------


def test_current_model_spec_passes_and_returns_the_registry_pin() -> None:
    assert require_canonical_golfer(_canonical_spec()) == current_golfer_model()


@pytest.mark.parametrize(
    ("field", "value", "part"),
    [
        ("hand_visual_backend", "capsule", "capsule"),
        ("body_visual_backend", "loft", "loft"),
    ],
)
def test_deprecated_part_is_refused_and_named(field: str, value: str, part: str) -> None:
    spec = _canonical_spec()
    spec["subject"][field] = value
    with pytest.raises(NonCanonicalGolferError, match=part) as info:
        require_canonical_golfer(spec)
    assert info.value.part == part


def test_missing_selectors_default_to_deprecated_parts() -> None:
    spec = _canonical_spec()
    del spec["subject"]["hand_visual_backend"]
    with pytest.raises(NonCanonicalGolferError) as info:
        require_canonical_golfer(spec)
    assert info.value.part == "capsule"


@pytest.mark.parametrize("revision", [None, {"version": 1, "configuration": "other"}])
def test_unrevised_spec_is_refused(revision: dict[str, Any] | None) -> None:
    spec = _canonical_spec()
    if revision is None:
        del spec["spec_revision"]
    else:
        spec["spec_revision"] = revision
    with pytest.raises(NonCanonicalGolferError) as info:
        require_canonical_golfer(spec)
    assert info.value.part == "spec_revision"


def test_error_is_a_value_error_for_existing_handlers() -> None:
    assert issubclass(NonCanonicalGolferError, ValueError)


def test_gate_reads_the_pin_from_the_registry_not_a_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = replace(current_golfer_model(), spec_revision_configuration="next_rev")
    monkeypatch.setattr(gate, "current_golfer_model", lambda: pinned)
    with pytest.raises(NonCanonicalGolferError, match="next_rev"):
        require_canonical_golfer(_canonical_spec())


# -- three-way hash split -----------------------------------------------------


def test_plant_hash_ignores_visual_backends_and_revision_metadata() -> None:
    a, b = _canonical_spec(), _canonical_spec()
    b["subject"]["hand_visual_backend"] = "capsule"
    b["subject"]["body_visual_backend"] = "loft"
    b["spec_revision"] = {"version": 9, "configuration": "skinned_hand_wrists"}
    b["visual_hints"] = {"anything": 1}
    assert plant_hash(a) == plant_hash(b)


def test_plant_hash_moves_with_the_dynamics() -> None:
    a, b = _canonical_spec(), _canonical_spec()
    b["subject"]["mass_kg"] += 1.0
    assert plant_hash(a) != plant_hash(b)


def test_controller_and_solver_hashes_are_independent_domains() -> None:
    base = controller_hash(KP, KD, KNOTS, CONTROL)
    assert controller_hash(KP * 2, KD, KNOTS, CONTROL) != base
    assert controller_hash(KP, KD, KNOTS[:-1], CONTROL) != base
    assert controller_hash(KP, KD, KNOTS, CONTROL) == base
    assert solver_hash({**INTEGRATOR, "dt": 0.01}) != solver_hash(INTEGRATOR)
    assert solver_hash({"dt": 0.05, "name": "rk4"}) == solver_hash(INTEGRATOR)


@pytest.mark.parametrize("integrator", [{"name": "rk4"}, {"dt": 0.05}, {}])
def test_solver_hash_requires_integrator_and_timestep(integrator: dict) -> None:
    with pytest.raises(ValueError, match="integrator"):
        solver_hash(integrator)


# -- lock verification --------------------------------------------------------


def test_lock_for_a_non_canonical_spec_is_refused() -> None:
    spec = _canonical_spec()
    spec["subject"]["body_visual_backend"] = "loft"
    with pytest.raises(NonCanonicalGolferError):
        _lock(spec)


def test_lock_roundtrips_through_a_plain_mapping() -> None:
    lock = _lock()
    assert GolferModelLock.from_dict(lock.to_dict()) == lock
    assert lock.registry_version == current_golfer_model().version


@pytest.mark.parametrize(
    ("domain", "scope"),
    [
        ("plant_hash", "optimization"),
        ("controller_hash", "optimization"),
        ("solver_hash", "optimization"),
        ("plant_hash", "allocation"),
        ("registry_version", "allocation"),
    ],
)
def test_mismatch_names_the_domain(domain: str, scope: str) -> None:
    expected = _lock()
    recorded = replace(expected, **{domain: "x" * 64})
    with pytest.raises(NonCanonicalGolferError) as info:
        verify_lock(recorded, expected, scope=scope)
    assert info.value.part == domain


def test_allocation_pins_the_plant_only() -> None:
    expected = _lock()
    recorded = replace(expected, controller_hash="c" * 64, solver_hash="s" * 64)
    verify_lock(recorded, expected, scope="allocation")
    with pytest.raises(NonCanonicalGolferError):
        verify_lock(recorded, expected, scope="replay")


def test_unknown_scope_is_a_contract_violation() -> None:
    with pytest.raises(ValueError, match="scope"):
        verify_lock(_lock(), _lock(), scope="anything")


# -- Solution Bundle mandatory lock -------------------------------------------


def _bundle(lock: GolferModelLock | None) -> SolutionBundle:
    rng = np.random.default_rng(15659)
    n, j = 4, 3
    arrays = {
        name: rng.normal(size=(n, j))
        for name in (
            "q_ref",
            "qd_ref",
            "tau_ff_ref",
            "q_rollout",
            "qd_rollout",
            "tau_cmd",
            "tau_applied",
        )
    }
    arrays.update(
        time=np.linspace(0.0, 0.15, n),
        knots=KNOTS,
        delta_tau_coeffs=rng.normal(size=(4, j)),
        q0=np.zeros(j),
        qd0=np.zeros(j),
        controller_state0=np.zeros(0),
        kp=KP,
        kd=KD,
    )
    return SolutionBundle(
        solution_id="sol-15659",
        parent_id=None,
        git_sha="a" * 40,
        model_hash="m" * 64,
        asset_hashes={},
        frame="world_z_up",
        joint_order=("a", "b", "c"),
        control=CONTROL,
        integrator=INTEGRATOR,
        environment={},
        optimizer_seed=0,
        impact_event={"t": 0.1, "q": [0.0] * j, "qd": [0.0] * j},
        flight_metrics={},
        feasibility_margins={},
        assumptions=(),
        arrays=arrays,
        model_lock=None if lock is None else lock.to_dict(),
    )


def test_locked_bundle_roundtrips_and_seeds_optimization(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(_lock()), tmp_path / "b")
    loaded = load_solution_bundle(tmp_path / "b")
    assert bundle_lock_status(loaded) == "locked"
    assert require_locked_bundle(loaded, _lock(), scope="optimization") == _lock()


def test_bundle_lock_must_match_its_own_controller_and_solver() -> None:
    lock = replace(_lock(), solver_hash=solver_hash({"name": "euler", "dt": 0.05}))
    with pytest.raises(NonCanonicalGolferError) as info:
        require_locked_bundle(_bundle(lock), lock, scope="optimization")
    assert info.value.part == "solver_hash"


def test_legacy_bundle_is_replay_only_and_cannot_seed(tmp_path: Path) -> None:
    save_solution_bundle(_bundle(None), tmp_path / "legacy")
    legacy = load_solution_bundle(tmp_path / "legacy")
    assert bundle_lock_status(legacy) == "legacy_replay_only"
    for scope in ("optimization", "allocation", "replay"):
        with pytest.raises(NonCanonicalGolferError, match="legacy_replay_only"):
            require_locked_bundle(legacy, _lock(), scope=scope)


def test_bundle_with_a_malformed_lock_is_refused(tmp_path: Path) -> None:
    bad = replace(_bundle(_lock()), model_lock={"plant_hash": "p" * 64})
    with pytest.raises(SolutionBundleError, match="model_lock"):
        save_solution_bundle(bad, tmp_path / "bad")
