"""Integration tests for NSX-56 Two-Track Cross-Engine Conformance (#12584).

Verifies:
- Divergence registry entries for NSX closed-loop snapshot Track N and MuJoCo
  soft-constraint vs hard-KKT enforcement.
- Manifold projection via ``project_to_closure`` prior to snapshot evaluation.
- Track N (``1e-6`` scaled relative tolerance on canonical ``M, h, J_c, Jdot_v``)
  and Track S (mathematically propagated ``cond(A) * max(delta_Track_N, 1e-12)``
  bound on null-space projector, RHS, and static force envelopes) across
  Pinocchio, MuJoCo hard-KKT, MyoSuite, Drake, OpenSim, and Simscape R2025b.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.shared.python.loop_nullspace.conformance import (
    evaluate_two_track_conformance,
)
from src.shared.python.loop_nullspace.engine_adapters import (
    DrakeClosedLoopSnapshotAdapter,
    MuJoCoClosedLoopSnapshotAdapter,
    MyoSuiteClosedLoopSnapshotAdapter,
    OpenSimClosedLoopSnapshotAdapter,
    PinocchioClosedLoopSnapshotAdapter,
)
from src.shared.python.loop_nullspace.simscape_snapshot_loader import (
    SimscapeClosedLoopSnapshotAdapter,
)
from tests.unit.loop_nullspace.test_engine_adapters_and_conformance import (
    FIXTURE_PATH,
    _build_all_adapters_on_reference,
    _ReferenceClosedLoopBackend,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = Path(__file__).resolve().parent / "divergence_registry.yaml"
SPEC_PATH = (
    REPO_ROOT
    / "docs/development/full_body_models/evidence/ground_support/full_body_spec_hipcal_scaled.json"
)
RECORD_PATH = (
    REPO_ROOT
    / "docs/development/full_body_models/evidence/ground_support/dynamics_record.npz"
)


def test_nsx56_divergence_registry_entries_present_and_valid() -> None:
    """Verify NSX-56 entries exist in divergence_registry.yaml with non-empty rationale."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in registry["divergences"]}

    assert "nsx-closed-loop-snapshot-track-n" in by_id
    track_n_entry = by_id["nsx-closed-loop-snapshot-track-n"]
    assert track_n_entry["tolerance"] == pytest.approx(1e-6)
    assert set(track_n_entry["engines"]) == {
        "pinocchio",
        "mujoco",
        "myosuite",
        "drake",
        "opensim",
        "simscape",
    }
    assert len(track_n_entry["rationale"].strip()) > 20

    assert "nsx-mujoco-soft-equality-vs-hard-kkt" in by_id
    mj_entry = by_id["nsx-mujoco-soft-equality-vs-hard-kkt"]
    assert "kkt_regularization == 0" in mj_entry["rationale"]


def test_nsx56_two_track_conformance_matches_registry_tolerance() -> None:
    """Verify two-track conformance across all 6 engine adapters within registry tolerance."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in registry["divergences"]}
    track_n_tol = float(by_id["nsx-closed-loop-snapshot-track-n"]["tolerance"])

    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    adapters["simscape"] = SimscapeClosedLoopSnapshotAdapter(FIXTURE_PATH)

    q_off_manifold = ref.q_ref + 5e-5 * np.linspace(-1.0, 1.0, ref.nv)
    v_off_manifold = ref.v_ref + 5e-5 * np.linspace(1.0, -1.0, ref.nv)

    report = evaluate_two_track_conformance(
        reference_adapter=adapters["pinocchio"],
        candidate_adapters={k: v for k, v in adapters.items() if k != "pinocchio"},
        q=q_off_manifold,
        v=v_off_manifold,
        a=ref.a_ref,
        track_n_tolerance=track_n_tol,
    )

    assert report.reference_engine == "pinocchio"
    assert report.projection_pose_residual_after <= 1e-12
    assert report.projection_rate_residual_after <= 1e-12
    assert report.all_passed

    for name, res in report.results.items():
        assert res.track_n.passed, f"Track N failed for {name}: {res.track_n}"
        assert res.track_s.passed, f"Track S failed for {name}: {res.track_s}"
        assert res.track_s.propagated_relative_bound == pytest.approx(
            res.track_s.condition_number * max(res.track_n.max_delta, 1e-12),
            rel=1e-12,
        )


@pytest.mark.requires_mujoco
def test_nsx56_live_mujoco_hard_kkt_two_track_conformance() -> None:
    """Run live 41-DOF MuJoCo hard-KKT two-track conformance (analytic vs FD drift)."""
    pytest.importorskip("mujoco")
    spec_bytes = SPEC_PATH.read_bytes()
    with np.load(RECORD_PATH) as data:
        q0, v0 = data["q"][0], data["v"][0]
        a0 = data["a"][0] if "a" in data else np.zeros_like(v0)

    ref_mj = MuJoCoClosedLoopSnapshotAdapter(
        spec_bytes, kkt_regularization=0.0, drift_method="analytic"
    )
    cand_mj_fd = MuJoCoClosedLoopSnapshotAdapter(
        spec_bytes,
        kkt_regularization=0.0,
        drift_method="finite_difference_centered",
    )

    report = evaluate_two_track_conformance(
        reference_adapter=ref_mj,
        candidate_adapters={"mujoco_fd": cand_mj_fd},
        q=q0,
        v=v0,
        a=a0,
        track_n_tolerance=1e-6,
    )
    assert report.projection_pose_residual_after <= 1e-12
    assert report.all_passed


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    ("engine_name", "adapter_cls", "module_name"),
    [
        ("mujoco", MuJoCoClosedLoopSnapshotAdapter, "mujoco"),
        ("myosuite", MyoSuiteClosedLoopSnapshotAdapter, "myosuite"),
        ("drake", DrakeClosedLoopSnapshotAdapter, "pydrake"),
        ("opensim", OpenSimClosedLoopSnapshotAdapter, "opensim"),
    ],
)
def test_nsx56_live_cross_engine_conformance_against_pinocchio(
    engine_name: str, adapter_cls: type, module_name: str
) -> None:
    """Live runtime Track N & Track S conformance against Pinocchio when bindings exist."""
    pin_mod = pytest.importorskip("pinocchio")
    target_mod = pytest.importorskip(module_name)
    if type(pin_mod).__module__ == "unittest.mock":
        pytest.skip("pinocchio is a MagicMock stub on this host")
    if type(target_mod).__module__ == "unittest.mock":
        pytest.skip(f"{module_name} is a MagicMock stub on this host")
    spec_bytes = SPEC_PATH.read_bytes()
    with np.load(RECORD_PATH) as data:
        q0, v0 = data["q"][0], data["v"][0]
        a0 = data["a"][0] if "a" in data else np.zeros_like(v0)

    pin_adapter = PinocchioClosedLoopSnapshotAdapter(spec_bytes)
    cand_adapter = adapter_cls(spec_bytes)
    report = evaluate_two_track_conformance(
        reference_adapter=pin_adapter,
        candidate_adapters={engine_name: cand_adapter},
        q=q0,
        v=v0,
        a=a0,
        track_n_tolerance=1e-6,
    )
    assert report.all_passed
