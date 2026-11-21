"""SHOW-C3c slice 3b-2 (#14559): the real-run entry point's pure parts.

The ladder itself is NOT executed here (held until A6, supervisor D24); these
tests pin the joint subset, the held-joint list and the output-location rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.full_body_ladder_run import (
    CONTROLLED_JOINTS,
    baseline_band,
    ladder_spec,
    require_outside_repo,
    select_controlled,
)
from src.shared.python.motion_matching.full_body_v2_showpiece import SHOWPIECE_V2

pytestmark = pytest.mark.unit

ORDER = tuple(json.loads(SHOWPIECE_V2.read_text(encoding="utf-8"))["coordinate_order"])


def test_subset_has_trunk_neck_and_the_lead_arm_chain() -> None:
    chosen = select_controlled(ORDER)
    for needed in (
        "lumbar_axial_rotation",
        "thoracic_axial_rotation",
        "NeckInputZ",
        "LSInputX",
        "LEInput",
        "LFInput",
        "LWInputX",
    ):
        assert needed in chosen
    assert chosen == tuple(j for j in ORDER if j in CONTROLLED_JOINTS)  # plant order


def test_subset_never_contains_root_or_legs() -> None:
    chosen = select_controlled(ORDER)
    assert not any(j.startswith(("Translation", "HipInput")) for j in chosen)
    assert not any(j.endswith(("_r", "_l")) for j in chosen)


def test_a_missing_joint_is_refused() -> None:
    with pytest.raises(ContractViolationError, match="NeckInputZ"):
        select_controlled(tuple(j for j in ORDER if j != "NeckInputZ"))


def test_default_ladder_does_not_stop_below_the_firm_125_target() -> None:
    spec = ladder_spec()
    assert spec.stop_mph == 125.0 and spec.start_mph < spec.stop_mph


def test_outputs_must_live_outside_the_repo(tmp_path: Path) -> None:
    require_outside_repo(tmp_path)
    with pytest.raises(ContractViolationError, match="outside the repo"):
        require_outside_repo(Path(__file__).resolve().parents[3] / "docs" / "x")


def test_run_gate_envelope_builds_on_v2_without_unbounded_rom() -> None:
    """Precondition for the real ladder run: no ``allow_unbounded_rom`` (D29).

    Trunk (D35) and shoulder (D39) ROM are declared, so the gate is open.
    """
    from src.shared.python.impact_optimization.physiological_envelope import (
        DEFAULT_LIMITS_PATH,
        PhysiologicalEnvelope,
        load_limits,
    )
    from src.shared.python.loop_nullspace.capacity import CapacitySubject

    spec = json.loads(SHOWPIECE_V2.read_text(encoding="utf-8"))
    env = PhysiologicalEnvelope.from_spec(
        spec,
        CapacitySubject(mass_kg=78.0, stature_m=1.71),
        load_limits(DEFAULT_LIMITS_PATH),
    )
    assert set(CONTROLLED_JOINTS) <= set(env.names)


def test_face_is_checked_unless_the_face_free_flag_is_set() -> None:
    assert baseline_band(face_free=False).max_abs_face_deg == 25.0
    free = baseline_band(face_free=True)
    assert free.max_abs_face_deg is None
    assert (free.speed_mph_min, free.speed_mph_max) == (100.0, 120.0)
    assert free.max_abs_path_deg == 15.0
