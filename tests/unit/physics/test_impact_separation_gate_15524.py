"""Only separated impact results may release a state to playback or flight."""

from types import SimpleNamespace

import pytest

from src.shared.python.physics.impact_interval_audit import (
    ImpactAuditError,
    audit_and_export_impact_run,
    audit_impact_interval,
    qualified_post_impact_state,
)
from tests.unit.physics.test_impact_interval_audit_dbc import (
    _POST_IMPACT,
    _TrackedResult,
    _build_audit,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("enum_like", [False, True])
@pytest.mark.parametrize(
    ("termination", "status"),
    [
        ("TIME_LIMIT", "time_limit"),
        ("NO_CONTACT", "no_contact"),
        ("NEW_STOP", "incomplete"),
    ],
)
def test_completed_flag_cannot_override_nonseparated_termination(
    termination: str, status: str, enum_like: bool
) -> None:
    result = _TrackedResult(_build_audit(), contact_completed=True)
    result.termination = SimpleNamespace(name=termination) if enum_like else termination

    verdict = audit_impact_interval(result)
    assert not verdict.qualified
    assert verdict.contact_completed is True  # Preserve the provider diagnostic.
    assert verdict.termination == termination
    assert any("termination" in failure for failure in verdict.failures)
    with pytest.raises(ImpactAuditError):
        qualified_post_impact_state(result)
    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is False
    assert exported["completion_status"] == status
    assert exported["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("enum_like", [False, True])
def test_separated_termination_also_requires_completed_contact(
    completed: bool, enum_like: bool
) -> None:
    result = _TrackedResult(_build_audit(), contact_completed=completed)
    result.termination = SimpleNamespace(name="SEPARATED") if enum_like else "SEPARATED"

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is completed
    assert exported["completion_status"] == ("separated" if completed else "incomplete")
    assert exported["post_impact_state"] == (_POST_IMPACT if completed else None)
    assert result.to_post_impact_state_calls == int(completed)
