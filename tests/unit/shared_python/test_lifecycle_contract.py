"""Unit tests for the canonical lifecycle contract and state transitions."""

from __future__ import annotations

from typing import Any

import pytest

import src.shared.python.lifecycle as lc

_LS = lc.LifecycleStatus

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.unit
class TestLifecycleContract:
    def test_constants_and_terminal(self) -> None:
        words = {"QUEUED", "RUNNING", "CANCEL_REQUESTED"} | {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
        }
        assert words == lc.CANONICAL_LIFECYCLE_WORDS and {s.name for s in _LS} == words
        expected_terminals = {
            _LS.SUCCEEDED,
            _LS.FAILED,
            _LS.CANCELLED,
        }
        assert expected_terminals == lc.TERMINAL_LIFECYCLE_STATES
        for t in lc.TERMINAL_LIFECYCLE_STATES:
            assert len(lc.CANONICAL_LIFECYCLE_TRANSITIONS[t]) == 0
        t_cases = (_LS.SUCCEEDED, _LS.FAILED, _LS.CANCELLED, "SUCCEEDED", "failed")
        assert all(lc.is_terminal(x) for x in t_cases)
        nt_cases = (_LS.QUEUED, _LS.RUNNING, _LS.CANCEL_REQUESTED, "queued")
        assert not any(lc.is_terminal(x) for x in nt_cases)
        for inv in ("UNKNOWN", 123, None):
            with pytest.raises(ValueError, match="Unknown lifecycle state"):
                lc.is_terminal(inv)

    def test_transition_table_shapes(self) -> None:
        assert lc.transition({"q": frozenset({"r"})}, "q", "r") == "r"
        assert lc.transition({("q", "start"): "r"}, "q", "start") == "r"
        assert lc.transition({"q": {"start": "r"}}, "q", "start") == "r"
        with pytest.raises(lc.IllegalTransitionError):
            lc.transition({"q": frozenset({"r"})}, "q", "s")
        with pytest.raises(lc.IllegalTransitionError):
            lc.transition({("q", "start"): "r"}, "q", "done")
        with pytest.raises(lc.IllegalTransitionError):
            lc.transition({}, "q", "r")

    def test_oracle2_illegal_transitions(self) -> None:
        tr = lc.CANONICAL_LIFECYCLE_TRANSITIONS
        disallowed = (
            (_LS.SUCCEEDED, _LS.RUNNING),
            (_LS.CANCELLED, _LS.SUCCEEDED),
            (_LS.QUEUED, _LS.SUCCEEDED),
            (_LS.FAILED, _LS.RUNNING),
        )
        for src, dst in disallowed:
            with pytest.raises(lc.IllegalTransitionError):
                lc.transition(tr, src, dst)
        for term in (_LS.SUCCEEDED, _LS.FAILED, _LS.CANCELLED):
            for target in list(_LS):
                with pytest.raises(lc.IllegalTransitionError):
                    lc.transition(tr, term, target)

    def test_oracle3_cancel_races(self) -> None:
        tr = lc.CANONICAL_LIFECYCLE_TRANSITIONS
        req = lc.transition(tr, _LS.RUNNING, _LS.CANCEL_REQUESTED)
        s1 = lc.LifecycleSnapshot(
            status=lc.transition(tr, req, _LS.SUCCEEDED), outcome="fast"
        )
        assert s1.outcome == "fast" and s1.ok and s1.done and not s1.cancelled
        s2 = lc.LifecycleSnapshot(status=lc.transition(tr, req, _LS.CANCELLED))
        assert s2.cancelled and s2.done and not s2.ok

    def test_oracle4_derived_booleans_agreement(self) -> None:
        fail = lc.LifecycleFailure(code="E", stage="s", message="m")
        for status in _LS:
            snap = lc.LifecycleSnapshot(
                status=status, failure=fail if status is _LS.FAILED else None
            )
            assert snap.ok == (status is _LS.SUCCEEDED) == snap.success
            assert snap.done == lc.is_terminal(status)
            assert snap.cancelled == (status is _LS.CANCELLED)

    def test_oracle4_snapshot_and_failure_validation(self) -> None:
        fail = lc.LifecycleFailure(code="ERR", stage="stage", message="msg")
        bad_kwargs: dict[str, Any] = {"status": _LS.SUCCEEDED, "ok": True}
        with pytest.raises(TypeError, match="unexpected keyword argument 'ok'"):
            lc.LifecycleSnapshot(**bad_kwargs)
        with pytest.raises(ValueError, match="cannot carry a failure"):
            lc.LifecycleSnapshot(status=_LS.SUCCEEDED, failure=fail)
        with pytest.raises(ValueError, match="failure required on FAILED"):
            lc.LifecycleSnapshot(status=_LS.FAILED, failure=None)
        bad_status: Any = "succeeded"
        with pytest.raises(TypeError, match="instance of LifecycleStatus"):
            lc.LifecycleSnapshot(status=bad_status)
        for bad_ver in (0, True):
            bad_v: Any = bad_ver
            with pytest.raises(ValueError, match="schema_version"):
                lc.LifecycleSnapshot(status=_LS.QUEUED, schema_version=bad_v)
        for c, s, m in ((" ", "s", "m"), ("c", "", "m"), ("c", "s", "  ")):
            with pytest.raises(ValueError):
                lc.LifecycleFailure(code=c, stage=s, message=m)
