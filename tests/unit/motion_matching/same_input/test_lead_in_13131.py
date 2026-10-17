"""Lead-in of a closed-loop reference run (OSL-17, #13131).

A reference run starts with the feet preloaded for the static weight, so its
first states settle onto the contact.  ``drop_lead_in`` keeps the run from
``steps`` on and records what it dropped; ``generate_lead_in_bundle``
validates the lead-in before it runs anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.constants import DT_S
from src.shared.python.motion_matching.same_input import Rollout
from src.shared.python.motion_matching.same_input.reference import (
    drop_lead_in,
    generate_lead_in_bundle,
)

pytestmark = pytest.mark.unit

STEPS, NV, HOLDS = 10, 3, 4


def _rollout() -> Rollout:
    rng = np.random.default_rng(13131)
    return Rollout(
        time_s=np.arange(STEPS + 1) * DT_S + 0.5,
        q=rng.normal(size=(STEPS + 1, NV)),
        v=rng.normal(size=(STEPS + 1, NV)),
        efforts=rng.normal(size=(STEPS * HOLDS, NV)),
        pose_drift=rng.uniform(size=STEPS),
    )


def test_no_lead_in_returns_the_run_unchanged() -> None:
    rollout = _rollout()
    trimmed, record = drop_lead_in(rollout, 0, HOLDS)
    assert trimmed is rollout
    assert record == {}


def test_lead_in_is_the_tail_of_the_run_and_its_record() -> None:
    rollout, lead = _rollout(), 3
    trimmed, record = drop_lead_in(rollout, lead, HOLDS)
    np.testing.assert_array_equal(trimmed.q, rollout.q[lead:])
    np.testing.assert_array_equal(trimmed.v, rollout.v[lead:])
    np.testing.assert_array_equal(trimmed.efforts, rollout.efforts[lead * HOLDS :])
    np.testing.assert_array_equal(trimmed.pose_drift, rollout.pose_drift[lead:])
    assert trimmed.time_s[0] == 0.0
    np.testing.assert_allclose(np.diff(trimmed.time_s), DT_S)
    info = record["lead_in"]
    assert info["steps"] == lead
    assert info["duration_s"] == pytest.approx(lead * DT_S)
    assert info["max_pose_drift_m"] == pytest.approx(rollout.pose_drift[:lead].max())
    np.testing.assert_array_equal(info["reference_q"], rollout.q[:lead])


def test_lead_in_must_leave_a_bundle_step() -> None:
    with pytest.raises(ValueError, match="at least one"):
        drop_lead_in(_rollout(), STEPS, HOLDS)


@pytest.mark.parametrize(
    ("lead_in", "error"), [(-1, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_generate_rejects_a_bad_lead_in(lead_in: object, error: type) -> None:
    with pytest.raises(error):
        generate_lead_in_bundle(
            b"{}",
            np.zeros(2),
            np.zeros((2, 1)),
            duration_s=0.001,
            lead_in_steps=lead_in,  # type: ignore[arg-type]
        )
