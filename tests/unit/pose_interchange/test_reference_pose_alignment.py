"""Verify the canonical pose convention agrees with the existing reference FK.

The canonical convention deliberately mirrors what
:func:`forward_kinematics` and :func:`reference_golfer_setup` already
use. This test pins that down: feeding ``canonical_from_reference_setup()``
through the canonical FK must reproduce the same Cartesian skeleton as
:func:`forward_kinematics(reference_golfer_setup())`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    reference_golfer_setup,
)
from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
)

pytestmark = pytest.mark.unit


def test_canonical_reference_matches_existing_fk_landmarks() -> None:
    """The canonical reference pose, fed back through the existing FK, matches.

    This is a no-op equality at the dict level: both paths come from
    the same :func:`reference_golfer_setup` source. The test exists to
    guard against future drift where someone "improves" the canonical
    convention in a way that silently desynchronises from the FK input
    contract.
    """
    canonical = canonical_from_reference_setup()
    canonical_dict = canonical.angles_full_dict_deg()
    reference_dict = reference_golfer_setup()

    # Every reference field must appear with the same numeric value.
    for key, value in reference_dict.items():
        assert canonical_dict[key] == pytest.approx(value), (
            f"Canonical pose drifted from reference for {key}: "
            f"{canonical_dict[key]} vs {value}"
        )


def test_canonical_pose_reaches_same_landmarks_as_raw_reference() -> None:
    """End-effector positions match between the two equivalent inputs."""
    canonical = canonical_from_reference_setup()
    pose_via_canonical = forward_kinematics(canonical.angles_full_dict_deg())
    pose_via_reference = forward_kinematics(reference_golfer_setup())

    assert set(pose_via_canonical.points.keys()) == set(
        pose_via_reference.points.keys()
    )
    for landmark, p_canon in pose_via_canonical.points.items():
        p_ref = pose_via_reference.points[landmark]
        np.testing.assert_allclose(
            p_canon,
            p_ref,
            atol=1e-9,
            err_msg=f"Landmark {landmark!r} drifted between canonical and reference",
        )


def test_zero_canonical_pose_does_not_match_reference() -> None:
    """Sanity: the canonical *zero* pose differs from the reference setup.

    Catches a class of bugs where someone accidentally returns the
    zero pose from :func:`canonical_from_reference_setup`.
    """
    from src.shared.python.pose_interchange.canonical import canonical_zero_pose

    zero_pose = forward_kinematics(canonical_zero_pose().angles_full_dict_deg())
    ref_pose = forward_kinematics(reference_golfer_setup())

    differing = [
        landmark
        for landmark in zero_pose.points
        if not np.allclose(zero_pose.points[landmark], ref_pose.points[landmark])
    ]
    assert differing, "zero canonical pose unexpectedly equals reference pose"
