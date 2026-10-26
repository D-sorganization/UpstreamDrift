"""SubjectAnthropometrics hand length and breadth (#13760)."""

from __future__ import annotations

import numpy as np
import pytest

from anthropometrics import SegmentProperties, Sex, SubjectAnthropometrics
from src.shared.python.anthropometrics import hand_size


def _segment(name: str = "hand_left") -> SegmentProperties:
    return SegmentProperties(
        name=name,
        body_part_id="hand",
        length_m=0.19,
        proximal_marker="L_WRI",
        distal_marker="L_FIN",
        mass_kg=0.4,
        com_xyz_m=np.zeros(3),
        inertia_tensor=np.diag([0.001, 0.001, 0.001]),
        source_method="test",
        source_subject_height_m=1.75,
        source_subject_mass_kg=75.0,
    )


@pytest.mark.unit
def test_subject_defaults_hand_size_from_stature_and_sex() -> None:
    subject = SubjectAnthropometrics(
        subject_id="S1",
        height_m=1.80,
        mass_kg=75.0,
        segments=(("hand_left", _segment()),),
        source_method="test",
        sex=Sex.FEMALE.value,
    )
    expected_length = hand_size.resolve_hand_length_m(1.80, Sex.FEMALE.value)
    expected_breadth = hand_size.resolve_hand_breadth_m(
        expected_length, Sex.FEMALE.value
    )
    assert subject.hand_length_m == pytest.approx(expected_length)
    assert subject.hand_breadth_m == pytest.approx(expected_breadth)


@pytest.mark.unit
def test_contact_topology_and_subject_share_hand_length() -> None:
    from src.shared.python.grip_contact.contact_topology import HandAnatomy

    stature = 1.75
    subject = SubjectAnthropometrics(
        subject_id="S3",
        height_m=stature,
        mass_kg=75.0,
        segments=(("hand_left", _segment()),),
        source_method="test",
    )
    length = hand_size.resolve_hand_length_m(stature)
    assert subject.hand_length_m == pytest.approx(length)
    assert HandAnatomy(hand_length_m=length).hand_length_m == pytest.approx(length)


@pytest.mark.unit
def test_subject_hand_overrides_preserved() -> None:
    subject = SubjectAnthropometrics(
        subject_id="S2",
        height_m=1.80,
        mass_kg=75.0,
        segments=(("hand_left", _segment()),),
        source_method="test",
        hand_length_m=0.20,
        hand_breadth_m=0.09,
    )
    assert subject.hand_length_m == 0.20
    assert subject.hand_breadth_m == 0.09
