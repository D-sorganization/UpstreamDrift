"""Lead-finger curl sequencing for GB-2 (issue #15124)."""

from __future__ import annotations

from typing import Final

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.grip_hand_mesh import FINGER_NAMES

LEAD_FINGER_CURL_ORDER: Final[tuple[str, ...]] = FINGER_NAMES


def lead_finger_curl_order(spec: GripSpec) -> tuple[str, ...]:
    """Resolve index→pinky order from ``GripSpec.curl_order`` when finger tokens appear."""
    finger_set = set(LEAD_FINGER_CURL_ORDER)
    picked = tuple(name for name in spec.curl_order if name in finger_set)
    return picked or LEAD_FINGER_CURL_ORDER
