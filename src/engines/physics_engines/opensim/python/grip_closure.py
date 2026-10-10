"""OpenSim two-hand grip closure: structure and residual (OSV-2, #11728).

``constrained_hands`` reads a ``.osim`` as plain XML (no OpenSim import) and
reports how each hand is tied to the club.  The Rajagopal golf models follow
the OSV-9 topology of ``msk_club``: the lead ``WeldJoint hand_l_to_club``
carries the club and the trail ``WeldConstraint hand_r_to_club`` closes the
loop (or one ``BushingForce`` per hand for the bushing grip).
``weld_closure_series`` evaluates the closure weld of a generated full-body
model over a swing with the OpenSim bindings.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_contact.closure_series import GripClosureSeries

logger = logging.getLogger(__name__)

CLUB_BODY_NAMES = ("Club", "Clubhead")
FULL_BODY_WELD = "two_hand_grip_closure"
FULL_BODY_LEAD_JOINT = "joint_Clubhead"
_ELEMENT = re.compile(
    r"<(?P<tag>WeldJoint|WeldConstraint|PointConstraint|BushingForce) "
    r'name="(?P<name>[^"]*)">'
    r".*?</(?P=tag)>",
    re.DOTALL,
)


def constrained_hands(model_path: Path | str) -> dict[str, str]:
    """Map ``"lead"``/``"trail"`` to the element that ties that hand to the club.

    Values are ``"<Tag>:<name>"``.  Hands that are not tied are absent.  The
    file is scanned as text: OpenSim writes ``Class::Name`` element names that
    strict XML parsers reject.  See :func:`constrained_hands_from_text`.
    """
    path = Path(model_path)
    require(path.is_file(), f"model not found: {path}")
    return constrained_hands_from_text(path.read_text(encoding="utf-8"))


def constrained_hands_from_text(text: str) -> dict[str, str]:
    """:func:`constrained_hands` of a model's XML text.

    Golf humanoid (OSV-9): ``hand_l`` through the lead weld joint (or its
    bushing), ``hand_r`` through the trail weld constraint (or its bushing).
    Generated full-body model: the lead hand is the parent chain of the club
    joint and the trail hand is the closure weld.
    """
    require(isinstance(text, str), "text must be a string")
    found: dict[str, str] = {}
    for match in _ELEMENT.finditer(text):
        tag, name, body = match.group("tag"), match.group("name"), match.group(0)
        if not any(f"/bodyset/{club}" in body for club in CLUB_BODY_NAMES):
            continue
        if "/bodyset/hand_r" in body or name == FULL_BODY_WELD:
            found["trail"] = f"{tag}:{name}"
        elif "/bodyset/hand_l" in body:
            found["lead"] = f"{tag}:{name}"
    if f'<CustomJoint name="{FULL_BODY_LEAD_JOINT}"' in text:
        found["lead"] = f"CustomJoint:{FULL_BODY_LEAD_JOINT}"
    return found


def weld_closure_series(
    model_path: Path | str,
    coordinate_names: Sequence[str],
    q: np.ndarray,
) -> GripClosureSeries:
    """Translational gap of the closure weld frames at each swing frame.

    The frames are the model's own ``two_hand_grip_closure`` weld frames, so a
    model without that weld (or without the OpenSim bindings) is unavailable.
    """
    frames = np.asarray(q, dtype=float)
    require(
        frames.ndim == 2 and frames.shape[1] == len(coordinate_names),
        "q must be (frames, len(coordinate_names))",
    )
    try:
        import opensim as osim  # noqa: PLC0415 - optional heavy binding
    except ImportError as exc:
        return GripClosureSeries.unavailable(
            "opensim", f"opensim not importable: {exc}"
        )
    model = osim.Model(str(model_path))
    state = model.initSystem()
    try:
        frame_a = model.getComponent(f"/constraintset/{FULL_BODY_WELD}/closure_frame_a")
        frame_b = model.getComponent(f"/constraintset/{FULL_BODY_WELD}/closure_frame_b")
    except RuntimeError as exc:
        return GripClosureSeries.unavailable(
            "opensim", f"model has no {FULL_BODY_WELD} weld: {exc}"
        )
    coords = model.getCoordinateSet()
    out = np.empty(frames.shape[0])
    for k, row in enumerate(frames):
        for name, value in zip(coordinate_names, row, strict=True):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        pa = frame_a.getPositionInGround(state)
        pb = frame_b.getPositionInGround(state)
        out[k] = float(np.linalg.norm([pa.get(i) - pb.get(i) for i in range(3)]))
    return GripClosureSeries("opensim", out)
