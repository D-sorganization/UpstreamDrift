"""Spec revisions ``build_anthropometric_spec`` applies after composing (#15648).

The builder writes a spec on the current golfer model
(``src/config/golfer_model_versions.json``): the physical grip picks (GR4)
first, then the skinned hand wrists, posed from those picks (#15654), and the skinned
body. A left-handed spec keeps the deprecated capsule hands and loft body,
logging a deprecation, because the skinned pair is posed right-handed
(#15652).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.appearance_selection import (
    resolve_appearance,
)
from src.shared.python.model_appearance.golfer_model_versions import (
    current_golfer_model,
    warn_if_deprecated,
)
from src.shared.python.model_appearance.schema import appearance_path_for
from src.shared.python.motion_matching.execution.grip_selection import (
    GripOverrides,
    apply_grip_selection,
    apply_visual_picks,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)

_BACKENDS = ("hand_visual_backend", "body_visual_backend")


def with_grip_revision(
    document: dict[str, Any],
    subject: dict[str, Any],
    overrides: GripOverrides | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The document and receipt subject after the physical grip picks (GR4).

    Physical picks make a GR2 spec revision (a refit is needed); picture-only
    picks do not touch the spec (they belong in an appearance sidecar).
    """
    if overrides is None or not overrides.has_physical:
        return document, subject
    revised, _grip = apply_grip_selection(document, overrides)
    return revised, {**subject, **revised["subject"]}


def write_grip_sidecar(spec_path: Path, overrides: GripOverrides | None) -> None:
    """Write the picture-only grip picks beside the spec (never physical ones)."""
    if overrides is None or not overrides.has_visual:
        return
    sidecar = resolve_appearance(None, apply_visual_picks({}, overrides.visual_only()))
    appearance_path_for(spec_path).write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


def with_current_golfer_model(
    document: Mapping[str, Any], subject: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The document and receipt subject on the current golfer model.

    Right-handed: the skinned hand wrist revision plus the registry's body.
    Left-handed: unchanged, with a deprecation logged for each old part.
    Postconditions: the input is not mutated; a right-handed result has the
    registry's hand and body backends and spec revision configuration.
    """
    if subject_handedness(document) == "left":
        for field in _BACKENDS:
            warn_if_deprecated(document, field)
        return dict(document), dict(subject)
    current = current_golfer_model()
    revised = apply_skinned_hand_wrists(document)
    revised["subject"]["body_visual_backend"] = current.body_visual_backend
    if (
        revised["subject"]["hand_visual_backend"] != current.hand_visual_backend
        or revised["spec_revision"]["configuration"]
        != current.spec_revision_configuration
    ):
        raise ValueError("the wrist revision does not make the registry's model")
    return revised, {**subject, **{key: revised["subject"][key] for key in _BACKENDS}}
