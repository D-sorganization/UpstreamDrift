"""Spec revisions ``build_anthropometric_spec`` applies after composing (#15648).

The builder writes a spec on the current golfer model
(``src/config/golfer_model_versions.json``): the physical grip picks (GR4)
first, then the skinned hand wrists, posed from those picks (#15654), and the skinned
body. A left-handed golfer is the exact sagittal mirror of that right-handed
spec (``mirrored_spec``, #15652), so it is on the current model too.

Last comes the current model's flexible shaft (#15684): a chain shaft with
the registry's flex. A rigid club is only the explicit ballpark fallback
(``shaft_model="rigid"``), and choosing it logs a warning.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.shared.python.core.handedness import subject_handedness, validate_handedness
from src.shared.python.model_appearance.appearance_selection import (
    resolve_appearance,
)
from src.shared.python.motion_matching.mirrored_spec import (
    mirror_spec_to_left_handed,
)
from src.shared.python.model_appearance.golfer_model_versions import (
    current_golfer_model,
)
from src.shared.python.model_appearance.schema import appearance_path_for
from src.shared.python.motion_matching.club_models import recorded_club
from src.shared.python.motion_matching.club_shaft_chain import (
    SHAFT_MODELS,
    apply_shaft_chain,
    build_shaft_chain,
    checked_shaft_options,
    shaft_model,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    GripOverrides,
    apply_grip_selection,
    apply_visual_picks,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)

_BACKENDS = ("hand_visual_backend", "body_visual_backend")
logger = logging.getLogger(__name__)


def current_shaft(
    requested: str | None, options: Mapping[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """The shaft model and chain options a current-model spec gets (#15684).

    ``None`` is the registry's flexible shaft, with ``options`` (flex,
    stiffness scale, links) over its flex. ``"rigid"`` is the ballpark
    fallback for a quick first match and logs a warning.

    Raises:
        ValueError: an unknown ``requested`` model or invalid options.
    """
    current = current_golfer_model()
    model = current.shaft_model if requested is None else requested
    if model not in SHAFT_MODELS:
        raise ValueError(f"shaft_model must be one of {SHAFT_MODELS}, got {model!r}")
    if model == "rigid":
        checked_shaft_options(model, options)
        logger.warning(
            "Rigid club: ballpark fallback only; the current golfer model %s "
            "uses a flexible %s shaft, so reintroduce it once the match solves",
            current.version,
            current.shaft_flex,
        )
        return model, {}
    chosen = {"flex": current.shaft_flex, **dict(options or {})}
    checked_shaft_options(model, chosen)
    return model, chosen


def with_current_shaft(
    document: Mapping[str, Any],
    requested: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``document`` with the current model's shaft; a chain spec is kept as is."""
    model, chosen = current_shaft(requested, options)
    if model == "rigid" or shaft_model(document) != "rigid":
        return dict(document)
    club = recorded_club(document, document["club"]["name"])
    return apply_shaft_chain(document, build_shaft_chain(club, **chosen))


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
    document: Mapping[str, Any],
    subject: Mapping[str, Any],
    *,
    shaft_model: str | None = None,
    shaft_options: Mapping[str, Any] | None = None,
    handedness: str = "right",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The document and receipt subject on the current golfer model.

    ``document`` is the composed right-handed spec. The skinned hand wrist
    revision, the registry's body and the current shaft are applied
    (:func:`with_current_shaft`; ``shaft_model="rigid"`` is the logged ballpark
    fallback). ``handedness="left"`` then mirrors the result through the
    sagittal plane (:func:`mirror_spec_to_left_handed`).

    Raises:
        ValueError: ``document`` is labelled left-handed (compose right-handed
            and pass ``handedness='left'``), or an unknown ``handedness``.

    Postconditions: the input is not mutated; the result has the registry's
    hand and body backends and spec revision configuration, and
    ``subject.handedness`` equals ``handedness``.
    """
    side = validate_handedness(handedness)
    if subject_handedness(document) == "left":
        raise ValueError(
            "compose the right-handed spec and pass handedness='left'; a left "
            "label on right-handed geometry draws right-handed hands"
        )
    current = current_golfer_model()
    revised = apply_skinned_hand_wrists(document)
    revised["subject"]["body_visual_backend"] = current.body_visual_backend
    if (
        revised["subject"]["hand_visual_backend"] != current.hand_visual_backend
        or revised["spec_revision"]["configuration"]
        != current.spec_revision_configuration
    ):
        raise ValueError("the wrist revision does not make the registry's model")
    shafted = with_current_shaft(revised, shaft_model, shaft_options)
    receipt = {**subject, **{key: revised["subject"][key] for key in _BACKENDS}}
    if side == "right":
        return shafted, receipt
    return mirror_spec_to_left_handed(shafted), {**receipt, "handedness": side}
