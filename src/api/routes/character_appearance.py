"""Character Appearance API routes (CMB-7b, #11658).

The material-library listing and a validated ``appearance-v1`` sidecar,
optionally bound to a compiled character's spec hash so a picked appearance
travels with the physics spec it was made for. Sibling to
``character_builder.py``; routes are discovered the same way (see
``src/api/route_registry.py``) and are registered under the shared ``/api``
prefix, so the final paths are ``/api/character-builder/appearance/...``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from src.api.services.character_appearance_service import (
    appearance_library,
    appearance_sidecar_filename,
    build_appearance,
    grip_revision_summary,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    GripOverrides,
)

from ..models.requests import CharacterSpecRequest
from .character_builder import _compile

logger = logging.getLogger(__name__)
router = APIRouter()

_CHOICE_FIELDS = (
    "skin_tone",
    "clothing",
    "club_finish",
    "headwear",
    "headwear_material",
    "ground_material",
    "skin_color",
    "hair_color",
    "top_color",
    "bottom_color",
    "glove",
    "grip_style",
    "grip_thumb_offset_deg",
    "name",
)


_HEX_PATTERN = r"^#[0-9a-fA-F]{6}$"


class CharacterAppearanceRequest(BaseModel):
    """Appearance picks, optionally bound to a compiled character spec.

    ``character``, when given, is compiled the same way as
    ``/character-builder/build`` and its spec hash is stamped onto the
    returned appearance document as ``spec_sha256``.
    """

    character: CharacterSpecRequest | None = Field(
        None, description="Optional character spec to bind spec_sha256 to"
    )
    skin_tone: str | None = Field(None, description="Skin tone material name")
    clothing: str | None = Field(None, description="Clothing preset name")
    club_finish: str | None = Field(None, description="Club finish material name")
    headwear: str | None = Field(None, description="'none', 'hair' or 'cap'")
    headwear_material: str | None = Field(
        None, description="Hair/cap material name override"
    )
    ground_material: str | None = Field(None, description="Ground material name")
    skin_color: str | None = Field(
        None, pattern=_HEX_PATTERN, description="Custom skin colour #rrggbb"
    )
    hair_color: str | None = Field(
        None, pattern=_HEX_PATTERN, description="Custom hair colour #rrggbb"
    )
    top_color: str | None = Field(
        None, pattern=_HEX_PATTERN, description="Custom shirt colour #rrggbb"
    )
    bottom_color: str | None = Field(
        None, pattern=_HEX_PATTERN, description="Custom trousers/shorts colour #rrggbb"
    )
    glove: Literal["none", "lead", "trail", "both"] | None = Field(
        None,
        description="Golf glove by role; the lead hand is the left of a right-hander",
    )
    grip_style: Literal["overlap", "interlock", "ten_finger"] | None = Field(
        None, description="Grip style (picture only): overlap, interlock, ten_finger"
    )
    grip_thumb_offset_deg: float | None = Field(
        None, ge=0.0, le=45.0, description="Lead thumb offset, degrees (picture only)"
    )
    # The name becomes the export filename stem, so it is restricted to a
    # header- and path-safe identifier.
    name: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        description="Appearance document name (letters, digits, '_' or '-')",
    )

    model_config = {"extra": "forbid"}


def _choices(request: CharacterAppearanceRequest) -> dict[str, Any]:
    return {
        field: getattr(request, field)
        for field in _CHOICE_FIELDS
        if getattr(request, field) is not None
    }


def _build(
    request: CharacterAppearanceRequest,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Compile the optional character, then build and validate the appearance.

    Maps domain errors to HTTP statuses the same way ``character_builder``'s
    ``_compile`` already does for an unknown preset (404); an unknown
    appearance name (skin tone, clothing, finish or material) is a 422.
    """
    spec_sha256: str | None = None
    preset_id: str | None = None
    if request.character is not None:
        compiled = _compile(request.character)
        spec_sha256 = compiled.sha256
        preset_id = compiled.preset_id
    try:
        document = build_appearance(_choices(request), spec_sha256)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return document, spec_sha256, preset_id


@router.get("/character-builder/appearance/library")
def get_appearance_library() -> dict[str, Any]:
    """List skin tones, clothing presets, finishes and every named material."""
    return appearance_library()


@router.post("/character-builder/appearance")
def build_character_appearance(request: CharacterAppearanceRequest) -> dict[str, Any]:
    """Build and validate an appearance document from the given picks."""
    document, spec_sha256, _preset_id = _build(request)
    return {"appearance": document, "spec_sha256": spec_sha256}


@router.post(
    "/character-builder/appearance/export",
    response_class=Response,
    responses={200: {"description": "appearance-v1 JSON sidecar download."}},
)
def export_character_appearance(request: CharacterAppearanceRequest) -> Response:
    """Export the picked appearance as a downloadable JSON sidecar."""
    document, spec_sha256, preset_id = _build(request)
    filename = appearance_sidecar_filename(
        document, preset_id=preset_id, spec_sha256=spec_sha256
    )
    return Response(
        content=json.dumps(document, indent=2) + "\n",
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class GripRevisionRequest(BaseModel):
    """Grip picks for a character: physical ones make a new model revision.

    Golf words: ``preset`` is the grip strength; the ``*_shaft_rotation``
    values turn a hand about the shaft (positive is stronger); the
    ``*_palm_diagonal`` values set where the shaft crosses the palm; the
    ``*_set_angle`` values the angle at the wrist; ``lead_below_butt`` and
    ``hand_spacing`` are metres.  ``grip_style`` and ``grip_thumb_offset_deg``
    are picture-only.
    """

    character: CharacterSpecRequest | None = None
    preset: (
        Literal[
            "weak",
            "neutral",
            "strong",
            "palmer_overlap",
            "woods_interlock",
            "ten_finger",
        ]
        | None
    ) = None
    lead_shaft_rotation: float | None = None
    trail_shaft_rotation: float | None = None
    lead_palm_diagonal: float | None = None
    trail_palm_diagonal: float | None = None
    lead_set_angle: float | None = None
    trail_set_angle: float | None = None
    lead_below_butt: float | None = None
    hand_spacing: float | None = None
    grip_style: Literal["overlap", "interlock", "ten_finger"] | None = None
    grip_thumb_offset_deg: float | None = None

    model_config = {"extra": "forbid"}


@router.post("/character-builder/grip")
def build_grip_revision(request: GripRevisionRequest) -> dict[str, Any]:
    """Resolve grip picks into the grip, its sidecar block and the spec hashes.

    A physical pick returns the new revision's hashes and ``refit_required``;
    picture-only picks leave ``physics_spec_sha256`` equal to the parent's.
    """
    picks = request.model_dump(exclude={"character"}, exclude_none=True)
    try:
        compiled = _compile(
            request.character or CharacterSpecRequest.model_validate({})
        )
        return grip_revision_summary(compiled, GripOverrides.from_picks(picks))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
