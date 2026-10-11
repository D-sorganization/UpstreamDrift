"""Character Builder API routes.

``/character-builder/generate`` returns a mesh-side humanoid URDF from height,
weight and build type. The spec-native endpoints (CMB-3, #11654) list
presets, build and preview a ``full-body-v1`` character, and export it as
spec JSON, URDF, MJCF or OpenSim XML through the shared exporters.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.humanoid_character_builder import spec_export as service

from ..dependencies import get_logger
from ..models.requests import CharacterBuilderRequest, CharacterSpecRequest

logger = logging.getLogger(__name__)
router = APIRouter()

_OVERRIDE_FIELDS = (
    "stature_m",
    "mass_kg",
    "trunk_scale",
    "arm_scale",
    "shoulder_scale",
    "grip_roll_deg",
    "club",
)


def _compile(request: CharacterSpecRequest) -> service.CompiledCharacter:
    """Compile a request, mapping domain errors to HTTP statuses."""
    overrides = {name: getattr(request, name) for name in _OVERRIDE_FIELDS}
    try:
        return service.compile_character(request.preset, overrides)
    except ValueError as exc:
        status = 404 if str(exc).startswith("Unknown character preset") else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Character builder reference assets are unavailable: {exc}",
        ) from exc


@router.get("/character-builder/presets")
def list_character_presets() -> dict[str, Any]:
    """List shipped character presets with their parameters and limitations."""
    return {"presets": service.preset_listing()}


@router.post("/character-builder/build")
def build_character(request: CharacterSpecRequest) -> dict[str, Any]:
    """Compile parameters to a full-body spec and return its summary."""
    return service.build_summary(_compile(request))


@router.post("/character-builder/preview")
def preview_character(request: CharacterSpecRequest) -> dict[str, Any]:
    """Return body masses and joint topology for a preview panel."""
    return service.build_preview(_compile(request))


@router.post(
    "/character-builder/export/{fmt}",
    response_class=Response,
    responses={200: {"description": "Spec JSON, URDF, MJCF or OpenSim XML."}},
)
def export_character(fmt: str, request: CharacterSpecRequest) -> Response:
    """Compile and export as ``spec``, ``urdf``, ``mjcf`` or ``osim``."""
    if fmt not in service.EXPORT_FORMATS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported export format {fmt!r}; "
            f"use one of {sorted(service.EXPORT_FORMATS)}",
        )
    text, media_type, filename = service.export_character(_compile(request), fmt)
    return Response(
        content=text,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _load_character_builder_provider() -> tuple[type[Any], type[Any], type[Any]]:
    from src.shared.python.humanoid_character_builder.core.body_parameters import (
        BodyParameters,
        BuildType,
    )
    from src.shared.python.humanoid_character_builder.generators.urdf_generator import (
        HumanoidURDFGenerator,
    )

    return BodyParameters, BuildType, HumanoidURDFGenerator


@router.post(
    "/character-builder/generate",
    response_class=Response,
    responses={
        200: {
            "content": {"text/xml": {}},
            "description": "Generated URDF XML content.",
        }
    },
)
@handle_api_errors
async def generate_character_urdf(
    request: CharacterBuilderRequest,
    logger: Any = Depends(get_logger),
) -> Response:
    """Generate a custom humanoid URDF model from body parameters.

    Args:
        request: CharacterBuilderRequest with height, weight, and build type.
        logger: Injected logger.

    Returns:
        Response containing URDF XML.
    """
    if logger:
        logger.info(
            "Generating humanoid URDF: height=%.2fm, mass=%.1fkg, build=%s",
            request.height_m,
            request.mass_kg,
            request.build_type,
        )

    BodyParameters, BuildType, HumanoidURDFGenerator = (
        _load_character_builder_provider()
    )
    build_map = {
        "athletic": BuildType.MESOMORPH,
        "average": BuildType.AVERAGE,
        "heavy": BuildType.ENDOMORPH,
        "slim": BuildType.ECTOMORPH,
    }

    try:
        params = BodyParameters(
            height_m=request.height_m,
            mass_kg=request.mass_kg,
            build_type=build_map[request.build_type],
        )

        generator = HumanoidURDFGenerator()
        urdf_xml = generator.generate(params)

        return Response(
            content=urdf_xml,
            media_type="text/xml",
            headers={
                "Content-Disposition": f'attachment; filename="{request.build_type.lower()}_humanoid.urdf"'
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        if logger:
            logger.error("Failed to generate humanoid URDF: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Character builder generation failed: {str(exc)}",
        ) from exc
