"""Character Builder API routes.

Provides an endpoint to generate a custom humanoid URDF based on height,
weight, and build type.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from humanoid_character_builder.core.body_parameters import BodyParameters, BuildType
from humanoid_character_builder.generators.urdf_generator import HumanoidURDFGenerator
from src.api.middleware.error_handler import handle_api_errors

from ..dependencies import get_logger
from ..models.requests import CharacterBuilderRequest

logger = logging.getLogger(__name__)
router = APIRouter()


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
    except Exception as exc:
        if logger:
            logger.error("Failed to generate humanoid URDF: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Character builder generation failed: {str(exc)}",
        ) from exc
