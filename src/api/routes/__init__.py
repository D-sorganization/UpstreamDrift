"""API route registry.

Centralizes route registration to decouple server.py from individual
route modules.  Each module is imported lazily so that a single broken
dependency does not prevent the rest of the API from starting.

Addresses:
- Issue #1485 (architecture: decouple API layer)
- Issue #1488 (API versioning)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

logger = logging.getLogger(__name__)


def get_all_routers() -> list[APIRouter]:
    """Discover and return all API routers.

    Each import is wrapped in a try/except so that a missing optional
    dependency (e.g. numpy, mediapipe) only disables its own routes
    rather than crashing the entire server.

    Returns:
        List of APIRouter instances to include in the app.
    """
    routers: list[APIRouter] = []

    # --- Core & infrastructure ---
    try:
        from .core import router as core_router

        routers.append(core_router)
    except ImportError:
        logger.warning("Failed to load core routes")

    try:
        from .auth import router as auth_router

        routers.append(auth_router)
    except ImportError:
        logger.warning("Failed to load auth routes")

    # --- Engine management ---
    try:
        from .engines import router as engines_router

        routers.append(engines_router)
    except ImportError:
        logger.warning("Failed to load engine routes")

    # --- Simulation ---
    try:
        from .simulation import router as simulation_router

        routers.append(simulation_router)
    except ImportError:
        logger.warning("Failed to load simulation routes")

    # --- Video & analysis ---
    try:
        from .video import router as video_router

        routers.append(video_router)
    except ImportError:
        logger.warning("Failed to load video routes")

    try:
        from .analysis import router as analysis_router

        routers.append(analysis_router)
    except ImportError:
        logger.warning("Failed to load analysis routes")

    # --- Export ---
    try:
        from .export import router as export_router

        routers.append(export_router)
    except ImportError:
        logger.warning("Failed to load export routes")

    # --- Launcher ---
    try:
        from .launcher import router as launcher_router

        routers.append(launcher_router)
    except ImportError:
        logger.warning("Failed to load launcher routes")

    # --- Terrain ---
    try:
        from .terrain import router as terrain_router

        routers.append(terrain_router)
    except ImportError:
        logger.warning("Failed to load terrain routes")

    # --- Dataset ---
    try:
        from .dataset import router as dataset_router

        routers.append(dataset_router)
    except ImportError:
        logger.warning("Failed to load dataset routes")

    # --- Physics ---
    try:
        from .physics import router as physics_router

        routers.append(physics_router)
    except ImportError:
        logger.warning("Failed to load physics routes")

    # --- Models ---
    try:
        from .models import router as models_router

        routers.append(models_router)
    except ImportError:
        logger.warning("Failed to load model routes")

    # --- Analysis tools ---
    try:
        from .analysis_tools import router as analysis_tools_router

        routers.append(analysis_tools_router)
    except ImportError:
        logger.warning("Failed to load analysis_tools routes")

    # --- Phase 4: Force overlays, actuator controls, model explorer, AIP ---
    try:
        from .force_overlays import router as force_overlay_router

        routers.append(force_overlay_router)
    except ImportError:
        logger.warning("Failed to load force_overlays routes")

    try:
        from .actuator_controls import router as actuator_controls_router

        routers.append(actuator_controls_router)
    except ImportError:
        logger.warning("Failed to load actuator_controls routes")

    try:
        from .model_explorer import router as model_explorer_router

        routers.append(model_explorer_router)
    except ImportError:
        logger.warning("Failed to load model_explorer routes")

    try:
        from .aip import router as aip_router

        routers.append(aip_router)
    except ImportError:
        logger.warning("Failed to load aip routes")

    # --- Phase 5: Tool pages ---
    try:
        from .putting_green import router as putting_green_router

        routers.append(putting_green_router)
    except ImportError:
        logger.warning("Failed to load putting_green routes")

    try:
        from .data_explorer import router as data_explorer_router

        routers.append(data_explorer_router)
    except ImportError:
        logger.warning("Failed to load data_explorer routes")

    try:
        from .motion_capture import router as motion_capture_router

        routers.append(motion_capture_router)
    except ImportError:
        logger.warning("Failed to load motion_capture routes")

    return routers
