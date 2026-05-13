"""Theme management endpoints for UpstreamDrift.

Provides access to the fleet-wide ThemeManager so the React UI can synchronize
its styling with the PyQt6 desktop launcher.
"""

import logging

from fastapi import APIRouter

from src.shared.python.theme.api import create_theme_router
from src.shared.python.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)

# Initialize the theme manager singleton
# We use the standard D-sorganization FleetTheme settings path so it picks up
# what the desktop launcher has saved.
theme_manager = ThemeManager.instance(
    settings_org="D-sorganization", settings_app="FleetTheme"
)

# Create the router using the shared factory
# This will expose /themes/active, /themes/, /themes/builtin, /themes/custom endpoints
# The /themes prefix ensures proper routing under /api/v1/themes when mounted
router = create_theme_router(theme_manager, prefix="/themes")
