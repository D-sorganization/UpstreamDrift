"""Shared theme REST API router for FastAPI applications.

Provides a reusable router factory that any app with FastAPI can mount
to expose theme CRUD operations via REST endpoints.

Usage:
    from shared.python.theme.api import create_theme_router
    from shared.python.theme import ThemeManager

    manager = ThemeManager.instance()
    theme_router = create_theme_router(manager)
    app.include_router(theme_router, prefix="/api/v1/themes", tags=["Themes"])
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ThemeColors(BaseModel):
    """Color dictionary for a theme."""

    bg: str = Field(..., description="Main background color")
    group_bg: str = Field(..., description="Group/panel background")
    border: str = Field(..., description="Border color")
    text: str = Field(..., description="Primary text color")
    text_secondary: str = Field(..., description="Secondary text color")
    label: str = Field(..., description="Label/muted text color")
    focus: str = Field(..., description="Focus highlight color")
    input_bg: str = Field(..., description="Input background color")
    accent: str = Field(..., description="Accent color")
    title_bg: str = Field(..., description="Title background color")
    title_border: str = Field(..., description="Title border color")
    table_header: str = Field(..., description="Table header background")
    table_alt: str = Field(..., description="Alternate table row color")
    button_hover: str = Field(..., description="Button hover color")


class ThemeDefinition(BaseModel):
    """Full theme definition with name and colors."""

    name: str
    is_builtin: bool = False
    colors: dict[str, str]


class ThemeListResponse(BaseModel):
    """Response for listing themes."""

    themes: dict[str, ThemeDefinition]


class ActiveThemeResponse(BaseModel):
    """Response for the active theme."""

    name: str
    is_builtin: bool
    colors: dict[str, str]


class SetActiveThemeRequest(BaseModel):
    """Request to change the active theme."""

    name: str = Field(..., description="Theme name to activate")


class SaveCustomThemeRequest(BaseModel):
    """Request to create or update a custom theme."""

    name: str = Field(..., description="Custom theme name")
    colors: dict[str, str] = Field(..., description="Color key-value pairs")
    apply: bool = Field(False, description="Apply theme immediately after saving")


class ThemeOperationResponse(BaseModel):
    """Generic response for theme operations."""

    success: bool
    message: str
    theme_name: str | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def _register_builtin_endpoints(router: APIRouter, theme_manager: Any) -> None:
    """Register built-in theme listing endpoint."""

    @router.get(
        "/builtin",
        response_model=ThemeListResponse,
        summary="List all built-in themes",
    )
    async def get_builtin_themes() -> ThemeListResponse:
        """Return all built-in theme definitions."""
        themes: dict[str, ThemeDefinition] = {}
        for name in theme_manager.get_builtin_themes():
            colors = theme_manager.get_theme_colors(name)
            if colors:
                themes[name] = ThemeDefinition(
                    name=name, is_builtin=True, colors=colors
                )
        return ThemeListResponse(themes=themes)


def _register_custom_endpoints(router: APIRouter, theme_manager: Any) -> None:
    """Register custom theme CRUD endpoints."""

    if router is None:
        raise ValueError("router must be provided")

    @router.get(
        "/custom",
        response_model=ThemeListResponse,
        summary="List all custom themes",
    )
    async def get_custom_themes() -> ThemeListResponse:
        """Return all user-defined custom themes."""
        themes: dict[str, ThemeDefinition] = {}
        for name in theme_manager.get_custom_theme_names():
            colors = theme_manager.get_theme_colors(name)
            if colors:
                themes[name] = ThemeDefinition(
                    name=name, is_builtin=False, colors=colors
                )
        return ThemeListResponse(themes=themes)

    @router.post(
        "/custom",
        response_model=ThemeOperationResponse,
        summary="Create or update a custom theme",
    )
    async def save_custom_theme(
        request: SaveCustomThemeRequest,
    ) -> ThemeOperationResponse:
        """Save a custom theme. Overwrites if the name already exists."""
        try:
            saved_name = theme_manager.save_custom_theme(
                request.name, request.colors, request.apply
            )
            return ThemeOperationResponse(
                success=True,
                message=f"Theme '{saved_name}' saved successfully",
                theme_name=saved_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete(
        "/custom/{theme_id}",
        response_model=ThemeOperationResponse,
        summary="Delete a custom theme",
    )
    async def delete_custom_theme(theme_id: str) -> ThemeOperationResponse:
        """Delete a user-defined custom theme by name."""
        success = theme_manager.delete_custom_theme(theme_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Custom theme '{theme_id}' not found"
            )
        return ThemeOperationResponse(
            success=True,
            message=f"Theme '{theme_id}' deleted",
            theme_name=theme_id,
        )


def _register_active_and_list_endpoints(router: APIRouter, theme_manager: Any) -> None:
    """Register active theme and full listing endpoints."""

    if router is None:
        raise ValueError("router must be provided")

    @router.get(
        "/active",
        response_model=ActiveThemeResponse,
        summary="Get the currently active theme",
    )
    async def get_active_theme() -> ActiveThemeResponse:
        """Return the currently active theme with its colors."""
        name = theme_manager.get_current_theme_name()
        colors = theme_manager.get_current_colors()
        is_builtin = name in theme_manager.get_builtin_themes()
        return ActiveThemeResponse(name=name, is_builtin=is_builtin, colors=colors)

    @router.put(
        "/active",
        response_model=ThemeOperationResponse,
        summary="Set the active theme",
    )
    async def set_active_theme(
        request: SetActiveThemeRequest,
    ) -> ThemeOperationResponse:
        """Change the currently active theme."""
        available = theme_manager.get_available_themes()
        if request.name not in available:
            raise HTTPException(
                status_code=404,
                detail=f"Theme '{request.name}' not found. "
                f"Available: {', '.join(available)}",
            )
        theme_manager.change_theme(request.name)
        return ThemeOperationResponse(
            success=True,
            message=f"Active theme set to '{request.name}'",
            theme_name=request.name,
        )

    @router.get(
        "/",
        response_model=ThemeListResponse,
        summary="List all available themes",
    )
    async def get_all_themes() -> ThemeListResponse:
        """Return all themes (built-in and custom)."""
        themes: dict[str, ThemeDefinition] = {}

        for name in theme_manager.get_builtin_themes():
            colors = theme_manager.get_theme_colors(name)
            if colors:
                themes[name] = ThemeDefinition(
                    name=name, is_builtin=True, colors=colors
                )

        for name in theme_manager.get_custom_theme_names():
            colors = theme_manager.get_theme_colors(name)
            if colors:
                themes[name] = ThemeDefinition(
                    name=name, is_builtin=False, colors=colors
                )

        return ThemeListResponse(themes=themes)


def create_theme_router(theme_manager: Any) -> APIRouter:
    """Create a FastAPI router for theme CRUD operations.

    The router exposes endpoints for listing, creating, updating, and
    deleting themes. It wraps an existing ThemeManager instance.

    Args:
        theme_manager: A ThemeManager instance (from shared.python.theme)

    Returns:
        FastAPI APIRouter ready to be mounted
    """
    router = APIRouter()
    _register_builtin_endpoints(router, theme_manager)
    _register_custom_endpoints(router, theme_manager)
    _register_active_and_list_endpoints(router, theme_manager)
    return router
