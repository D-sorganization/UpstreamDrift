"""HTTP routes for the runtime feature registry.

Endpoints
---------
``GET  /capabilities``
    Returns the full snapshot — one entry per registered feature.

``GET  /capabilities/{name}``
    Returns a single feature report.

``POST /capabilities/refresh``
    Re-runs every probe and returns the new snapshot. Idempotent.
    Auth-gated in cloud mode.

``POST /capabilities/{name}/install``
    Opt-in install of a missing feature. Refused inside non-root
    Docker containers (use a profile rebuild instead). Auth-gated:
    the caller must be authenticated when ``GOLF_AUTH_DISABLED`` is
    falsy. After a successful install the registry is refreshed
    automatically and the post-install report is returned.

The ``GET`` endpoints stay unauthenticated so a client can negotiate
capabilities before login; the two mutating ``POST`` endpoints carry
``Depends(require_cloud_auth)`` individually because the router as a
whole is in ``_PUBLIC_ROUTERS``.

This route is auto-discovered by ``src.api.route_registry`` — no
explicit registration in ``server.py`` is needed.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.auth.dependencies import require_cloud_auth
from src.shared.python.feature_registry import (
    InstallResult,
    get_registry,
    install_feature,
)

router = APIRouter(tags=["capabilities"])


class FeatureReportModel(BaseModel):
    """Pydantic mirror of :class:`FeatureReport` for the OpenAPI schema."""

    name: str
    display_name: str
    available: bool
    version: str | None = None
    tier: str
    docker_stage: str | None = None
    install_channel: str
    install_command: str
    pip_extra: str | None = None
    approx_size_mb: int
    message: str
    missing: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class InstallRequest(BaseModel):
    """Body for ``POST /capabilities/{name}/install``."""

    allow_user_site: bool = Field(
        default=False,
        description="Pass --user to pip. Default false: install into the active venv.",
    )
    dry_run: bool = Field(
        default=False,
        description="Return the command we would run without executing it.",
    )
    timeout_seconds: float = Field(default=600.0, gt=0.0, le=3600.0)


class InstallResponse(BaseModel):
    """Body for the install endpoint's response."""

    install: dict[str, Any]
    post_install_report: FeatureReportModel | None = None


def _report_to_model(report: Any) -> FeatureReportModel:
    return FeatureReportModel(**report.to_dict())


def _install_to_dict(result: InstallResult) -> dict[str, Any]:
    return {
        "feature": result.feature,
        "success": result.success,
        "command": result.command,
        "returncode": result.returncode,
        "reason": result.reason,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@router.get("/capabilities", response_model=list[FeatureReportModel])
def list_capabilities() -> list[FeatureReportModel]:
    """Return one report per registered feature."""
    snapshot = get_registry().snapshot()
    return [_report_to_model(r) for r in snapshot]


@router.get("/capabilities/candidate_session")
def get_candidate_session_capability(
    candidate_path: str,
    model_path: str | None = None,
    receipt_path: str | None = None,
) -> dict[str, Any]:
    """Ingest candidate session metadata, verify integrity, and report capabilities."""
    from src.shared.python.engine_core.wsl_probe import get_wsl_engine_status
    from src.shared.python.motion_matching.candidate_session import (
        ingest_candidate_session,
    )

    try:
        session = ingest_candidate_session(
            candidate_path=candidate_path,
            model_path=model_path,
            receipt_path=receipt_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    wsl_status = get_wsl_engine_status(session.engine).to_dict()

    return {
        "status": session.status,
        "is_accepted": session.is_accepted,
        "engine": session.engine,
        "candidate_sha256": session.candidate_sha256,
        "model_sha256": session.model_sha256,
        "capabilities": {
            "supports_forces": session.supports_forces,
            "supports_counterfactuals": session.supports_counterfactuals,
            "has_tau": session.tau is not None,
            "has_external_forces": session.external_forces is not None,
        },
        "wsl_runtime": wsl_status,
        "frame_count": session.replay.frame_count,
        "time_range_s": [
            float(session.replay.time_s[0]),
            float(session.replay.time_s[-1]),
        ],
    }


@router.get("/capabilities/{name}", response_model=FeatureReportModel)
def get_capability(name: str) -> FeatureReportModel:
    """Return the report for a single feature."""
    try:
        report = get_registry().check(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return _report_to_model(report)


@router.post(
    "/capabilities/refresh",
    response_model=list[FeatureReportModel],
    dependencies=[Depends(require_cloud_auth)],
)
def refresh_capabilities() -> list[FeatureReportModel]:
    """Re-run every probe and return the new snapshot.

    Auth-gated in cloud mode: probing every feature is an expensive,
    state-mutating operation (issue #7987).
    """
    snapshot = get_registry().refresh()
    return [_report_to_model(r) for r in snapshot]


@router.post(
    "/capabilities/{name}/install",
    response_model=InstallResponse,
    dependencies=[Depends(require_cloud_auth)],
)
def install_capability(name: str, body: InstallRequest) -> InstallResponse:
    """Install a missing feature in the active venv.

    Refused inside non-root Docker containers — the install runner
    returns ``success=False`` with a hint that points to the
    profile-rebuild documentation.

    Returns the install command output plus the post-install feature
    report so the caller can confirm the new state in one round-trip.
    """
    try:
        result = install_feature(
            name,
            allow_user_site=body.allow_user_site,
            timeout=body.timeout_seconds,
            dry_run=body.dry_run,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    post_report: FeatureReportModel | None = None
    if result.success and not body.dry_run:
        get_registry().refresh()
        post_report = _report_to_model(get_registry().check(name))

    return InstallResponse(
        install=_install_to_dict(result),
        post_install_report=post_report,
    )


@router.get("/engines/matrix")
def get_engine_matrix() -> dict[str, Any]:
    """Get the authoritative engine capability and qualification matrix."""
    import json
    from src.config.launcher_manifest_loader import CONFIG_DIR

    matrix_path = CONFIG_DIR / "engine_capability_matrix.json"
    if not matrix_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Engine capability matrix not found",
        )
    try:
        return cast(dict[str, Any], json.loads(matrix_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read engine capability matrix: {exc}",
        ) from exc
