"""About/version information routes (issue #7459, parity G11).

Exposes the same version information as the desktop About dialog
(``src/launchers/about_dialog.py``) so web users can see — and include in
bug reports — the exact backend version. The app-version resolution chain
is shared with the desktop dialog via
``src.shared.python.version_info`` (VERSION file -> importlib.metadata ->
fallback), satisfying the "one implementation" acceptance criterion.

Also serves the single-sourced onboarding card copy
(``src/config/onboarding_cards.json``) consumed by both the Qt onboarding
dialog and the web onboarding overlay.
"""

from __future__ import annotations

import json
import platform
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.shared.python.version_info import (
    ISSUES_URL,
    REPO_URL,
    USER_GUIDE_URL,
    get_repo_root,
    installed_dist_version,
    read_git_commit,
    resolve_app_version,
    safe_module_version,
)

router = APIRouter(prefix="/about", tags=["about"])

#: Dependencies probed cheaply via importlib.metadata (no import of the
#: heavy package itself). Keys are response labels, values dist names.
_METADATA_PROBED_DISTS: dict[str, str] = {
    "mujoco": "mujoco",
    "drake": "drake",
    "pinocchio": "pin",
}

_ONBOARDING_CARDS_PATH = get_repo_root() / "src" / "config" / "onboarding_cards.json"


@router.get("")
async def get_about() -> dict[str, Any]:
    """Return backend version information for the web About modal.

    Mirrors the desktop About dialog's content: app version (shared
    resolution chain), Python version, key dependency versions, platform,
    git commit (``None`` when no ``.git`` is present), and support links.
    """
    dependencies = {"numpy": safe_module_version("numpy")}
    dependencies["ezc3d"] = installed_dist_version("ezc3d")
    for label, dist_name in _METADATA_PROBED_DISTS.items():
        dependencies[label] = installed_dist_version(dist_name)

    return {
        "app_name": "UpstreamDrift",
        "app_version": resolve_app_version(),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "git_commit": read_git_commit(),
        "dependencies": dependencies,
        "links": {
            "repository": REPO_URL,
            "report_bug": ISSUES_URL,
            "user_guide": USER_GUIDE_URL,
        },
    }


@router.get("/onboarding")
async def get_onboarding_copy() -> dict[str, Any]:
    """Return the single-sourced onboarding card copy.

    The same JSON drives the desktop onboarding dialog
    (``src/launchers/onboarding_dialog.py``) and the web overlay.
    """
    try:
        with open(_ONBOARDING_CARDS_PATH, encoding="utf-8") as f:
            copy = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Onboarding copy unavailable",
        ) from exc
    if not isinstance(copy, dict) or not isinstance(copy.get("cards"), list):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Onboarding copy malformed",
        )
    return copy
