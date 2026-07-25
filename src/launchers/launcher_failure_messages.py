"""Turn raw tile-launch exceptions into specific, actionable user messages.

Functional QA (#8062) found that every failure mode of the desktop launcher
presented as either a raw traceback, a blank panel, or nothing at all.  The
guiding rule from that campaign is:

    A missing optional dependency must never take down the launcher, and the
    user must be told *what* is missing and *how* to install it.

This module is deliberately pure (no Qt, no I/O) so the wording is unit
testable and shared by every launch surface: the tile launcher
(``launcher_simulation``), the biomechanics exercise dashboard, and the
provider-model asset handler.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "ENGINE_INSTALL_HINTS",
    "describe_launch_failure",
    "missing_module_hint",
]


#: ``pip`` command that restores each optional engine runtime.
ENGINE_INSTALL_HINTS: Final[dict[str, str]] = {
    "mujoco": "pip install mujoco",
    "drake": "pip install drake",
    "pinocchio": "conda install -c conda-forge pinocchio",
    "opensim": "conda install -c opensim-org opensim",
    "myosuite": "pip install myosuite",
    "myosim": "pip install myosuite",
    "jaxsim": "pip install jaxsim",
    "matplotlib": "pip install upstream-drift[gui-tools]",
    "pyqtgraph": "pip install upstream-drift[gui-tools]",
    "PyQt6": "pip install upstream-drift[gui-tools]",
    "sidekick": "pip install upstream-drift[gui-tools]",
    "humanoid_character_builder": "pip install upstream-drift[gui-tools]",
}

_DEFAULT_HINT: Final[str] = "pip install upstream-drift[gui-tools]"

_MISSING_MODULE_RE: Final[re.Pattern[str]] = re.compile(
    r"No module named ['\"]([\w.]+)['\"]"
)

_DLL_INIT_ADVICE: Final[str] = (
    "A native library failed to initialise (Windows error 1114).\n"
    "Try, in order:\n"
    "  1. pip install --force-reinstall --no-cache-dir {package}\n"
    "  2. Install the Microsoft Visual C++ Redistributable (x64)\n"
    "  3. Update your GPU driver, then restart UpstreamDrift\n"
    "The rest of the launcher is unaffected - other tiles still work."
)


def missing_module_hint(module_name: str) -> str:
    """Return the install command that provides ``module_name``.

    Args:
        module_name: Dotted or top-level module name, e.g. ``"sidekick.lab"``.

    Returns:
        A ``pip``/``conda`` command string.

    Raises:
        ValueError: If ``module_name`` is empty.
    """
    if not module_name or not module_name.strip():
        raise ValueError("module_name must be a non-empty string")
    root = module_name.split(".", 1)[0]
    return ENGINE_INSTALL_HINTS.get(root, _DEFAULT_HINT)


def _dll_package_guess(text: str) -> str:
    for package in ENGINE_INSTALL_HINTS:
        if package.lower() in text.lower():
            return package
    return "the affected package"


def describe_launch_failure(
    exc: BaseException, tile_name: str, package_hint: str | None = None
) -> str:
    """Build a specific, actionable message for a failed tile launch.

    Args:
        exc: The exception raised by the launch attempt.
        tile_name: Human-readable name of the tile the user clicked.
        package_hint: Optional package name to name in DLL-failure advice when
            the exception text does not identify one (e.g. "mujoco").

    Returns:
        A multi-line message naming what is missing and how to fix it.  Never
        contains a traceback — the traceback goes to the log and the console
        dock instead.

    Raises:
        ValueError: If ``tile_name`` is empty.
    """
    if not tile_name or not tile_name.strip():
        raise ValueError("tile_name must be a non-empty string")

    detail = str(exc) or exc.__class__.__name__
    header = f"{tile_name} could not be started."

    # Handlers may raise an exception whose message is already a complete,
    # user-facing explanation (see EngineRuntimeUnavailableError, #8087).
    if getattr(exc, "is_user_facing_message", False):
        return detail

    if isinstance(exc, ModuleNotFoundError | ImportError):
        match = _MISSING_MODULE_RE.search(detail)
        module = match.group(1) if match else (getattr(exc, "name", None) or "")
        if module:
            return (
                f"{header}\n\n"
                f"Missing Python module: {module}\n"
                f"Install it with:\n    {missing_module_hint(module)}\n\n"
                "The launcher is still running - other tiles are unaffected."
            )
        return (
            f"{header}\n\n"
            f"An optional dependency is missing: {detail}\n"
            f"Install the desktop extras with:\n    {_DEFAULT_HINT}\n\n"
            "The launcher is still running - other tiles are unaffected."
        )

    if isinstance(exc, OSError):
        if "1114" in detail or "initialization routine failed" in detail.lower():
            return f"{header}\n\n" + _DLL_INIT_ADVICE.format(
                package=package_hint or _dll_package_guess(detail)
            )
        if (
            isinstance(exc, FileNotFoundError)
            or "cannot find the file" in detail.lower()
        ):
            return (
                f"{header}\n\n"
                f"A required external program or file was not found:\n    {detail}\n\n"
                "If this tile needs MATLAB/Simulink, install it and make sure "
                "the 'matlab' executable is on your PATH.\n"
                "The launcher is still running - other tiles are unaffected."
            )

    return (
        f"{header}\n\n"
        f"{type(exc).__name__}: {detail}\n\n"
        "Open the console dock (View -> Console) for the full traceback.\n"
        "The launcher is still running - other tiles are unaffected."
    )
