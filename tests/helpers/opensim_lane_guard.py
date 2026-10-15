"""Decision logic for the OpenSim lane guard (#12752).

In the OpenSim CI lane (``UD_REQUIRE_OPENSIM=1``) a spine test that is skipped
only because ``opensim`` cannot be imported must fail, so an absent SDK can
never be reported as a green run. The functions here are pure; the pytest
hooks that apply them live in ``tests/opensim/spine/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tests.helpers.opensim_support import SKIP_REASON

__all__ = [
    "REQUIRE_ENV_VAR",
    "failure_message",
    "is_opensim_absence_skip",
    "is_within_directory",
    "opensim_lane_required",
    "should_fail_skip",
]

REQUIRE_ENV_VAR = "UD_REQUIRE_OPENSIM"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_IMPORTORSKIP_ABSENCE = "could not import 'opensim'"


def opensim_lane_required(environ: Mapping[str, str]) -> bool:
    """Return True when ``UD_REQUIRE_OPENSIM`` marks this run as the OpenSim lane.

    Precondition: ``environ`` is a mapping (``TypeError`` otherwise).
    Postcondition: the result is a plain ``bool``; unset or non-truthy values
    give False.
    """
    if not isinstance(environ, Mapping):
        raise TypeError(f"environ must be a mapping, got {type(environ).__name__}")
    value = str(environ.get(REQUIRE_ENV_VAR, "")).strip().lower()
    return value in _TRUTHY


def is_opensim_absence_skip(reason: str) -> bool:
    """Return True when a skip reason says the OpenSim SDK is not importable.

    Recognises both the ``require_opensim`` skip text and the message
    ``pytest.importorskip`` produces for a missing module.

    Precondition: ``reason`` is a string (``TypeError`` otherwise).
    """
    if not isinstance(reason, str):
        raise TypeError(f"reason must be a string, got {type(reason).__name__}")
    return SKIP_REASON in reason or _IMPORTORSKIP_ABSENCE in reason


def should_fail_skip(reason: str, environ: Mapping[str, str]) -> bool:
    """Return True when a skip must be reported as a failure in the OpenSim lane.

    Postcondition: True only if the lane is required AND the skip is an
    OpenSim absence. Boundary and asset skips are never converted.
    """
    return opensim_lane_required(environ) and is_opensim_absence_skip(reason)


def failure_message(reason: str) -> str:
    """Return the failure text that replaces an absent-OpenSim skip.

    Precondition: ``reason`` is a string (``TypeError`` otherwise).
    """
    if not isinstance(reason, str):
        raise TypeError(f"reason must be a string, got {type(reason).__name__}")
    return (
        f"{REQUIRE_ENV_VAR}=1 (OpenSim lane): a spine test was skipped because "
        f"OpenSim is not importable. Skips are not acceptance in this lane; "
        f"install the OpenSim bindings. Original skip: {reason}"
    )


def is_within_directory(path: Path | str, root: Path | str) -> bool:
    """Return True when ``path`` lies inside ``root`` (both resolved).

    A sibling that merely shares a name prefix (``spine_other`` vs ``spine``)
    is not inside.
    """
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents
