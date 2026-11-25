"""Kinetic match as an Agent API capability, shared by REST and MCP (#15150 D10).

The request is ``{"bundle": str, "out": str, "shaft_source"?: str}``:
``bundle`` is a saved kinematic bundle below the input roots, ``out`` the
output bundle (``.npz``) below the output root and ``shaft_source`` an optional
override of the settings' value (one of ``SHAFT_SOURCES``). The solve is
:func:`~src.shared.python.motion_matching.kinetic_match.run_kinetic_match`,
the same function the command line calls; this module only checks the request
and shapes the response (``RESPONSE_FIELDS``).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from .remote_paths import contained_below, resolve_roots

_SETTINGS_PATH = Path(__file__).with_name("kinetic_match_capability_settings.json")
REQUEST_FIELDS = ("bundle", "out", "shaft_source")
#: Fields of a succeeded response (the capability record's delivery value).
RESPONSE_FIELDS: tuple[str, ...] = (
    "bundle",
    "out_bundle",
    "receipt_path",
    "method",
    "verdict",
)
RESPONSE_FIELDS += ("reasons", "peaks", "receipt")


@cache
def load_capability_settings() -> dict[str, Any]:
    """The capability settings document (roots and bundle suffix)."""
    return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))


def validate_request(request: Any) -> dict[str, Any]:
    """Checked request: resolved ``bundle`` and ``out`` paths and ``shaft_source``.

    Raises:
        TypeError: the request is not an object, or a path is not a string.
        ValueError: a field is unknown or missing, a path leaves its root, the
            bundle does not exist, the output is not a bundle file, or
            ``shaft_source`` is not one of ``SHAFT_SOURCES``.
    """
    from src.shared.python.motion_matching.kinetic_match import SHAFT_SOURCES

    if not isinstance(request, dict):
        raise TypeError("kinetic match request must be an object")
    unknown = sorted(set(request) - set(REQUEST_FIELDS))
    if unknown:
        raise ValueError(f"unknown kinetic match request fields {unknown}")
    for name in ("bundle", "out"):
        if name not in request:
            raise ValueError(f"kinetic match request needs {name!r}")
    settings = load_capability_settings()
    inputs, out_root = resolve_roots(settings["roots"])
    bundle = contained_below(request["bundle"], inputs, "input roots")
    out = contained_below(request["out"], (out_root,), "output root")
    if not bundle.is_file():
        raise ValueError(f"bundle {request['bundle']!r} does not exist")
    if out.suffix != settings["bundle_suffix"]:
        raise ValueError(f"out must be a {settings['bundle_suffix']} bundle")
    shaft_source = request.get("shaft_source")
    if shaft_source is not None and shaft_source not in SHAFT_SOURCES:
        raise ValueError(f"shaft_source must be one of {SHAFT_SOURCES}")
    return {"bundle": bundle, "out": out, "shaft_source": shaft_source}


def run_kinetic_match_capability(request: dict[str, Any] | None) -> dict[str, Any]:
    """Validate ``request``, run the shared kinetic match and shape the response.

    Postcondition: the result holds every name in ``RESPONSE_FIELDS``; the
    output directory exists.
    """
    from src.shared.python.motion_matching import kinetic_match

    checked = validate_request(request if request is not None else {})
    out: Path = checked["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    result = kinetic_match.run_kinetic_match(
        checked["bundle"], out, shaft_source=checked["shaft_source"]
    )
    receipt = result["receipt"]
    return {
        "bundle": str(checked["bundle"]),
        "out_bundle": str(result["out"]),
        "receipt_path": str(result["receipt_path"]),
        "method": receipt["method"],
        "verdict": receipt["verdict"],
        "reasons": receipt["reasons"],
        "peaks": receipt["peaks"],
        "receipt": receipt,
    }
