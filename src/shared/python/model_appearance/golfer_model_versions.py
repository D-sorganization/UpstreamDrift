"""Which golfer model is current, and which parts of it are deprecated (#15648).

``src/config/golfer_model_versions.json`` is the single answer to "which hands
and body should this golfer have". :func:`current_golfer_model` reads the
current model; :func:`deprecated_parts` lists what a spec still resolves to
that the registry deprecates; :func:`warn_if_deprecated` logs that once per
part and process, so a run on an old spec says so without flooding the log.

This module reads JSON only (no appearance imports), so the hand and body
resolvers can call it without an import cycle.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETTINGS_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "config"
    / "golfer_model_versions.json"
)
#: Values each selector field accepts, and the value a spec without it gets.
_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "hand_visual_backend": (("capsule", "skinned_human"), "capsule"),
    "body_visual_backend": (("loft", "skinned_cc0"), "loft"),
}
_TEXT_KEYS = ("version", "spec_revision_configuration", "make_with")
#: Shaft models the registry knows (#15684); the current model must bend.
SHAFT_MODELS = ("rigid", "chain")


@dataclass(frozen=True)
class CurrentGolferModel:
    """The hands, body and spec revision the main golfer model uses."""

    version: str
    hand_visual_backend: str
    body_visual_backend: str
    spec_revision_configuration: str
    make_with: str
    refit_poses_with: str
    docs: str
    shaft_model: str = "chain"
    shaft_flex: str = "Stiff"


@dataclass(frozen=True)
class DeprecatedValue:
    """A selector value the registry deprecates, with its replacement."""

    field: str
    value: str
    replacement: str
    reason: str
    still_allowed_for: str


@dataclass(frozen=True)
class GolferModelVersions:
    """The parsed registry. Construct with :func:`load_golfer_model_versions`."""

    current: CurrentGolferModel
    deprecated: tuple[DeprecatedValue, ...]


def _known(field: str, value: str) -> str:
    if field not in _FIELDS or value not in _FIELDS[field][0]:
        raise ValueError(f"unknown {field} {value!r} in the golfer model registry")
    return value


def _current_shaft(shaft: Mapping[str, Any]) -> dict[str, str]:
    """The current model's shaft fields (#15684).

    Raises:
        ValueError: an unknown shaft model, a rigid current shaft (rigid is
            only the ballpark fallback) or an empty flex rating.
    """
    model, flex = str(shaft["model"]), str(shaft["flex"])
    if model not in SHAFT_MODELS:
        raise ValueError(f"unknown shaft model {model!r} in the golfer model registry")
    if model == "rigid":
        raise ValueError("the current golfer model cannot use a rigid shaft")
    if not flex:
        raise ValueError("the current golfer model's shaft needs a flex rating")
    return {"shaft_model": model, "shaft_flex": flex}


def load_golfer_model_versions(path: Path | None = None) -> GolferModelVersions:
    """Parse and validate the registry.

    Raises:
        ValueError: an unknown backend, a deprecated current backend, an
            empty text field, or a deprecated value without a replacement.
    """
    raw: dict[str, Any] = json.loads(
        (path or SETTINGS_PATH).read_text(encoding="utf-8")
    )
    cur = raw["current"]
    current = CurrentGolferModel(
        version=str(cur["version"]),
        hand_visual_backend=_known(
            "hand_visual_backend", str(cur["hand_visual_backend"])
        ),
        body_visual_backend=_known(
            "body_visual_backend", str(cur["body_visual_backend"])
        ),
        spec_revision_configuration=str(cur["spec_revision_configuration"]),
        make_with=str(cur["make_with"]),
        refit_poses_with=str(cur["refit_poses_with"]),
        docs=str(cur["docs"]),
        **_current_shaft(cur["shaft"]),
    )
    if not all(getattr(current, key) for key in _TEXT_KEYS):
        raise ValueError("the current golfer model needs a version, revision and maker")
    deprecated = tuple(
        DeprecatedValue(
            field=str(item["field"]),
            value=_known(str(item["field"]), str(item["value"])),
            replacement=_known(str(item["field"]), str(item["replacement"])),
            reason=str(item["reason"]),
            still_allowed_for=str(item["still_allowed_for"]),
        )
        for item in raw["deprecated"]
    )
    for item in deprecated:
        if getattr(current, item.field) == item.value:
            raise ValueError(f"current {item.field} {item.value!r} is deprecated")
    return GolferModelVersions(current, deprecated)


@cache
def _registry() -> GolferModelVersions:
    return load_golfer_model_versions()


def current_golfer_model() -> CurrentGolferModel:
    """The current golfer model from the committed registry."""
    return _registry().current


def deprecated_values() -> tuple[DeprecatedValue, ...]:
    """Every deprecated selector value in the committed registry."""
    return _registry().deprecated


def deprecated_parts(spec: Mapping[str, Any]) -> tuple[DeprecatedValue, ...]:
    """Deprecated values ``spec`` resolves to (a missing selector's default too).

    Postcondition: empty for a spec on the current model.
    """
    subject = spec.get("subject", {})
    subject = subject if isinstance(subject, Mapping) else {}
    resolved = {
        field: str(subject.get(field, default))
        for field, (_, default) in _FIELDS.items()
    }
    return tuple(
        item for item in deprecated_values() if resolved[item.field] == item.value
    )


@cache
def _warn_once(field: str, value: str, replacement: str, allowed: str) -> None:
    logger.warning(
        "golfer model uses deprecated %s %r; the current model uses %r "
        "(src/config/golfer_model_versions.json). Still allowed for: %s",
        field,
        value,
        replacement,
        allowed,
    )


def warn_if_deprecated(spec: Mapping[str, Any], field: str) -> None:
    """Log (once per value and process) if ``spec``'s ``field`` is deprecated."""
    for item in deprecated_parts(spec):
        if item.field == field:
            _warn_once(item.field, item.value, item.replacement, item.still_allowed_for)


def reset_deprecation_warnings() -> None:
    """Forget which deprecations were logged (tests, long-lived hosts)."""
    _warn_once.cache_clear()
