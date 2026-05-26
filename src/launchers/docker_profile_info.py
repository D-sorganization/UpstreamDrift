"""Docker profile metadata loader.

Reads ``docker/profiles.yaml`` and resolves each profile's *fully-expanded*
feature list (composing ``extends:`` chains), then enriches it with metadata
from the canonical :mod:`src.shared.python.feature_registry.features` so the
launcher UI can show users **exactly** what each tier installs and how big
the resulting image will be — no more vague "Professional" labels.

Single source of truth: the YAML file *defines* tiers; the feature registry
*describes* features. This module only joins them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from src.shared.python.feature_registry.features import (
    Feature,
    get_feature,
)
from src.shared.python.logging_pkg.logging_config import get_logger

from .startup import REPOS_ROOT

logger = get_logger(__name__)

_PROFILES_PATH = REPOS_ROOT / "docker" / "profiles.yaml"


@dataclass(frozen=True)
class ProfileInfo:
    """A fully-resolved Docker build profile.

    Attributes:
        name: Profile id as it appears in ``profiles.yaml`` (e.g. ``"slim"``).
        description: One-line description from the YAML.
        max_size_mb: Size budget enforced by the CI gate.
        feature_names: Every feature included after walking ``extends:``.
        features: Full :class:`Feature` metadata for each ``feature_name`` that
            resolves through the feature registry. Unknown names are dropped
            with a logged warning — the UI shouldn't crash on a typo.
        approx_total_mb: Sum of ``approx_size_mb`` across resolved features.
            A useful lower-bound hint distinct from ``max_size_mb`` (which is
            the upper budget). Both numbers are rough — they exist to inform
            user choice, not to be exact.
    """

    name: str
    description: str
    max_size_mb: int
    feature_names: tuple[str, ...]
    features: tuple[Feature, ...] = field(default_factory=tuple)
    approx_total_mb: int = 0


def _resolve_features(
    raw_profiles: dict[str, Any], profile_name: str, _seen: set[str] | None = None
) -> list[str]:
    """Walk the ``extends:`` chain and return the union of feature names."""
    if _seen is None:
        _seen = set()
    if profile_name in _seen:
        raise ValueError(f"Profile cycle detected at {profile_name!r}")
    _seen = _seen | {profile_name}

    entry = raw_profiles.get(profile_name)
    if not entry:
        return []

    resolved: list[str] = []
    parent = entry.get("extends")
    if parent:
        for f in _resolve_features(raw_profiles, parent, _seen):
            if f not in resolved:
                resolved.append(f)
    for f in entry.get("features", []) or []:
        if f not in resolved:
            resolved.append(f)
    return resolved


def load_docker_profiles() -> dict[str, ProfileInfo]:
    """Return ``{profile_name: ProfileInfo}`` for every profile in the YAML.

    Falls back to an empty dict (with a warning) if the YAML can't be read.
    The launcher dialog handles that case by showing a generic combobox
    without enrichment, preserving the pre-rich-UI behaviour.
    """
    try:
        text = _PROFILES_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", _PROFILES_PATH, exc)
        return {}

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s: %s", _PROFILES_PATH, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Parsed YAML %s is not a dictionary", _PROFILES_PATH)
        return {}

    raw_profiles: dict[str, Any] = data.get("profiles", {})
    out: dict[str, ProfileInfo] = {}

    for name, entry in raw_profiles.items():
        try:
            feature_names = tuple(_resolve_features(raw_profiles, name))
        except ValueError as exc:
            logger.warning("Skipping profile %r: %s", name, exc)
            continue

        features: list[Feature] = []
        for fname in feature_names:
            try:
                features.append(get_feature(fname))
            except KeyError:
                logger.warning(
                    "Profile %r references unknown feature %r; skipping",
                    name,
                    fname,
                )

        approx_total = sum(f.approx_size_mb for f in features)

        out[name] = ProfileInfo(
            name=name,
            description=str(entry.get("description", "")).strip(),
            max_size_mb=int(entry.get("max_size_mb", 0)),
            feature_names=feature_names,
            features=tuple(features),
            approx_total_mb=approx_total,
        )

    return out


def format_profile_summary(info: ProfileInfo) -> str:
    """Return a multi-line human-readable summary for tooltip / details panel.

    Example output::

        Standard — Default research build — API + MuJoCo + Pinocchio.

        Budget: ≤ 2200 MB   ·   Estimated install: ~450 MB

        Includes 5 features:
          • API server (120 MB)
          • Pendulum models (0 MB)
          • MuJoCo (120 MB)
          • Pinocchio (210 MB)
          • …
    """
    title = info.name.replace("-", " ").title()
    lines = [f"{title} — {info.description}".rstrip(" —")]
    if info.max_size_mb:
        lines.append("")
        lines.append(
            f"Budget: ≤ {info.max_size_mb} MB   ·   "
            f"Estimated install: ~{info.approx_total_mb} MB"
        )
    if info.features:
        lines.append("")
        lines.append(f"Includes {len(info.features)} feature(s):")
        for f in info.features:
            size = f"{f.approx_size_mb} MB" if f.approx_size_mb else "negligible"
            lines.append(f"  • {f.display_name} ({size})")
    elif info.feature_names:
        lines.append("")
        lines.append(f"Features: {', '.join(info.feature_names)}")
    return "\n".join(lines)


__all__ = [
    "ProfileInfo",
    "format_profile_summary",
    "load_docker_profiles",
]
