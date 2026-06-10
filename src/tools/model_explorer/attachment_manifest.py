"""Attachment-point sidecar manifests for model composition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logger_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttachmentInterfaceFrame:
    """Pose of the attaching interface relative to the declared link."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        return {"xyz": list(self.xyz), "rpy": list(self.rpy)}


@dataclass(frozen=True)
class AttachmentPoint:
    """One declared semantic mount point for a model."""

    name: str
    link_name: str
    role: str
    interface_frame: AttachmentInterfaceFrame = AttachmentInterfaceFrame()
    max_payload_kg: float | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "link_name": self.link_name,
            "role": self.role,
            "interface_frame": self.interface_frame.to_dict(),
            "tags": list(self.tags),
        }
        if self.max_payload_kg is not None:
            payload["max_payload_kg"] = self.max_payload_kg
        return payload


@dataclass(frozen=True)
class AttachmentManifestResult:
    """Parsed sidecar state plus non-fatal validation warnings."""

    path: Path
    attachment_points: tuple[AttachmentPoint, ...]
    warnings: tuple[str, ...] = ()

    def points_as_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(point.to_dict() for point in self.attachment_points)


def attachment_sidecar_path(model_path: Path | str) -> Path:
    """Return the sidecar path for a model file."""
    if model_path is None:
        raise ValueError("model_path must be provided")
    return Path(model_path).with_suffix(".attachments.json")


def load_attachment_manifest(model_path: Path | str) -> AttachmentManifestResult:
    """Load and validate a model's attachment sidecar.

    Missing or malformed sidecars are non-fatal. The caller receives any
    warnings and an empty point set so model discovery can continue.
    """
    sidecar = attachment_sidecar_path(model_path)
    if not sidecar.exists():
        return AttachmentManifestResult(path=sidecar, attachment_points=())

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _warn_result(sidecar, f"invalid JSON: {exc.msg}")
    except OSError as exc:
        return _warn_result(sidecar, f"could not read manifest: {exc}")

    points, warnings = _parse_manifest(raw)
    for warning in warnings:
        logger.warning("Attachment manifest %s: %s", sidecar, warning)
    return AttachmentManifestResult(
        path=sidecar,
        attachment_points=tuple(points),
        warnings=tuple(warnings),
    )


def _warn_result(path: Path, warning: str) -> AttachmentManifestResult:
    logger.warning("Attachment manifest %s: %s", path, warning)
    return AttachmentManifestResult(
        path=path, attachment_points=(), warnings=(warning,)
    )


def _parse_manifest(raw: Any) -> tuple[list[AttachmentPoint], list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return [], ["manifest root must be an object"]

    version = raw.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        warnings.append(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}; got {version!r}"
        )

    entries = raw.get("attachment_points")
    if not isinstance(entries, list):
        return [], warnings + ["attachment_points must be a list"]

    points: list[AttachmentPoint] = []
    for index, entry in enumerate(entries):
        point, entry_warnings = _parse_point(entry, f"attachment_points[{index}]")
        warnings.extend(entry_warnings)
        if point is not None:
            points.append(point)
    return points, warnings


def _parse_point(raw: Any, path: str) -> tuple[AttachmentPoint | None, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"{path} must be an object"]

    name = _required_text(raw, "name", f"{path}.name", warnings)
    link_name = _required_text(raw, "link_name", f"{path}.link_name", warnings)
    role = _required_text(raw, "role", f"{path}.role", warnings)
    if not name or not link_name or not role:
        return None, warnings

    frame = _parse_frame(
        raw.get("interface_frame", {}), f"{path}.interface_frame", warnings
    )
    max_payload_kg = _optional_positive_float(
        raw.get("max_payload_kg"),
        f"{path}.max_payload_kg",
        warnings,
    )
    tags = _parse_tags(raw.get("tags", ()), f"{path}.tags", warnings)

    return (
        AttachmentPoint(
            name=name,
            link_name=link_name,
            role=role,
            interface_frame=frame,
            max_payload_kg=max_payload_kg,
            tags=tags,
        ),
        warnings,
    )


def _required_text(
    raw: dict[str, Any],
    key: str,
    path: str,
    warnings: list[str],
) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        warnings.append(f"{path} must be a non-empty string")
        return None
    return value.strip()


def _parse_frame(
    raw: Any,
    path: str,
    warnings: list[str],
) -> AttachmentInterfaceFrame:
    if raw in (None, {}):
        return AttachmentInterfaceFrame()
    if not isinstance(raw, dict):
        warnings.append(f"{path} must be an object")
        return AttachmentInterfaceFrame()
    return AttachmentInterfaceFrame(
        xyz=_float_triplet(raw.get("xyz", (0.0, 0.0, 0.0)), f"{path}.xyz", warnings),
        rpy=_float_triplet(raw.get("rpy", (0.0, 0.0, 0.0)), f"{path}.rpy", warnings),
    )


def _float_triplet(
    raw: Any, path: str, warnings: list[str]
) -> tuple[float, float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        warnings.append(f"{path} must be a three-number array")
        return (0.0, 0.0, 0.0)
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        warnings.append(f"{path} must contain only numbers")
        return (0.0, 0.0, 0.0)


def _optional_positive_float(
    raw: Any,
    path: str,
    warnings: list[str],
) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        warnings.append(f"{path} must be a positive number")
        return None
    if value <= 0:
        warnings.append(f"{path} must be greater than zero")
        return None
    return value


def _parse_tags(raw: Any, path: str, warnings: list[str]) -> tuple[str, ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list):
        warnings.append(f"{path} must be a list of strings")
        return ()
    tags: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            warnings.append(f"{path}[{index}] must be a non-empty string")
            continue
        tags.append(value.strip())
    return tuple(tags)
