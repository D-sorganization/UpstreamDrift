"""Attachment-point sidecar parsing for first-party model explorer models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InterfaceFrame:
    """Offset from the attachment link frame to the semantic interface frame."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable frame mapping."""
        return {"xyz": list(self.xyz), "rpy": list(self.rpy)}


@dataclass(frozen=True)
class AttachmentPoint:
    """A declared semantic mount point for model composition."""

    link_name: str
    role: str
    interface_frame: InterfaceFrame = field(default_factory=InterfaceFrame)
    max_payload_kg: float | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable attachment point mapping."""
        data: dict[str, Any] = {
            "link_name": self.link_name,
            "role": self.role,
            "interface_frame": self.interface_frame.to_dict(),
            "tags": list(self.tags),
        }
        if self.max_payload_kg is not None:
            data["max_payload_kg"] = self.max_payload_kg
        return data


@dataclass(frozen=True)
class AttachmentManifest:
    """Parsed attachment sidecar data plus non-fatal validation warnings."""

    schema_version: int = SCHEMA_VERSION
    attachment_points: tuple[AttachmentPoint, ...] = ()
    warnings: tuple[str, ...] = ()
    sidecar_path: Path | None = None

    @property
    def by_link_name(self) -> dict[str, AttachmentPoint]:
        """Map declared link names to their first attachment declaration."""
        return {point.link_name: point for point in self.attachment_points}

    def get(self, link_name: str) -> AttachmentPoint | None:
        """Return the attachment point declared for ``link_name``, if any."""
        if link_name is None:
            raise ValueError("link_name must be provided")
        return self.by_link_name.get(link_name)

    def to_model_info(self) -> dict[str, Any]:
        """Return the manifest shape exposed through loaded model info."""
        return {
            "schema_version": self.schema_version,
            "sidecar_path": str(self.sidecar_path) if self.sidecar_path else None,
            "attachment_points": [point.to_dict() for point in self.attachment_points],
            "warnings": list(self.warnings),
        }


def sidecar_path_for_model(model_path: Path) -> Path:
    """Return the sidecar path for ``<model>.attachments.json``."""
    if model_path is None:
        raise ValueError("model_path must be provided")
    return model_path.with_suffix(".attachments.json")


def load_attachment_manifest(
    model_path: Path,
    *,
    known_links: set[str] | None = None,
) -> AttachmentManifest:
    """Load and validate a model attachment sidecar without raising on bad input."""
    if model_path is None:
        raise ValueError("model_path must be provided")
    sidecar_path = sidecar_path_for_model(model_path)
    if not sidecar_path.exists():
        return AttachmentManifest(sidecar_path=sidecar_path)

    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return AttachmentManifest(
            warnings=(
                f"{sidecar_path.name}: failed to read attachment sidecar: {exc}",
            ),
            sidecar_path=sidecar_path,
        )

    points, warnings = _parse_manifest(raw, known_links=known_links)
    return AttachmentManifest(
        attachment_points=tuple(points),
        warnings=tuple(warnings),
        sidecar_path=sidecar_path,
    )


def attachment_warnings_for_link(
    manifest: AttachmentManifest,
    link_name: str,
    *,
    payload_kg: float | None = None,
) -> tuple[str, ...]:
    """Return declaration and payload warnings for an editor attachment target."""
    if manifest is None:
        raise ValueError("manifest must be provided")
    if link_name is None:
        raise ValueError("link_name must be provided")

    point = manifest.get(link_name)
    if point is None:
        if manifest.attachment_points:
            return (f"Link '{link_name}' is not a declared attachment point.",)
        return ()

    if (
        payload_kg is not None
        and point.max_payload_kg is not None
        and payload_kg > point.max_payload_kg
    ):
        return (
            f"Payload {payload_kg:g} kg exceeds max_payload_kg "
            f"{point.max_payload_kg:g} kg for '{link_name}'.",
        )
    return ()


def _parse_manifest(
    raw: Any,
    *,
    known_links: set[str] | None,
) -> tuple[list[AttachmentPoint], list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return [], ["Attachment sidecar root must be an object."]

    if raw.get("schema_version") != SCHEMA_VERSION:
        warnings.append(f"schema_version must be {SCHEMA_VERSION}.")

    raw_points = raw.get("attachments", [])
    if not isinstance(raw_points, list):
        return [], [*warnings, "attachments must be a list."]

    points: list[AttachmentPoint] = []
    seen_links: set[str] = set()
    for index, raw_point in enumerate(raw_points):
        point = _parse_point(index, raw_point, known_links=known_links)
        warnings.extend(point[1])
        if point[0] is None:
            continue
        if point[0].link_name in seen_links:
            warnings.append(
                f"attachments[{index}].link_name duplicates "
                f"'{point[0].link_name}' and was skipped."
            )
            continue
        seen_links.add(point[0].link_name)
        points.append(point[0])
    return points, warnings


def _parse_point(
    index: int,
    raw_point: Any,
    *,
    known_links: set[str] | None,
) -> tuple[AttachmentPoint | None, list[str]]:
    prefix = f"attachments[{index}]"
    warnings: list[str] = []
    if not isinstance(raw_point, dict):
        return None, [f"{prefix} must be an object."]

    link_name = _required_string(raw_point, "link_name", prefix, warnings)
    role = _required_string(raw_point, "role", prefix, warnings)
    if link_name is None or role is None:
        return None, warnings

    if known_links is not None and link_name not in known_links:
        warnings.append(f"{prefix}.link_name '{link_name}' is not in the model.")

    frame = _parse_frame(raw_point.get("interface_frame", {}), prefix, warnings)
    max_payload = _optional_positive_float(
        raw_point.get("max_payload_kg"), "max_payload_kg", prefix, warnings
    )
    tags = _parse_tags(raw_point.get("tags", []), prefix, warnings)
    return (
        AttachmentPoint(
            link_name=link_name,
            role=role,
            interface_frame=frame,
            max_payload_kg=max_payload,
            tags=tuple(tags),
        ),
        warnings,
    )


def _required_string(
    raw_point: dict[str, Any],
    key: str,
    prefix: str,
    warnings: list[str],
) -> str | None:
    value = raw_point.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    warnings.append(f"{prefix}.{key} must be a non-empty string.")
    return None


def _parse_frame(raw_frame: Any, prefix: str, warnings: list[str]) -> InterfaceFrame:
    if not isinstance(raw_frame, dict):
        warnings.append(f"{prefix}.interface_frame must be an object.")
        return InterfaceFrame()
    return InterfaceFrame(
        xyz=_triple(
            raw_frame.get("xyz", [0, 0, 0]), f"{prefix}.interface_frame.xyz", warnings
        ),
        rpy=_triple(
            raw_frame.get("rpy", [0, 0, 0]), f"{prefix}.interface_frame.rpy", warnings
        ),
    )


def _triple(
    raw_values: Any, label: str, warnings: list[str]
) -> tuple[float, float, float]:
    if not isinstance(raw_values, list | tuple) or len(raw_values) != 3:
        warnings.append(f"{label} must contain exactly three numbers.")
        return (0.0, 0.0, 0.0)
    try:
        return (float(raw_values[0]), float(raw_values[1]), float(raw_values[2]))
    except (TypeError, ValueError):
        warnings.append(f"{label} must contain only numbers.")
        return (0.0, 0.0, 0.0)


def _optional_positive_float(
    raw_value: Any,
    key: str,
    prefix: str,
    warnings: list[str],
) -> float | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        warnings.append(f"{prefix}.{key} must be a positive number.")
        return None
    if value < 0:
        warnings.append(f"{prefix}.{key} must be a positive number.")
        return None
    return value


def _parse_tags(raw_tags: Any, prefix: str, warnings: list[str]) -> list[str]:
    if not isinstance(raw_tags, list):
        warnings.append(f"{prefix}.tags must be a list of strings.")
        return []
    tags = [tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip()]
    if len(tags) != len(raw_tags):
        warnings.append(f"{prefix}.tags ignored non-string or empty entries.")
    return tags
