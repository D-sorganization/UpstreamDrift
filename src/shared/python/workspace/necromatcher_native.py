"""One compiled, hash-bound research model shared by review and native fitting.

Native coordinate units are checked independently of producer declarations.
This resource check does not certify source timing, anatomy or physical replay.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.historical_fit import CameraProjection
from src.shared.python.motion_matching.pipeline.plant import (
    MatchingPlant,
    ScalarCoordinateUnits,
    get_plant,
)

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

from .necromatcher_efforts import AuthoredEffortProfile, effort_units_for_coordinates
from .necromatcher_ranges import AuthoredCoordinateBounds, extract_authored_bounds


@dataclass(frozen=True)
class NativeModelBinding:
    """Exact registered geometry verified independently of a stored trajectory."""

    model_id: str
    model_hash: str
    definition_bytes: bytes
    plant: MatchingPlant
    coordinate_units: tuple[str, ...]


def load_native_model_binding(
    library: NecromatcherLibrary,
    model_id: str,
    definition_bytes: bytes,
    coordinate_units: tuple[str, ...],
) -> NativeModelBinding:
    """Verify stored XML, declared order and compiled scalar units before use."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    model = library.load_asset(model_id)
    if model.kind != "native_model" or model.metadata["engine"] != "mujoco":
        raise ValueError("Native binding currently requires a MuJoCo full-body model")
    if not isinstance(definition_bytes, bytes) or not definition_bytes:
        raise ValueError("Native model requires exact definition bytes")
    try:
        definition = json.loads(definition_bytes)
        order = tuple(definition["coordinate_order"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Native model definition requires coordinate order") from exc
    if order != tuple(model.metadata["dofs"]):
        raise ValueError("Native model coordinate order differs from registered model")
    xml, _ = export_full_body_mjcf(definition_bytes)
    model_hash = model.metadata["hash"]
    if "sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest() != model_hash:
        raise ValueError("Definition does not reproduce the bound native model hash")
    native = get_plant("mujoco", definition_bytes)
    if tuple(native.coordinate_order) != order:
        raise ValueError("Rebuilt native model coordinate order differs from model")
    if not isinstance(native, ScalarCoordinateUnits):
        raise ValueError("Native model does not report compiled coordinate units")
    units = native.coordinate_units
    if units != tuple(coordinate_units):
        raise ValueError(
            "Declared coordinate units differ from the compiled native model"
        )
    return NativeModelBinding(model_id, model_hash, definition_bytes, native, units)


@dataclass(frozen=True)
class NativeFitBinding:
    """A verified native resource; all saved motion remains a research hypothesis."""

    fit_id: str
    fit_hash: str
    model_id: str
    model_hash: str
    fit: dict[str, Any]
    plant: MatchingPlant
    coordinate_units: tuple[str, ...]
    definition_bytes: bytes = b""

    def authored_coordinate_bounds(self) -> AuthoredCoordinateBounds:
        """Extract authored hypotheses from the captured exact bound definition."""
        if (
            not isinstance(self.plant, ScalarCoordinateUnits)
            or self.plant.coordinate_units != self.coordinate_units
        ):
            raise ValueError("Authored range units differ from compiled native units")
        return extract_authored_bounds(
            self.definition_bytes,
            self.model_hash,
            tuple(self.plant.coordinate_order),
            self.plant.coordinate_units,
        )

    def efforts(
        self, controls: AuthoredEffortProfile, time_s: float
    ) -> dict[str, float]:
        """Map checked authored forces/torques to this native resource's DOFs."""
        if (
            controls.model_id != self.model_id
            or controls.model_hash != self.model_hash
            or controls.fit_id != self.fit_id
            or controls.fit_hash != self.fit_hash
            or controls.dofs != tuple(self.plant.coordinate_order)
            or controls.coordinate_units != self.coordinate_units
            or controls.effort_units
            != effort_units_for_coordinates(self.coordinate_units)
        ):
            raise ValueError("Controls do not match this native model, fit and units")
        return dict(
            zip(controls.dofs, map(float, controls.evaluate(time_s)), strict=True)
        )

    def project(self, frame_index: int) -> dict[str, Any]:
        """Project one exact saved source frame through this compiled model."""
        if type(frame_index) is not int:
            raise IndexError("Source frame index must be an integer")
        try:
            position = self.fit["frame_indices"].index(frame_index)
        except ValueError as exc:
            raise IndexError("Source frame has no stored fit sample") from exc
        camera, attachments = self.review_inputs()
        pixels = camera.project(
            self.plant.marker_positions(
                np.asarray(self.fit["q"][position]), attachments
            )
        )
        if not np.isfinite(pixels).all():
            raise ValueError("Native model projection must be finite")
        return {
            "fit_id": self.fit_id,
            "capture_id": self.fit["capture_id"],
            "frame_index": frame_index,
            "frame": self.fit["frames"][position],
            "coordinates": "image_pixels",
            "qualification": "monocular_research_hypothesis",
            "camera_qualified": False,
            "physical_time_qualified": False,
            "points": {
                name: {"x": float(point[0]), "y": float(point[1]), "visibility": None}
                for name, point in zip(attachments, pixels, strict=True)
            },
        }

    def review_inputs(self) -> tuple[CameraProjection, dict[str, Any]]:
        """Validate camera and attachments without recompiling the native model."""
        try:
            original = self.fit["evidence"]["original_fit"]
            camera_record, attachments = original["camera"], original["attachments"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Fit lacks native definition, camera or attachment provenance"
            ) from exc
        if not isinstance(attachments, dict) or not attachments:
            raise ValueError("Fit requires named native marker attachments")
        if not isinstance(camera_record, dict) or set(camera_record) != {
            item.name for item in fields(CameraProjection)
        }:
            raise ValueError("Camera record must match the camera projection fields")
        for record in attachments.values():
            if not isinstance(record, (list, tuple)) or len(record) != 2:
                raise ValueError(
                    "Native marker attachments require a body and finite 3-vector"
                )
            body, offset = record
            try:
                values = np.asarray(offset, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Native marker attachments require a finite 3-vector"
                ) from exc
            if (
                not isinstance(body, str)
                or values.shape != (3,)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "Native marker attachments require a body and finite 3-vector"
                )
        return CameraProjection(**camera_record), attachments


def load_native_fit_binding(
    library: NecromatcherLibrary, fit_id: str
) -> NativeFitBinding:
    """Compile exact saved geometry and reject declared/native unit disagreement."""
    fit = library.load_fit(fit_id)
    try:
        definition = fit["provenance"]["native_definition"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Fit lacks native definition, camera or attachment provenance"
        ) from exc
    model_bytes = json.dumps(definition, allow_nan=False).encode("utf-8")
    model = load_native_model_binding(
        library, fit["model_id"], model_bytes, tuple(fit["coordinate_units"])
    )
    if tuple(model.plant.coordinate_order) != tuple(fit["coordinate_order"]):
        raise ValueError("Rebuilt native model coordinate order differs from fit")
    return NativeFitBinding(
        fit_id,
        library.load_asset(fit_id).metadata["hash"],
        fit["model_id"],
        fit["model_hash"],
        fit,
        model.plant,
        model.coordinate_units,
        model_bytes,
    )
