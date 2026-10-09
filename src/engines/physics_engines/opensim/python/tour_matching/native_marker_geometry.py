"""Native OpenSim geometry for explicit calibrated marker bindings (#11903).

This is a position-level provider, not a dynamics/replay qualification adapter.
It keeps the source model intact and admits only achieved independent coordinate
requests. Shared calibration and IK consume its body poses without body aliases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    MarkerPlacement,
)
from src.shared.python.motion_matching.marker_calibration import Array, Pose

Bindings = Mapping[str, tuple[str, Sequence[float]]]


class NativeMarkerGeometry:
    """Evaluate exact native physical frames with a fixed source initialization.

    Coordinate paths select independent native scalar coordinates in SI units.
    Other independent coordinates retain their initialized values; dependent
    coordinates follow native assembly. Every evaluation starts from the saved
    initialized state. Muscle states are not equilibrated or qualified here.
    """

    def __init__(
        self,
        model_path: Path,
        coordinate_paths: Sequence[str],
        *,
        assembly_accuracy: float = 1e-14,
        coordinate_tolerance: float = 1e-7,
    ) -> None:
        import opensim as osim

        paths = tuple(coordinate_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Distinct native coordinate paths are required")
        if not all(isinstance(p, str) and p.startswith("/") for p in paths):
            raise ValueError("Coordinate paths must be absolute native component paths")
        if not all(
            np.isfinite(v) and v > 0 for v in (assembly_accuracy, coordinate_tolerance)
        ):
            raise ValueError(
                "Assembly accuracy and coordinate tolerance must be positive"
            )
        source = Path(model_path).read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.runtime_version = str(osim.GetVersionAndDate())
        self.coordinate_order = paths
        self.assembly_accuracy = float(assembly_accuracy)
        self.coordinate_tolerance = float(coordinate_tolerance)
        self._osim = osim
        self._model = osim.Model(str(model_path))
        self._model.set_assembly_accuracy(self.assembly_accuracy)
        self._model.finalizeConnections()
        self._initial_state = self._model.initSystem()
        self._coordinates = {
            str(c.getAbsolutePathString()): c for c in self._model.getCoordinateSet()
        }
        self._validate_coordinate_selection()
        self._baseline = {
            path: float(c.getValue(self._initial_state))
            for path, c in self._coordinates.items()
        }
        self._state = osim.State(self._initial_state)
        self.loaded_sha256 = hashlib.sha256(
            self._model.dump().encode("utf-8")
        ).hexdigest()
        extension_path = importlib.import_module("opensim._simulation").__file__
        if extension_path is None:
            raise ValueError("Native simulation extension file is unavailable")
        extension = Path(extension_path)
        self.runtime_extension_sha256 = hashlib.sha256(
            extension.read_bytes()
        ).hexdigest()
        self.provider_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        identity = {
            "source": self.source_sha256,
            "loaded": self.loaded_sha256,
            "runtime": self.runtime_version,
            "extension": self.runtime_extension_sha256,
            "provider": self.provider_sha256,
            "coordinates": self.coordinate_order,
            "assembly_accuracy": self.assembly_accuracy,
            "coordinate_tolerance": self.coordinate_tolerance,
            "mode": "native-position-geometry-not-qualified",
        }
        self.identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        # Counts are structural observations; they are not an anatomical assessment.
        self.inventory = {
            "bodies": self._model.getBodySet().getSize(),
            "muscles": self._model.getMuscles().getSize(),
            "forces": self._model.getForceSet().getSize(),
            "constraints": self._model.getConstraintSet().getSize(),
            "markers": self._model.getMarkerSet().getSize(),
            "continuous_states": self._model.getStateVariableNames().getSize(),
        }

    def _validate_coordinate_selection(self) -> None:
        for path in self.coordinate_order:
            coordinate = self._coordinates.get(path)
            if coordinate is None:
                raise ValueError(f"Unknown native coordinate: {path}")
            if (
                coordinate.getLocked(self._initial_state)
                or coordinate.isPrescribed(self._initial_state)
                or coordinate.isDependent(self._initial_state)
            ):
                raise ValueError(f"Requested coordinate is not independent: {path}")

    @property
    def initial_coordinates(self) -> Array:
        """Return a fresh vector of source-initialized selected coordinates."""
        return np.array([self._baseline[path] for path in self.coordinate_order])

    @property
    def coordinate_bounds(self) -> tuple[Array, Array]:
        """Return fresh selected native source ranges in coordinate order."""
        selected = [self._coordinates[path] for path in self.coordinate_order]
        return (
            np.array([c.getRangeMin() for c in selected]),
            np.array([c.getRangeMax() for c in selected]),
        )

    @property
    def achieved_coordinates(self) -> Array:
        """Return actual native assembled values from the latest evaluation."""
        return np.array(
            [
                self._coordinates[path].getValue(self._state)
                for path in self.coordinate_order
            ]
        )

    def _set(self, q: Array) -> None:
        values = np.asarray(q, dtype=float)
        if (
            values.shape != (len(self.coordinate_order),)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Native coordinates require a finite ordered vector")
        self._state = self._osim.State(self._initial_state)
        for path, value in zip(self.coordinate_order, values, strict=True):
            coordinate = self._coordinates[path]
            if not coordinate.getRangeMin() <= value <= coordinate.getRangeMax():
                raise ValueError(f"Coordinate outside source range: {path}")
            coordinate.setValue(self._state, float(value), False)
        self._model.assemble(self._state)
        if not np.allclose(
            self.achieved_coordinates, values, rtol=0, atol=self.coordinate_tolerance
        ):
            raise ValueError("Native assembly did not achieve requested coordinates")
        for path, coordinate in self._coordinates.items():
            achieved = float(coordinate.getValue(self._state))
            if not coordinate.getRangeMin() <= achieved <= coordinate.getRangeMax():
                raise ValueError(f"Assembled coordinate outside source range: {path}")
            if path not in self.coordinate_order and not coordinate.isDependent(
                self._state
            ):
                if abs(achieved - self._baseline[path]) > self.coordinate_tolerance:
                    raise ValueError(
                        "Native assembly changed an unrequested independent coordinate"
                    )
        self._model.realizePosition(self._state)

    def _bindings(self, bindings: Bindings) -> list[tuple[str, Any, MarkerPlacement]]:
        if not bindings:
            raise ValueError("Explicit marker bindings are required")
        osim = self._osim
        result = []
        for label, (path, offset) in bindings.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("A nonempty observed marker label is required")
            if len(offset) != 3:
                raise ValueError("Marker offset must contain three local coordinates")
            placement = MarkerPlacement(path, (offset[0], offset[1], offset[2]))
            if not path.startswith("/") or not self._model.hasComponent(path):
                raise ValueError(f"Unknown absolute native frame path: {path}")
            frame = osim.PhysicalFrame.safeDownCast(self._model.getComponent(path))
            if frame is None or osim.Body.safeDownCast(frame.findBaseFrame()) is None:
                raise ValueError(
                    f"Marker frame must be attached to a native body: {path}"
                )
            result.append((label, frame, placement))
        return result

    def frame_poses(self, bindings: Bindings, q: Array) -> dict[str, Pose]:
        """Return actual world transforms of explicitly bound native frames."""
        entries = self._bindings(bindings)
        self._set(q)
        poses: dict[str, Pose] = {}
        for _, frame, placement in entries:
            transform = frame.getTransformInGround(self._state)
            rotation = np.array(
                [[transform.R().get(i, j) for j in range(3)] for i in range(3)]
            )
            translation = np.array([transform.p().get(i) for i in range(3)])
            poses[placement.body] = rotation, translation
        return poses

    def marker_positions(self, q: Array, bindings: Bindings) -> Array:
        """Evaluate native station positions in declared binding insertion order."""
        entries = self._bindings(bindings)
        self._set(q)
        points = []
        for _, frame, placement in entries:
            point = frame.findStationLocationInGround(
                self._state, self._osim.Vec3(*placement.offset_m)
            )
            points.append([point.get(i) for i in range(3)])
        result = np.asarray(points, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Native marker geometry is nonfinite")
        return result
