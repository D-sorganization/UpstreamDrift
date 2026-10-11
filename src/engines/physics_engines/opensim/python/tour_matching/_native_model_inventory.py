"""Native inheritance and state observations for the asset admission boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _component_fact(component: Any) -> dict[str, Any]:
    return {
        "path": component.getAbsolutePathString(),
        "class": component.getConcreteClassName(),
    }


def _muscle_fact(osim: Any, muscle: Any) -> dict[str, Any]:
    row = _component_fact(muscle)
    row.update(
        {
            "ignore_tendon_compliance": bool(muscle.get_ignore_tendon_compliance()),
            "ignore_activation_dynamics": bool(muscle.get_ignore_activation_dynamics()),
        }
    )
    geometry = osim.GeometryPath.safeDownCast(muscle.getPath())
    if geometry is None:
        row.update(
            {
                "path_status": "unavailable-nongeometric-path",
                "path_frames": None,
                "path_base_frames": None,
            }
        )
    else:
        points = geometry.getPathPointSet()
        frames, bases = [], []
        for i in range(points.getSize()):
            point = points.get(i)
            frame = point.getParentFrame()
            base = frame.findBaseFrame()
            frames.append(frame.getAbsolutePathString())
            bases.append(base.getAbsolutePathString())
        row.update(
            {
                "path_status": "observed-declared-path-points",
                "path_frames": tuple(frames),
                "path_base_frames": tuple(bases),
            }
        )
    return row


def _regions(
    regions: Mapping[str, tuple[str, ...]],
    frame_paths: set[str],
    muscles: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for region, paths in sorted(regions.items()):
        unknown = set(paths) - frame_paths
        if unknown:
            raise ValueError(f"unknown region frame: {sorted(unknown)}")
        attached, unavailable = [], []
        for muscle in muscles:
            if muscle["path_frames"] is None:
                unavailable.append(muscle["path"])
            elif set(paths).intersection(
                (*muscle["path_frames"], *muscle["path_base_frames"])
            ):
                attached.append(muscle["path"])
        rows.append(
            {
                "region": region,
                "declared_frames": paths,
                "attached_muscles": tuple(attached),
                "unavailable_muscle_paths": tuple(unavailable),
                "anatomy_status": "unverified",
                "interpretation": "attachment membership only; joint crossing and capacity unverified",
            }
        )
    return tuple(rows)


def _coordinate_facts(
    coordinates: list[Any],
    state: Any,
    roles: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, Any], ...]:
    actual = {coordinate.getAbsolutePathString() for coordinate in coordinates}
    declared = {path for paths in roles.values() for path in paths}
    if declared - actual:
        raise ValueError(f"unknown coordinate role path: {sorted(declared - actual)}")
    rows = []
    for coordinate in coordinates:
        row = _component_fact(coordinate)
        row.update(
            {
                "locked": bool(coordinate.getLocked(state)),
                "prescribed": bool(coordinate.isPrescribed(state)),
                "default_value": float(coordinate.getDefaultValue()),
                "declared_roles": tuple(
                    sorted(
                        role for role, paths in roles.items() if row["path"] in paths
                    )
                ),
            }
        )
        rows.append(row)
    return tuple(rows)


def observe_native_model(
    osim: Any,
    model: Any,
    state: Any,
    regions: Mapping[str, tuple[str, ...]],
    roles: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Inventory recursive native components after successful initialization."""
    components = list(model.getComponentsList())
    components.sort(key=lambda component: component.getAbsolutePathString())
    muscles, coordinates, frames = [], [], set()
    classified: dict[str, list[dict[str, Any]]] = {
        "forces": [],
        "nonmuscle_actuators": [],
        "controllers": [],
        "constraints": [],
        "contact_geometry": [],
        "bodies": [],
    }
    native_types = {
        "forces": osim.Force,
        "controllers": osim.Controller,
        "constraints": osim.Constraint,
        "contact_geometry": osim.ContactGeometry,
        "bodies": osim.Body,
    }
    for component in components:
        for group, native_type in native_types.items():
            if native_type.safeDownCast(component) is not None:
                classified[group].append(_component_fact(component))
        muscle = osim.Muscle.safeDownCast(component)
        if muscle is not None:
            muscles.append(_muscle_fact(osim, muscle))
        elif osim.Actuator.safeDownCast(component) is not None:
            classified["nonmuscle_actuators"].append(
                {**_component_fact(component), "role_status": "unverified"}
            )
        coordinate = osim.Coordinate.safeDownCast(component)
        if coordinate is not None:
            coordinates.append(coordinate)
        if osim.PhysicalFrame.safeDownCast(component) is not None:
            frames.add(component.getAbsolutePathString())
    names = model.getStateVariableNames()
    state_names = tuple(names.get(i) for i in range(names.getSize()))
    registry = model.getMuscles()
    registered = tuple(
        sorted(
            registry.get(i).getAbsolutePathString() for i in range(registry.getSize())
        )
    )
    unregistered = tuple(sorted({row["path"] for row in muscles} - set(registered)))
    serialized_digest = hashlib.sha256(model.dump().encode("utf-8")).hexdigest()
    version = str(osim.GetVersion())
    identity = {
        "runtime_version": version,
        "native_serialized_sha256": serialized_digest,
        "muscles": muscles,
        "state_names": state_names,
        "registered_muscle_paths": registered,
        "coordinates": _coordinate_facts(coordinates, state, {}),
    }
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {
        **classified,
        "model_name": model.getName(),
        "runtime_version": version,
        "native_serialized_sha256": serialized_digest,
        "loaded_identity_sha256": identity_digest,
        "components": tuple(_component_fact(c) for c in components),
        "muscles": tuple(muscles),
        "registered_muscle_paths": registered,
        "unregistered_muscle_paths": unregistered,
        "state_names": state_names,
        "coordinates": _coordinate_facts(coordinates, state, roles),
        "regions": _regions(regions, frames, muscles),
        "num_q": int(state.getNQ()),
        "num_u": int(state.getNU()),
        "independent_dof_status": "unverified-constraints-not-reduced",
        "initial_state_status": "native-defaults-not-equilibrated-or-replay-qualified",
    }
