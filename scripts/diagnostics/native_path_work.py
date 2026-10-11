"""Original single-pin moving-path force/length diagnostic for F07 (#11856).

Run with native OpenSim: ``python -m scripts.diagnostics.native_path_work
--output DIRECTORY``. Writes original fixture XML and an unqualified receipt.
No donor data, physiological parameters, runtime patches, or fitted controls.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

from src.engines.physics_engines.opensim.python.tour_matching import (
    MomentArmDerivativeMismatchError,
    compute_path_length_finite_difference_moment_arm,
    validate_moment_arm_consistency,
)

SOURCE_REVISION = "85aaf6450a2f22457dac4d1ab35adfed9d3a8e43"
SOURCE_HASHES = {
    "OpenSim/Simulation/Model/GeometryPath.cpp": "dbb13642bb2db9a46e6cc02653692f20e414d1a69f521ea106ecfd5bd2ccced5",
    "OpenSim/Simulation/MomentArmSolver.cpp": "42b6614770f04354b597fecfa8c166a6c4b3e2d9a72cd780682bbad1a0c40ad3",
}
INERTIA_KG_M2 = 0.02
STEPS_RAD = (1e-3, 1e-4, 1e-5)
TOLERANCE_M = 1e-7


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _build_fixture(osim: Any, moving: bool) -> Any:
    model = osim.Model()
    model.setName("original_moving_path" if moving else "original_fixed_path")
    model.setGravity(osim.Vec3(0))
    body = osim.Body("rotor", 1, osim.Vec3(0), osim.Inertia(INERTIA_KG_M2))
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    coordinate = joint.updCoordinate()
    coordinate.setName("angle")
    coordinate.setRangeMin(-1)
    coordinate.setRangeMax(1)
    model.addBody(body)
    model.addJoint(joint)
    cable = osim.PathActuator()
    cable.setName("cable")
    cable.setOptimalForce(1.0)
    cable.addNewPathPoint("origin", model.getGround(), osim.Vec3(-0.2, 0.2, 0))
    if moving:
        point = osim.MovingPathPoint()
        point.setName("via")
        point.setParentFrame(body)
        point.set_x_location(osim.LinearFunction(0.03, 0.1))
        point.set_y_location(osim.Constant(0.05))
        point.set_z_location(osim.Constant(0))
        point.connectSocket_x_coordinate(coordinate)
        point.connectSocket_y_coordinate(coordinate)
        point.connectSocket_z_coordinate(coordinate)
        cable.updGeometryPath().updPathPointSet().adoptAndAppend(point)
    else:
        cable.addNewPathPoint("via", body, osim.Vec3(0.1, 0.05, 0))
    cable.addNewPathPoint("insertion", body, osim.Vec3(0.2, 0.05, 0))
    model.addForce(cable)
    model.finalizeConnections()
    return model


def _observe_sample(
    osim: Any, model: Any, initial: Any, q: float, step: float
) -> dict[str, Any]:
    coordinate = model.updCoordinateSet().get("angle")
    cable = osim.PathActuator.safeDownCast(model.updForceSet().get("cable"))
    path = cable.getGeometryPath()
    points = path.getPathPointSet()
    via, insertion = points.get(1), points.get(2)
    achieved: dict[float, float] = {}

    def length_at(value: float, segment: bool = False) -> float:
        sample = osim.State(initial)
        coordinate.setValue(sample, value, False)
        model.realizePosition(sample)
        achieved[value] = coordinate.getValue(sample)
        return float(
            via.calcDistanceBetween(sample, insertion)
            if segment
            else path.getLength(sample)
        )

    state = osim.State(initial)
    coordinate.setValue(state, q, False)
    coordinate.setSpeedValue(state, 0.0)
    model.realizeVelocity(state)
    model.setControls(state, osim.Vector(1, 1.0))
    model.realizeAcceleration(state)
    native_arm = float(path.computeMomentArm(state, coordinate))
    full_arm = compute_path_length_finite_difference_moment_arm(length_at, q, step)
    segment_arm = compute_path_length_finite_difference_moment_arm(
        lambda value: length_at(value, True), q, step
    )
    rejected = False
    try:
        validate_moment_arm_consistency(native_arm, length_at, q, TOLERANCE_M, step)
    except MomentArmDerivativeMismatchError:
        rejected = True
    velocity_state = osim.State(initial)
    coordinate.setValue(velocity_state, q, False)
    coordinate.setSpeedValue(velocity_state, 1.0)
    model.realizeVelocity(velocity_state)
    return {
        "q_rad": q,
        "step_rad": step,
        "coordinate_span_rad": achieved[q + step] - achieved[q - step],
        "native_moment_arm_m": native_arm,
        "native_acceleration_rad_s2": coordinate.getAccelerationValue(state),
        "native_tension_n": cable.getActuation(state),
        "full_length_arm_m": full_arm,
        "same_body_segment_arm_m": segment_arm,
        "full_minus_native_m": full_arm - native_arm,
        "length_speed_m_s": path.getLengtheningSpeed(velocity_state),
        "gate_rejected": rejected,
    }


def run_probe(output: Path) -> dict[str, Any]:
    """Write original fixtures; return observations that never qualify a donor."""
    import opensim as osim

    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for moving in (False, True):
        model = _build_fixture(osim, moving)
        source = output / ("moving.osim" if moving else "fixed.osim")
        model.printToXML(str(source))
        source_hash = _hash_bytes(source.read_bytes())
        model = osim.Model(str(source))
        initial = model.initSystem()
        loaded_hash = _hash_bytes(model.dump().encode())
        samples = [
            _observe_sample(osim, model, initial, q, step)
            for q in (-0.2, 0.0, 0.2)
            for step in STEPS_RAD
        ]
        cases.append(
            {
                "moving": moving,
                "source_sha256": source_hash,
                "loaded_sha256": loaded_hash,
                "source_unchanged": source_hash == _hash_bytes(source.read_bytes()),
                "loaded_unchanged": loaded_hash == _hash_bytes(model.dump().encode()),
                "native_nq": initial.getNQ(),
                "native_nu": initial.getNU(),
                "native_nz": initial.getNZ(),
                "native_constraints": model.getConstraintSet().getSize(),
                "native_controllers": model.getControllerSet().getSize(),
                "inertia_kg_m2": INERTIA_KG_M2,
                "samples": samples,
                "gate": "workless_identity_not_satisfied"
                if any(s["gate_rejected"] for s in samples)
                else "workless_identity_satisfied_at_tested_samples",
                "guide_work_model": "unmodeled-moving-guide"
                if moving
                else "fixed-points",
            }
        )
    extensions = {
        name: _hash_bytes(Path(module_file).read_bytes())
        for name, module in tuple(sys.modules.items())
        if name.startswith("opensim.")
        and isinstance(module_file := getattr(module, "__file__", None), str)
        and module_file.endswith((".pyd", ".so"))
    }
    return {
        "scientific_status": "unqualified",
        "interpretation": "workless-identity-audit-not-runtime-defect-proof",
        "cases": cases,
        "tolerance_m": TOLERANCE_M,
        "runtime": {
            "version": osim.GetVersionAndDate(),
            "python_version": sys.version,
            "extension_sha256": extensions,
        },
        "reviewed_source_revision": SOURCE_REVISION,
        "reviewed_source_sha256": SOURCE_HASHES,
        "source_binary_equivalence": "unverified",
        "dll_closure": "unverified",
        "probe_sha256": _hash_bytes(Path(__file__).read_bytes()),
        "gate_provider_sha256": _hash_bytes(
            Path(inspect.getfile(validate_moment_arm_consistency)).read_bytes()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_probe(args.output)
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
