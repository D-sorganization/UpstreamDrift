"""Contracts for fixed native initialization across torque-fit prefixes."""

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def verify_native_initial_state(
    seed: Mapping[str, Any],
    q: ArrayLike,
    qd: ArrayLike,
    markers: ArrayLike,
    kinematic_markers: ArrayLike,
    observed: ArrayLike,
) -> dict[str, float | bool]:
    """Verify native q/qd and marker projection, reporting capture error separately.

    All inputs are finite SI arrays. Kinematic markers must come from the declared
    state's native body poses and fixed attachments. A model's nonzero capture
    residual is allowed; native state or projection disagreement is rejected.
    """

    def checked(value: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
        result = np.asarray(value, dtype=float)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError("native initial arrays must have finite matching shapes")
        return result

    count = len(seed["labels"])
    if count < 1:
        raise ValueError("native initial markers must be nonempty")
    expected_q = checked(seed["q"], (27,))
    expected_qd = checked(seed["qd"], (27,))
    position = checked(q, (27,))
    velocity = checked(qd, (27,))
    prediction = checked(markers, (count, 3))
    reference = checked(kinematic_markers, (count, 3))
    target = checked(observed, (count, 3))
    q_error = float(np.max(np.abs(position - expected_q)))
    qd_error = float(np.max(np.abs(velocity - expected_qd)))
    projection_error = float(np.max(np.abs(prediction - reference)))
    if max(q_error, qd_error, projection_error) >= 1e-8:
        raise ValueError(
            "native initial state or marker projection differs from its declaration"
        )
    diff = prediction - target
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=1)
    sq_distances = np.einsum("ij,ij->i", diff, diff)
    return {
        "initial_state_verified": True,
        "initial_q_max_error": q_error,
        "initial_qd_max_error": qd_error,
        "initial_projection_max_error_m": projection_error,
        "initial_target_rms_m": float(np.sqrt(np.mean(sq_distances))),
        "initial_target_max_error_m": float(np.sqrt(np.max(sq_distances))),
    }


BASIS_CONTROLS = {
    f"{name}-bernstein-6": degree + 1
    for degree, name in enumerate(
        ("constant", "linear", "quadratic", "cubic", "quartic", "quintic", "sextic")
    )
}


def transfer_prefix_candidate(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> list[float]:
    """Transfer a completed torque candidate to a fresh, non-shorter objective.

    Fixed state/attachments and joint effort scales must match. Polynomial controls
    are extrapolated in physical seconds; out-of-bounds extrapolation is rejected,
    never clipped. No objective values or optimizer history are transferred.
    """
    if (
        source.get("status") != "exploratory-fit-computed"
        or source.get("initial_state_verified") is not True
    ):
        raise ValueError("transfer requires a completed native source fit")
    if source.get("source_sha256") != target.get("source_sha256") or source.get(
        "labels"
    ) != target.get("labels"):
        raise ValueError("transfer capture or marker order differs")
    old, new = source["fit_identity"], target["fit_identity"]
    variable = {"basis", "basis_duration_s", "duration_s"}
    if {k: v for k, v in old.items() if k not in variable} != {
        k: v for k, v in new.items() if k not in variable
    }:
        raise ValueError("transfer native identity differs")
    durations = [
        old["duration_s"],
        new["duration_s"],
        old["basis_duration_s"],
        new["basis_duration_s"],
    ]
    if (
        not all(math.isfinite(v) and v > 0 for v in durations)
        or new["duration_s"] < old["duration_s"]
        or old["duration_s"] > old["basis_duration_s"]
        or new["duration_s"] > new["basis_duration_s"]
    ):
        raise ValueError("transfer duration must preserve or extend the prefix")
    if old["basis"] not in BASIS_CONTROLS or new["basis"] not in BASIS_CONTROLS:
        raise ValueError("unsupported transfer basis")
    before, after = BASIS_CONTROLS[old["basis"]], BASIS_CONTROLS[new["basis"]]
    if after < before:
        raise ValueError("transfer cannot remove polynomial degrees of freedom")
    joints = len(old["coordinate_names"])
    scales = []
    for report, count in ((source, before), (target, after)):
        values = np.asarray(report["effort_scales"], dtype=float)
        if (
            values.shape != (joints * count,)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError("transfer effort scales are invalid")
        matrix = values.reshape(joints, count)
        if not np.all(matrix == matrix[:, :1]):
            raise ValueError("transfer scales must agree within each joint")
        scales.append(matrix[:, 0])
    if not np.array_equal(scales[0], scales[1]):
        raise ValueError("transfer joint effort scales differ")
    parameters = np.asarray(source["stage"]["parameters"], dtype=float)
    if (
        parameters.shape != (joints * before,)
        or not np.isfinite(parameters).all()
        or np.any((parameters < 0) | (parameters > 2))
    ):
        raise ValueError("source parameter bounds are invalid")
    values = parameters.reshape(joints, before)
    # De Casteljau's left subdivision also extrapolates when ratio exceeds one.
    ratio = new["basis_duration_s"] / old["basis_duration_s"]
    result = np.empty_like(values)
    for level in range(before):
        result[:, level] = values[:, 0]
        values = (1 - ratio) * values[:, :-1] + ratio * values[:, 1:]
    # Degree elevation adds freedom without changing the physical-time curve.
    while result.shape[1] < after:
        count = result.shape[1]
        weight = np.arange(1, count) / count
        interior = weight * result[:, :-1] + (1 - weight) * result[:, 1:]
        result = np.column_stack((result[:, 0], interior, result[:, -1]))
    if np.any((result < 0) | (result > 2)):
        raise ValueError("extended candidate exceeds effort bounds")
    return [float(v) for v in result.ravel()]


def qualified_fit_identity(
    seed: Mapping[str, Any],
    capture_sha256: str,
    duration_s: float,
    basis_duration_s: float,
    *,
    basis: str = "constant-bernstein-6",
) -> dict[str, Any]:
    """Bind a verified SI state and inch geometry to a capture and polynomial basis.

    Qualification is established by native replay, not by this schema validator.
    Copies keep a later caller mutation from changing checkpoint identity.
    """
    if basis not in BASIS_CONTROLS:
        raise ValueError("unsupported torque basis")
    if (
        seed.get("initial_state_verified") is not True
        or seed.get("status") != "initial-state-qualified"
    ):
        raise ValueError("native initial state has not been qualified")
    if seed.get("source_sha256") != capture_sha256:
        raise ValueError("initial state belongs to another capture")
    if (
        not all(math.isfinite(v) for v in (duration_s, basis_duration_s))
        or not 0 < duration_s <= basis_duration_s
    ):
        raise ValueError("duration must lie within the polynomial basis duration")
    names = seed.get("coordinate_names", [])
    labels = seed.get("labels", [])
    bodies = seed.get("body_names", [])
    if (
        len(names) != 27
        or len(set(names)) != 27
        or not all(isinstance(n, str) and n for n in names)
    ):
        raise ValueError("expected 27 unique ordered native coordinates")
    if (
        not labels
        or len(set(labels)) != len(labels)
        or len(bodies) != len(labels)
        or not all(isinstance(n, str) and n for n in [*labels, *bodies])
    ):
        raise ValueError("marker labels and fixed body assignments are inconsistent")
    for key, shape in (
        ("q", (27,)),
        ("qd", (27,)),
        ("geometry_in", (2,)),
        ("offsets_m", (len(labels), 3)),
    ):
        values = np.asarray(seed.get(key, []), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid finite shape for {key}")
        if key == "geometry_in" and np.any(values <= 0):
            raise ValueError("geometry lengths must be positive inches")
    result = deepcopy(
        {
            key: seed[key]
            for key in (
                "coordinate_names",
                "labels",
                "body_names",
                "q",
                "qd",
                "geometry_in",
                "offsets_m",
            )
        }
    )
    result.update(
        basis=basis,
        duration_s=duration_s,
        basis_duration_s=basis_duration_s,
    )
    return result
