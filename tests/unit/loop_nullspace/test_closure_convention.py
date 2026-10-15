"""NSX-02: canonical closure convention against the ``closure_convention`` golden."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
    closure_conversion_matrix,
    from_canonical_closure,
    to_canonical_closure,
)

pytestmark = pytest.mark.unit

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/loop_nullspace/goldens/closure_convention.json"
)
RTOL, ATOL = 1e-12, 1e-13
ENGINES = ("pinocchio", "drake", "mujoco", "opensim")


def _arr(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.array([_arr(v) for v in value])
    return np.array(float(Fraction(value)))


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _kwargs(golden: dict[str, Any], engine: str) -> dict[str, Any]:
    cfg = golden["natives"][engine]["native_convention"]
    native = NativeClosureConvention(
        row_order=cfg["row_order"],
        frame=cfg["frame"],
        point=cfg["point"],
        relative_sign=cfg["relative_sign"],
    )
    rotation = (
        _arr(golden["R_native_local_frame_world"])
        if native.frame == "LOCAL"
        else np.eye(3)
    )
    return {
        "native": native,
        "native_frame_rotation": rotation,
        "grip_frame_rotation": _arr(golden["R_grip_world_from_frame"]),
        "native_reference_point": _arr(golden["P0_native_reference_point_world"]),
        "grip_reference_point": _arr(golden["P_canonical_reference_point_world"]),
    }


@pytest.mark.parametrize("engine", ENGINES)
def test_golden_conversion_matrix_and_canonical_jacobian(golden, engine) -> None:
    kw = _kwargs(golden, engine)
    j_native = _arr(golden["natives"][engine]["J_native"])
    drift = np.arange(6.0)
    j_c, drift_c = to_canonical_closure(j_native, drift, **kw)
    np.testing.assert_allclose(
        j_c, _arr(golden["J_canonical"]), rtol=RTOL, atol=ATOL, err_msg=engine
    )
    matrix = closure_conversion_matrix(**kw)
    np.testing.assert_allclose(
        matrix,
        _arr(golden["natives"][engine]["C_native_to_canonical"]),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(drift_c, matrix @ drift, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("engine", ENGINES)
def test_power_is_preserved_by_the_dual_transform(golden, engine) -> None:
    kw = _kwargs(golden, engine)
    j_native = _arr(golden["natives"][engine]["J_native"])
    j_c, _ = to_canonical_closure(j_native, np.zeros(6), **kw)
    v = _arr(golden["v"])
    lam_c = _arr(golden["lambda_canonical"])
    lam_native = closure_conversion_matrix(**kw).T @ lam_c
    np.testing.assert_allclose(
        lam_native,
        _arr(golden["natives"][engine]["lambda_native_for_lambda_canonical"]),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        lam_native @ (j_native @ v), lam_c @ (j_c @ v), rtol=RTOL, atol=ATOL
    )
    assert lam_c @ (j_c @ v) == pytest.approx(
        float(Fraction(golden["power_lambda_dot_Jv"]))
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_round_trip_reproduces_the_native_jacobian(golden, engine) -> None:
    kw = _kwargs(golden, engine)
    j_native = _arr(golden["natives"][engine]["J_native"])
    drift = np.array([0.3, -1.0, 2.0, 0.5, 0.1, -0.7])
    j_c, drift_c = to_canonical_closure(j_native, drift, **kw)
    j_back, drift_back = from_canonical_closure(j_c, drift_c, **kw)
    np.testing.assert_allclose(j_back, j_native, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(drift_back, drift, rtol=1e-13, atol=1e-13)


def test_outputs_are_read_only(golden) -> None:
    kw = _kwargs(golden, "pinocchio")
    j_c, drift_c = to_canonical_closure(
        _arr(golden["natives"]["pinocchio"]["J_native"]), np.zeros(6), **kw
    )
    assert not j_c.flags.writeable and not drift_c.flags.writeable


def test_convention_rejects_bad_fields() -> None:
    with pytest.raises(ValueError, match="row_order"):
        NativeClosureConvention("bad", "LOCAL", "p", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frame"):
        NativeClosureConvention("lin_ang", "bad", "p", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="relative_sign"):
        NativeClosureConvention("lin_ang", "LOCAL", "p", 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="point"):
        NativeClosureConvention("lin_ang", "LOCAL", "", 1)


def test_preconditions(golden) -> None:
    kw = _kwargs(golden, "pinocchio")
    j = _arr(golden["natives"]["pinocchio"]["J_native"])
    with pytest.raises(ValueError, match="6 x n"):
        to_canonical_closure(j[:5], np.zeros(6), **kw)
    with pytest.raises(ValueError, match="length 6"):
        to_canonical_closure(j, np.zeros(5), **kw)
    bad = j.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        to_canonical_closure(bad, np.zeros(6), **kw)
    with pytest.raises(ValueError, match="rotation"):
        to_canonical_closure(
            j, np.zeros(6), **{**kw, "grip_frame_rotation": 2.0 * np.eye(3)}
        )
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="rotation"):
        to_canonical_closure(
            j, np.zeros(6), **{**kw, "native_frame_rotation": reflection}
        )
    with pytest.raises(ValueError, match="length 3"):
        to_canonical_closure(
            j, np.zeros(6), **{**kw, "grip_reference_point": np.zeros(2)}
        )


def test_world_native_frame_requires_identity_rotation(golden) -> None:
    kw = _kwargs(golden, "mujoco")
    j = _arr(golden["natives"]["mujoco"]["J_native"])
    with pytest.raises(ValueError, match="WORLD"):
        to_canonical_closure(
            j,
            np.zeros(6),
            **{**kw, "native_frame_rotation": _arr(golden["R_grip_world_from_frame"])},
        )
