"""I/O helpers between Python and the MATLAB Simulink workspace.

Two responsibilities:

1. **Setup** — given a MATLAB engine, a model name and a flat
   coefficient vector, build a ``Simulink.SimulationInput`` object with
   the ``PolynomialInputs`` workspace variable populated. This is the
   payload :meth:`SimscapeAdapter.simulate_with_coefficients` hands to
   ``sim()``.

2. **Extraction** — convert the flat-double struct returned by
   ``simulate_with_coefficients.m`` (issue #018) into a fully validated
   :class:`SimscapeOutput`.

The functions here import ``matlab`` lazily so the module remains
importable without MATLAB installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engines.simscape._errors import SimscapeSimulationError
from src.engines.simscape._output import SimscapeOutput

__all__ = [
    "build_simulation_input",
    "logsout_to_simscape_output",
]


# Field name on the MATLAB-side SimulationInput workspace.
_COEFFS_WORKSPACE_VAR = "PolynomialInputs"


def _to_matlab_double(arr: np.ndarray) -> Any:
    """Convert a 1-D numpy array to a MATLAB double column vector.

    Imports :mod:`matlab` lazily so this module remains importable on
    hosts without MATLAB.
    """
    import matlab  # type: ignore[import-not-found]

    flat = np.ascontiguousarray(arr, dtype=np.float64).reshape(-1)
    return matlab.double(flat.tolist())


def build_simulation_input(
    eng: Any,
    *,
    model_name: str,
    coeffs: np.ndarray,
    n_joints: int,
) -> Any:
    """Build a ``Simulink.SimulationInput`` populated with ``coeffs``.

    The MATLAB side reshapes the flat coefficient vector into the
    ``n_joints x 7`` matrix expected by ``simulate_with_coefficients.m``
    (joint-major; each row = ``[A B C D E F G]`` polynomial terms).

    Args:
        eng: Live MATLAB engine.
        model_name: Loaded Simulink model name (no extension).
        coeffs: Flat 1-D float64 vector of length ``n_joints * 7``.
        n_joints: Number of joints; ``coeffs.size == n_joints * 7``.

    Returns:
        A ``Simulink.SimulationInput`` opaque MATLAB handle.

    Raises:
        ValueError: If ``coeffs`` is not 1-D, has the wrong size, or
            contains non-finite values.
        SimscapeSimulationError: If MATLAB raises while constructing
            the input object.
    """
    if coeffs.ndim != 1:
        raise ValueError(f"coeffs must be 1-D, got ndim={coeffs.ndim}")
    if coeffs.size != n_joints * 7:
        raise ValueError(
            f"coeffs size must be n_joints*7={n_joints * 7}, got {coeffs.size}"
        )
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coeffs must be finite")

    try:
        sim_input = eng.eval(f"Simulink.SimulationInput('{model_name}')", nargout=1)
        coeff_mx = _to_matlab_double(coeffs)
        # setVariable returns a *new* SimulationInput in MATLAB.
        sim_input = eng.setVariable(
            sim_input, _COEFFS_WORKSPACE_VAR, coeff_mx, nargout=1
        )
    except Exception as exc:  # noqa: BLE001 - wrap as SimscapeSimulationError
        err_id = getattr(exc, "MatlabError", "") or ""
        raise SimscapeSimulationError(
            f"failed to build Simulink.SimulationInput: {exc}",
            matlab_error_id=err_id,
        ) from exc
    return sim_input


def _as_ndarray_2d(value: Any, *, name: str) -> np.ndarray:
    """Convert a MATLAB matrix or numeric struct field to a 2-D numpy array."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise SimscapeSimulationError(
            f"logsout field '{name}' must be 2-D after coercion, got ndim={arr.ndim}"
        )
    return arr


def _as_ndarray_1d(value: Any, *, name: str) -> np.ndarray:
    """Convert a MATLAB column to a flat 1-D float64 array."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise SimscapeSimulationError(
            f"logsout field '{name}' must reduce to 1-D, got ndim={arr.ndim}"
        )
    return arr


def logsout_to_simscape_output(logsout: dict[str, Any]) -> SimscapeOutput:
    """Marshal a MATLAB ``logsout`` struct into a :class:`SimscapeOutput`.

    Expected keys (produced by ``simulate_with_coefficients.m``):

    ``time``, ``q``, ``qd``, ``qdd``, ``tau``, ``omega``,
    ``r_butt``, ``r_clubhead``, ``q_club``, ``v_clubhead``.

    The MATLAB side guarantees that ``time`` starts at 0 and
    ``q_club`` rows are unit-norm; we re-validate via
    :class:`SimscapeOutput`'s ``__post_init__``.

    Args:
        logsout: Dict-like view of the flat-double struct returned by
            ``simulate_with_coefficients.m``.

    Returns:
        A frozen :class:`SimscapeOutput` with all numpy arrays in
        contiguous float64 layout.

    Raises:
        SimscapeSimulationError: If a required key is missing or any
        invariant fails.
    """
    required = (
        "time",
        "q",
        "qd",
        "qdd",
        "tau",
        "omega",
        "r_butt",
        "r_clubhead",
        "q_club",
        "v_clubhead",
    )
    missing = [k for k in required if k not in logsout]
    if missing:
        raise SimscapeSimulationError(
            f"logsout missing required field(s): {sorted(missing)}"
        )

    try:
        return SimscapeOutput(
            time=_as_ndarray_1d(logsout["time"], name="time"),
            q=_as_ndarray_2d(logsout["q"], name="q"),
            qd=_as_ndarray_2d(logsout["qd"], name="qd"),
            qdd=_as_ndarray_2d(logsout["qdd"], name="qdd"),
            tau=_as_ndarray_2d(logsout["tau"], name="tau"),
            omega=_as_ndarray_2d(logsout["omega"], name="omega"),
            r_butt=_as_ndarray_2d(logsout["r_butt"], name="r_butt"),
            r_clubhead=_as_ndarray_2d(logsout["r_clubhead"], name="r_clubhead"),
            q_club=_as_ndarray_2d(logsout["q_club"], name="q_club"),
            v_clubhead=_as_ndarray_2d(logsout["v_clubhead"], name="v_clubhead"),
        )
    except (TypeError, ValueError) as exc:
        raise SimscapeSimulationError(
            f"logsout failed SimscapeOutput validation: {exc}"
        ) from exc
