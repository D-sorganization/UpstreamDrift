"""Integration tests for :class:`SimscapeAdapter.simulate_with_coefficients`.

All tests in this module require the MATLAB Engine for Python and the
GolfSwing3D_Kinetic Simulink model on disk. They are auto-skipped when
``matlab.engine`` cannot be imported, so CI without MATLAB still runs
the test discovery cleanly.

Markers:
    requires_matlab: gates every test in this file.
    slow:            additionally gates the ground-truth comparison
                     against the MATLAB ``simulate_with_coefficients.m``
                     reference, which spins up the engine and runs a
                     full simulation pass.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from src.engines.simscape import SimscapeAdapter, SimscapeOutput

pytestmark = [
    pytest.mark.requires_matlab,
    pytest.mark.live_simulation,
]


_MODEL_PATH_ENV = "UD_SIMSCAPE_MODEL_PATH"
"""Override the .slx path with this env var; otherwise we assume the
default repo location."""


def _model_path() -> str:
    """Return the .slx path to use for these integration tests."""
    override = os.environ.get(_MODEL_PATH_ENV)
    if override:
        return override
    repo = Path(__file__).resolve().parents[3]
    return str(
        repo
        / "src"
        / "engines"
        / "Simscape_Multibody_Models"
        / "3D_Golf_Model"
        / "GolfSwing3D_Kinetic.slx"
    )


@pytest.fixture(scope="module")
def adapter() -> Iterator[SimscapeAdapter]:
    """Module-scoped adapter so the engine starts at most once."""
    a = SimscapeAdapter()
    a.load_from_path(_model_path())
    yield a
    a.close()


def test_load_from_path_starts_engine_lazily() -> None:
    """The adapter starts the shared engine on the first ``load_from_path``."""
    a = SimscapeAdapter()
    a.load_from_path(_model_path())
    assert a.model_loaded
    assert a.dof > 0
    a.close()


def test_simulate_returns_canonical_output(adapter: SimscapeAdapter) -> None:
    coeffs = np.zeros(adapter.dof * 7, dtype=np.float64)
    out = adapter.simulate_with_coefficients(coeffs)
    assert isinstance(out, SimscapeOutput)
    assert out.n_samples > 1
    assert out.n_joints == adapter.dof


def test_zero_coefficients_static_pose(adapter: SimscapeAdapter) -> None:
    """All-zero polynomial coefficients should produce a near-static pose."""
    coeffs = np.zeros(adapter.dof * 7, dtype=np.float64)
    out = adapter.simulate_with_coefficients(coeffs)
    # Joint angular velocities should remain small throughout.
    assert float(np.max(np.abs(out.qd))) < 1.0


def test_close_quits_engine() -> None:
    """``close`` must release this adapter's MATLAB reference."""
    a = SimscapeAdapter()
    a.load_from_path(_model_path())
    a.close()
    # After close, simulate must fail (lifecycle is STOPPED).
    from src.engines.simscape._errors import SimscapeStateError

    with pytest.raises(SimscapeStateError):
        a.simulate_with_coefficients(np.zeros(a.dof * 7))


@pytest.mark.slow
def test_simulate_matches_matlab_direct_within_tol(
    adapter: SimscapeAdapter,
) -> None:
    """Compare adapter output against direct MATLAB ``simulate_with_coefficients``."""
    import matlab  # type: ignore[import-not-found]
    from src.engines.simscape._engine_pool import get_shared_engine

    coeffs = np.zeros(adapter.dof * 7, dtype=np.float64)
    out_py = adapter.simulate_with_coefficients(coeffs)

    eng = get_shared_engine()
    coeff_mx = matlab.double(coeffs.tolist())
    sim_input = eng.eval(f"Simulink.SimulationInput('{adapter.model_name}')", nargout=1)
    sim_input = eng.setVariable(sim_input, "PolynomialInputs", coeff_mx, nargout=1)
    direct = eng.simulate_with_coefficients(sim_input, nargout=1)
    direct_time = np.asarray(direct["time"], dtype=np.float64).reshape(-1)

    # Same number of samples and same time grid.
    assert out_py.time.shape == direct_time.shape
    np.testing.assert_allclose(out_py.time, direct_time, atol=1e-12)
