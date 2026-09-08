"""Phase 0.1 smoke test: bioptim imports and solves in a pip-only checkout.

Lives beside the ``ocp`` package rather than under ``tests/unit`` because
that tree replaces ``casadi`` with a spec-less ``MagicMock``.
"""

from __future__ import annotations

import os
import sys

import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = [pytest.mark.integration, pytest.mark.requires_bioptim]

os.environ.setdefault("MPLBACKEND", "Agg")


def test_import_hint_and_probe_are_consistent() -> None:
    assert "upstream-drift[bioptim]" in _compat.BIOPTIM_INSTALL_HINT
    if not _compat.bioptim_available():
        with pytest.raises(_compat.BioptimNotAvailableError, match="bioptim"):
            _compat.require_bioptim()
        assert _compat.bioptim_version() is None
        pytest.skip("bioptim not installed")


@pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed")
def test_shims_only_install_when_real_modules_are_absent() -> None:
    bioptim = _compat.require_bioptim()
    assert hasattr(bioptim, "OptimalControlProgram")
    biorbd = sys.modules["biorbd_casadi"]
    if getattr(biorbd, "__ud_shim__", False):
        assert biorbd.__version__ == _compat.BIORBD_SHIM_VERSION
        assert biorbd.Model("anything") is not None  # permissive attribute
    # Idempotent: a second call never re-installs.
    assert _compat.install_biorbd_shim() is False
    assert _compat.install_tkinter_shim() is False
    assert _compat.install_matplotlib_shim() is False
    assert _compat.bioptim_version() == str(bioptim.__version__)


@pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed")
def test_upstream_custom_model_pendulum_solves() -> None:
    """bioptim's own custom-model example converges with PyPI casadi."""
    import numpy as np

    bioptim = _compat.require_bioptim()
    from bioptim.examples.toy_examples.custom_model.custom_package import MyModel
    from bioptim.examples.toy_examples.custom_model.main import prepare_ocp

    ocp = prepare_ocp(model=MyModel(), final_time=1.0, n_shooting=30, n_threads=1)
    solver = bioptim.Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(200)
    sol = ocp.solve(solver=solver)
    assert sol.status == 0
    q = sol.decision_states(to_merge=bioptim.SolutionMerge.NODES)["q"]
    assert np.isclose(float(np.asarray(q)[0, -1]), 3.14, atol=1e-2)
