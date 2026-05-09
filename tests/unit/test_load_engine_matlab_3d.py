"""Tests for ``load_matlab_3d_engine`` registration in the engine loader registry.

Issue #4007 / docs/issues/backlog/038_register_matlab_3d_loader.md.

These tests validate that ``EngineType.MATLAB_3D`` resolves to a
``SimscapeAdapter`` via the loader registry. Tests that actually require
a live MATLAB Engine are marked ``requires_matlab`` (alias for
``live_simulation``) and skipped on hosts without MATLAB.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engines.loaders import (
    DEFAULT_MATLAB_3D_SLX_RELPATH,
    LOADER_MAP,
    load_matlab_3d_engine,
)
from src.engines.simscape import SimscapeAdapter
from src.shared.python.engine_core.engine_registry import EngineType


def _suite_root() -> Path:
    """Return the repository root (this worktree)."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Registration / dispatch
# ---------------------------------------------------------------------------


def test_loaders_dispatch_table_includes_matlab_3d() -> None:
    """``LOADER_MAP`` must contain a ``MATLAB_3D`` entry post-#4007."""
    assert EngineType.MATLAB_3D in LOADER_MAP
    assert LOADER_MAP[EngineType.MATLAB_3D] is load_matlab_3d_engine


def test_engine_type_matlab_3d_resolves_to_simscape_adapter(
    tmp_path: Path,
) -> None:
    """Calling the registered loader returns a ``SimscapeAdapter`` instance.

    Uses an empty ``tmp_path`` as ``suite_root`` so the default ``.slx``
    is missing — the loader must still return a usable (unloaded) adapter.
    """
    engine = LOADER_MAP[EngineType.MATLAB_3D](tmp_path)
    assert isinstance(engine, SimscapeAdapter)


def test_load_engine_matlab_3d_returns_protocol_compliant_instance(
    tmp_path: Path,
) -> None:
    """Returned engine exposes the documented PhysicsEngine surface."""
    engine = load_matlab_3d_engine(tmp_path)
    # Spot-check protocol composition without importing the protocol class.
    assert hasattr(engine, "load_from_path")
    assert hasattr(engine, "step")
    assert hasattr(engine, "get_state")
    assert hasattr(engine, "save_checkpoint")
    assert hasattr(engine, "engine_type")
    # Type identifier must match EngineType.MATLAB_3D.
    assert engine.engine_type == EngineType.MATLAB_3D.value


# ---------------------------------------------------------------------------
# Default model resolution
# ---------------------------------------------------------------------------


def test_default_model_path_resolves_to_GolfSwing3D_Kinetic_slx() -> None:
    """The relative default points at the canonical GolfSwing3D_Kinetic.slx."""
    assert DEFAULT_MATLAB_3D_SLX_RELPATH.name == "GolfSwing3D_Kinetic.slx"
    # Path is relative — caller composes it under suite_root.
    assert not DEFAULT_MATLAB_3D_SLX_RELPATH.is_absolute()
    # Component parts trace through Simscape Multibody Models.
    parts = DEFAULT_MATLAB_3D_SLX_RELPATH.parts
    assert "Simscape_Multibody_Models" in parts
    assert "3D_Golf_Model" in parts


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_model_path_raises_clear_error(tmp_path: Path) -> None:
    """A non-Path ``suite_root`` raises :class:`TypeError`."""
    with pytest.raises(TypeError, match="suite_root must be a Path"):
        load_matlab_3d_engine("not-a-path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Live-MATLAB tests — only run with a working MATLAB Engine for Python.
# ---------------------------------------------------------------------------


@pytest.mark.requires_matlab
def test_live_matlab_3d_simulate_smoke() -> None:  # pragma: no cover - requires MATLAB
    """Full simulate smoke test on a host with MATLAB."""
    import numpy as np

    engine = load_matlab_3d_engine(_suite_root())
    assert isinstance(engine, SimscapeAdapter)
    # 16 polynomial joints x 7 coefficients = 112 floats.
    coeffs = np.zeros(16 * 7, dtype=np.float64)
    out = engine.simulate_with_coefficients(coeffs)
    assert out is not None
