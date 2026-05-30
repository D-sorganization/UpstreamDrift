"""Unit tests for the MuJoCo Warp GPU backend.

These tests must pass on a **CPU-only machine without warp/mujoco_warp/CUDA**.
The contract is that:

* importing the backend module never imports the GPU stack at top level;
* the static capability descriptor is inspectable with no GPU;
* requesting the backend without ``[warp]`` raises ``BackendNotAvailableError``.

The only GPU-touching test is guarded by ``@pytest.mark.requires_gpu`` and skips
when no CUDA/warp device is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    BackendNotAvailableError,
    GolfModelParams,
    has_warp,
    make_backend,
    warp_device_available,
)
from src.shared.python.simulation_backends.mjwarp_backend import MJWarpBackend

pytestmark = pytest.mark.unit

# Seed all RNG per repo standard (deterministic, even if unused here).
_RNG = np.random.default_rng(0)

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "shared"
    / "python"
    / "simulation_backends"
    / "mjwarp_backend.py"
)


def test_module_imports_without_warp() -> None:
    """The backend module imports cleanly even though warp is absent here."""
    spec = importlib.util.find_spec(
        "src.shared.python.simulation_backends.mjwarp_backend"
    )
    assert spec is not None
    assert MJWarpBackend.__name__ == "MJWarpBackend"


def test_describe_capabilities_flags() -> None:
    """Static capability descriptor is correct with no GPU present."""
    caps = MJWarpBackend.describe_capabilities()
    assert caps.name == "mjwarp"
    assert caps.device == "cuda"
    assert caps.supports_batched is True
    assert caps.is_differentiable is False
    assert caps.provides_dynamics is False


def test_capabilities_property_matches_static() -> None:
    """The instance-less static and the descriptor agree on every flag."""
    caps = MJWarpBackend.describe_capabilities()
    assert caps == MJWarpBackend.describe_capabilities()


@pytest.mark.skipif(has_warp(), reason="warp installed")
def test_make_backend_without_warp_raises() -> None:
    """Requesting mjwarp without the [warp] extra raises a clear error."""
    with pytest.raises(BackendNotAvailableError):
        make_backend("mjwarp", GolfModelParams.default())


@pytest.mark.skipif(has_warp(), reason="warp installed")
def test_direct_construction_without_warp_raises() -> None:
    """Direct construction also gates on require_warp() first."""
    with pytest.raises(BackendNotAvailableError):
        MJWarpBackend(GolfModelParams.default())


def test_no_top_level_gpu_imports() -> None:
    """Source must not import warp / mujoco_warp at module scope.

    A top-level import would break the no-GPU import guarantee. We assert that
    no *unindented* line imports either package; such imports must live indented
    inside methods (after ``require_warp()``).
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    forbidden = ("import warp", "import mujoco_warp")
    offending: list[str] = []
    for raw_line in source.splitlines():
        # A top-level statement starts at column 0 (no leading whitespace).
        if raw_line != raw_line.lstrip():
            continue
        stripped = raw_line.strip()
        for token in forbidden:
            if stripped == token or stripped.startswith(token + " "):
                offending.append(raw_line)
    assert not offending, f"top-level GPU imports found: {offending}"


@pytest.mark.requires_gpu
@pytest.mark.skipif(not warp_device_available(), reason="no CUDA/warp")
def test_rollout_batch_gpu_smoke() -> None:
    """GPU smoke: construct the backend and run a 2-env batched rollout."""
    backend = make_backend("mjwarp", GolfModelParams.default(), dt=0.01)
    horizon, num_envs = 5, 2
    batch = backend.rollout_batch(None, horizon=horizon, dt=0.01, num_envs=num_envs)
    assert batch.num_envs == num_envs
    assert batch.num_steps == horizon + 1
    assert batch.q.shape == (num_envs, horizon + 1, 2)
    assert batch.v.shape == (num_envs, horizon + 1, 2)
    assert np.all(np.isfinite(batch.q))
    assert np.all(np.isfinite(batch.v))
