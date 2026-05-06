"""Unit tests for the SimscapeAdapter protocol-compliant skeleton (#4005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.engines.simscape import SimscapeAdapter
from src.engines.simscape._errors import (
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeStateError,
)
from src.engines.simscape._lifecycle import AdapterState
from src.shared.python.engine_core.checkpoint import Checkpointable
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.engine_core.sub_protocols import (
    CounterfactualComputable,
    DynamicsComputable,
    Loadable,
    Queryable,
    Recordable,
    Steppable,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slx_path(tmp_path: Path) -> str:
    """Create a fake .slx and the metadata sibling required by the skeleton."""
    slx = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx.write_bytes(b"FAKE_SLX_FOR_TESTS")
    metadata = tmp_path / "PolynomialInputValues.mat"
    metadata.write_bytes(b"FAKE_MAT_FOR_TESTS")
    return str(slx)


@pytest.fixture
def adapter() -> SimscapeAdapter:
    return SimscapeAdapter()


@pytest.fixture
def loaded_adapter(slx_path: str) -> SimscapeAdapter:
    a = SimscapeAdapter()
    a.load_from_path(slx_path)
    return a


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_satisfies_physics_engine_protocol(adapter: SimscapeAdapter) -> None:
    assert isinstance(adapter, PhysicsEngine)


@pytest.mark.unit
def test_satisfies_every_sub_protocol(adapter: SimscapeAdapter) -> None:
    for proto in (
        Loadable,
        Steppable,
        Queryable,
        DynamicsComputable,
        CounterfactualComputable,
        Recordable,
        Checkpointable,
    ):
        assert isinstance(adapter, proto), (
            f"SimscapeAdapter does not satisfy {proto.__name__}"
        )


# ---------------------------------------------------------------------------
# Init / metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_name_returns_simscape_3d_string(adapter: SimscapeAdapter) -> None:
    assert adapter.name == "simscape_3d"


@pytest.mark.unit
def test_engine_type_matches_registry(adapter: SimscapeAdapter) -> None:
    assert adapter.engine_type == "matlab_3d"


@pytest.mark.unit
def test_dof_unavailable_before_load(adapter: SimscapeAdapter) -> None:
    with pytest.raises(SimscapeStateError):
        _ = adapter.dof


@pytest.mark.unit
def test_dof_known_after_load_metadata(loaded_adapter: SimscapeAdapter) -> None:
    # 16 polynomial joints in GolfSwing3D_Kinetic.
    assert loaded_adapter.dof == 16
    assert len(loaded_adapter.joint_names) == 16
    assert loaded_adapter.model_name == "GolfSwing3D_Kinetic"


@pytest.mark.unit
def test_state_summary_contains_lifecycle_and_dof(
    loaded_adapter: SimscapeAdapter,
) -> None:
    summary = loaded_adapter.state_summary()
    assert summary["lifecycle"] == AdapterState.LOADED.value
    assert summary["dof"] == 16
    assert summary["model_loaded"] is True
    assert summary["name"] == "simscape_3d"


# ---------------------------------------------------------------------------
# load_from_path / load_from_string
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_from_string_raises_not_implemented(adapter: SimscapeAdapter) -> None:
    with pytest.raises(NotImplementedError, match="binary"):
        adapter.load_from_string("<not slx>")


@pytest.mark.unit
def test_load_from_path_rejects_non_slx(adapter: SimscapeAdapter) -> None:
    with pytest.raises(ValueError, match=r"\.slx"):
        adapter.load_from_path("model.urdf")


@pytest.mark.unit
def test_load_from_path_missing_file_raises(
    adapter: SimscapeAdapter, tmp_path: Path
) -> None:
    missing = str(tmp_path / "missing.slx")
    with pytest.raises(SimscapeModelNotFoundError):
        adapter.load_from_path(missing)


@pytest.mark.unit
def test_load_from_path_missing_metadata_raises(
    adapter: SimscapeAdapter, tmp_path: Path
) -> None:
    slx = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx.write_bytes(b"FAKE")
    with pytest.raises(SimscapeModelNotFoundError, match="PolynomialInputValues"):
        adapter.load_from_path(str(slx))


# ---------------------------------------------------------------------------
# Lifecycle / state-machine guards on real methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_before_load_raises_state_error(adapter: SimscapeAdapter) -> None:
    with pytest.raises(SimscapeStateError):
        adapter.step()


@pytest.mark.unit
def test_simulate_before_load_raises_state_error(
    adapter: SimscapeAdapter,
) -> None:
    with pytest.raises(SimscapeStateError):
        adapter.simulate_with_coefficients(np.zeros(16 * 7))


@pytest.mark.unit
def test_get_state_before_load_raises_state_error(
    adapter: SimscapeAdapter,
) -> None:
    with pytest.raises(SimscapeStateError):
        adapter.get_state()


@pytest.mark.unit
def test_step_after_load_raises_not_installed_without_matlab(
    loaded_adapter: SimscapeAdapter,
) -> None:
    """#4006: step now performs a real MATLAB call; without MATLAB it
    must raise :class:`SimscapeNotInstalledError`, not ``NotImplementedError``.
    """
    with pytest.raises(SimscapeNotInstalledError):
        loaded_adapter.step()


@pytest.mark.unit
def test_simulate_with_coefficients_raises_not_installed_without_matlab(
    loaded_adapter: SimscapeAdapter,
) -> None:
    """#4006: simulate now performs a real MATLAB call; without MATLAB it
    must raise :class:`SimscapeNotInstalledError`, not ``NotImplementedError``.
    """
    with pytest.raises(SimscapeNotInstalledError):
        loaded_adapter.simulate_with_coefficients(np.zeros(16 * 7))


@pytest.mark.unit
def test_get_state_returns_zero_arrays_after_load(
    loaded_adapter: SimscapeAdapter,
) -> None:
    q, v = loaded_adapter.get_state()
    assert q.shape == (16,)
    assert v.shape == (16,)
    assert np.all(q == 0.0)
    assert np.all(v == 0.0)


@pytest.mark.unit
def test_get_time_zero_after_load(loaded_adapter: SimscapeAdapter) -> None:
    assert loaded_adapter.get_time() == 0.0


@pytest.mark.unit
def test_set_control_stores_finite_vector(
    loaded_adapter: SimscapeAdapter,
) -> None:
    u = np.ones(16, dtype=np.float64)
    loaded_adapter.set_control(u)  # should not raise


@pytest.mark.unit
def test_reset_after_load_returns_to_loaded(
    loaded_adapter: SimscapeAdapter,
) -> None:
    loaded_adapter.reset()
    assert loaded_adapter.model_loaded


# ---------------------------------------------------------------------------
# close() + context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_close_is_idempotent(adapter: SimscapeAdapter) -> None:
    adapter.close()
    adapter.close()  # must not raise
    assert not adapter.model_loaded


@pytest.mark.unit
def test_close_after_load_idempotent(loaded_adapter: SimscapeAdapter) -> None:
    loaded_adapter.close()
    loaded_adapter.close()


@pytest.mark.unit
def test_context_manager_calls_close_on_exit(slx_path: str) -> None:
    with SimscapeAdapter() as a:
        a.load_from_path(slx_path)
        assert a.model_loaded
    assert not a.model_loaded


# ---------------------------------------------------------------------------
# repr / DRY / safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_repr_does_not_leak_paths(loaded_adapter: SimscapeAdapter) -> None:
    text = repr(loaded_adapter)
    # The fixture's slx lives under tmp_path; the model basename should be
    # the only piece carried into __repr__.
    assert "GolfSwing3D_Kinetic" in text
    assert "/" not in text and "\\" not in text


@pytest.mark.unit
def test_repr_unloaded_marker(adapter: SimscapeAdapter) -> None:
    assert "<unloaded>" in repr(adapter)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_save_and_restore_checkpoint(loaded_adapter: SimscapeAdapter) -> None:
    cp = loaded_adapter.save_checkpoint()
    assert cp.engine_type == "matlab_3d"
    loaded_adapter.restore_checkpoint(cp)


@pytest.mark.unit
def test_restore_checkpoint_rejects_foreign_engine(
    loaded_adapter: SimscapeAdapter,
) -> None:
    cp = loaded_adapter.save_checkpoint()
    foreign = type(cp)(
        id=cp.id,
        timestamp=cp.timestamp,
        wall_time=cp.wall_time,
        engine_type="mujoco",
        engine_state=cp.engine_state,
        q=cp.q,
        v=cp.v,
        step_count=cp.step_count,
        metadata=cp.metadata,
        checksum=cp.checksum,
    )
    with pytest.raises(ValueError, match="engine_type"):
        loaded_adapter.restore_checkpoint(foreign)
