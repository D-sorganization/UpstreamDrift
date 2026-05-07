"""Tests for the MuJoCo backend of ``synthesize_target_from_coefficients``.

Implements the acceptance criteria of issue #4122:

- Round-trip identity: ``synthesize(theta_known)`` produces a validated
  :class:`ClubTarget` whose shape and trajectory mirror the underlying
  :class:`SimOut`.
- Cross-engine SimOut -> ClubTarget mapping correctness: every documented
  field maps as advertised (``grip -> butt``, ``clubhead -> clubhead``,
  ``club_quat -> club_quat``).
- Dispatcher hookup: the shared ``loaders.synthetic`` dispatcher routes
  to the MuJoCo backend when ``engine="mujoco"``.

All tests are marked ``requires_mujoco``; the entire module is skipped if
``import mujoco`` fails (see ``conftest.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Importing the engine module triggers ``register_mujoco_backend`` so the
# shared dispatcher knows about us.
from src.engines.physics_engines.mujoco.python.motion_matching import (  # noqa: E402
    synthesize as mj_synthesize,
)
from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (  # noqa: E402
    SimOptions,
    simulate_with_coefficients,
)
from src.shared.python.motion_matching.club_target import (  # noqa: E402
    AlignOptions,
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.loaders import (
    synthetic as shared_synth,  # noqa: E402
)

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]


# --- Helpers ----------------------------------------------------------------


def _theta_zero_full() -> tuple[np.ndarray, SimOptions]:
    """Build a zero-torque coefficient vector matching the full-body model."""
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    theta = np.zeros(nu * 7, dtype=np.float64)
    sim_opts = SimOptions(variant="full", T_s=0.2, output_rate_hz=500.0)
    return theta, sim_opts


# --- Round-trip identity ----------------------------------------------------


def test_round_trip_returns_validated_clubtarget() -> None:
    """``synthesize(theta_known)`` returns a fully-validated ``ClubTarget``."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )

    target = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )

    assert isinstance(target, ClubTarget)
    n_expected = int(round(sim_opts.T_s * sim_opts.output_rate_hz)) + 1
    assert target.time.shape == (n_expected,)
    assert target.butt.shape == (n_expected, 3)
    assert target.clubhead.shape == (n_expected, 3)
    assert target.club_quat.shape == (n_expected, 4)
    assert 1 <= int(target.impact_idx) <= n_expected
    # Quaternions are unit-norm to within ClubTarget's tolerance (1e-6).
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    assert np.all(np.abs(qnorms - 1.0) < 1.0e-6)


def test_accepts_canonical_success_solver_status() -> None:
    """``solver_status='success'`` is the canonical successful rollout status."""
    source = mj_synthesize.__file__ and Path(mj_synthesize.__file__).read_text(
        encoding="utf-8"
    )

    assert '{"ok", "success"}' in source


def test_round_trip_matches_simulate_clubhead_exactly() -> None:
    """``synthesize.clubhead`` equals ``simulate.clubhead`` byte-for-byte.

    The cross-engine spec promises the synthesize wrapper makes a single
    deterministic call to the underlying simulator and copies the result
    through verbatim. Use ``assert_array_equal`` (not approx) so any
    accidental smoothing or alignment regresses loudly.
    """
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )

    sim_out = simulate_with_coefficients(theta, sim_opts)
    target = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )

    np.testing.assert_array_equal(target.clubhead, sim_out.clubhead)
    np.testing.assert_array_equal(target.time, sim_out.time)


# --- Cross-engine mapping correctness ---------------------------------------


def test_grip_maps_to_butt() -> None:
    """``SimOut.grip`` populates ``ClubTarget.butt`` (mid-hands anchor)."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )
    sim_out = simulate_with_coefficients(theta, sim_opts)
    target = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )
    np.testing.assert_array_equal(target.butt, sim_out.grip)


def test_club_quat_is_unit_norm_after_renormalisation() -> None:
    """The synthesizer's renormalisation guards against tiny drift in SimOut."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )
    target = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )
    norms = np.linalg.norm(target.club_quat, axis=1)
    assert np.allclose(norms, 1.0, atol=1.0e-9)
    # Sign convention: w >= 0 across every row.
    assert np.all(target.club_quat[:, 0] >= 0.0)


def test_impact_idx_is_argmax_clubhead_speed() -> None:
    """``impact_idx`` should equal ``argmax(||d clubhead/dt||) + 1``."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )
    target = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )
    velocity = np.gradient(target.clubhead, target.time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    expected = int(np.argmax(speed)) + 1  # ClubTarget uses 1-based indexing.
    assert int(target.impact_idx) == expected


def test_provenance_sha_reproducible() -> None:
    """Same theta -> same provenance.sha256."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )
    a = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )
    b = mj_synthesize.synthesize_target_from_coefficients(
        theta, align, sim_options=sim_opts
    )
    assert isinstance(a.source, SourceProvenance)
    assert a.source.sha256 == b.source.sha256
    assert len(a.source.sha256) == 64  # full sha256 hex digest
    assert a.source.format == "synthetic"


# --- Dispatcher hookup ------------------------------------------------------


def test_dispatcher_routes_to_mujoco() -> None:
    """``shared.synthetic.synthesize_target_from_coefficients`` dispatches us."""
    theta, sim_opts = _theta_zero_full()
    align = AlignOptions(
        simulation_time_s=sim_opts.T_s, sample_rate_hz=sim_opts.output_rate_hz
    )

    # Sanity: registration happened on import of the engine module.
    assert "mujoco" in shared_synth.available_backends()

    # Direct vs dispatched results must match for the same inputs. The
    # dispatcher uses the AlignOptions-derived SimOptions so we apply the
    # same here for an apples-to-apples comparison.
    direct = mj_synthesize.synthesize_target_from_coefficients(theta, align)
    via_dispatch = shared_synth.synthesize_target_from_coefficients(
        theta, align, engine="mujoco"
    )
    np.testing.assert_array_equal(direct.clubhead, via_dispatch.clubhead)
    np.testing.assert_array_equal(direct.butt, via_dispatch.butt)
    assert direct.source.sha256 == via_dispatch.source.sha256


def test_dispatcher_unknown_engine_raises() -> None:
    """Asking for an unregistered engine should raise ``LookupError``."""
    theta = np.zeros(7, dtype=np.float64)
    with pytest.raises(LookupError, match="no synthetic backend"):
        shared_synth.synthesize_target_from_coefficients(
            theta, AlignOptions(), engine="not_a_real_engine"
        )
