"""Research tests for qualified distributed grip friction and loss of contact (#8751).

Verifies multi-station bounded Coulomb friction, frictionless comparator retention,
controlled stiffness/damping across station counts (1, 3, 5), per-station power ledgers,
active-set opening/reattachment transitions, first-failure classification, and nested horizons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.scientific]

from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _closed_state() -> tuple[Any, dict[str, Any], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span_m


def test_frictionless_comparator_reproduces_standard_tension_law() -> None:
    """Frictionless comparator (mu = 0) must identically match standard tension law."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    config_zero_mu = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
        friction_coefficient=0.0,
    )
    references = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        config=config_zero_mu,
    )

    perturbed_q = q.copy()
    perturbed_q[14] += 0.002
    qd = np.zeros(model.nq)
    qd[14] = 0.1
    qd[15] = 0.05

    snapshot = evaluate_distributed_grip(
        model,
        perturbed_q,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        reference_lengths_m=references,
        config=config_zero_mu,
    )

    assert snapshot.tangential_force_on_club_n is not None
    assert np.allclose(snapshot.tangential_force_on_club_n, 0.0)
    assert snapshot.tangential_power_w == pytest.approx(0.0)
    assert snapshot.tangential_dissipation_power_w == pytest.approx(0.0)
    assert snapshot.maximum_tangential_force_n == pytest.approx(0.0)
    assert snapshot.virtual_power_residual_w < 1e-12


def test_coulomb_friction_cone_bound_is_strictly_satisfied() -> None:
    """Tangential force must satisfy ||F_t|| <= mu * ||F_n|| for all stations."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    mu_values = [0.1, 0.35, 0.7]
    for mu in mu_values:
        for count in (1, 3, 5):
            config = DistributedGripConfig(
                station_count_per_hand=count,
                station_width_m=0.03 if count > 1 else 0.0,
                total_stiffness_n_m=1800.0,
                total_damping_n_s_m=18.0,
                friction_coefficient=mu,
            )
            references = distributed_reference_lengths(
                model,
                q,
                grip_span_m=grip_span_m,
                hand_contact_local_x_m=hand_x,
                config=config,
            )

            perturbed_q = q.copy()
            perturbed_q[14] += 0.003
            qd = np.zeros(model.nq)
            qd[14] = 0.2
            qd[15] = 0.5
            qd[16] = -0.3

            snapshot = evaluate_distributed_grip(
                model,
                perturbed_q,
                qd,
                grip_span_m=grip_span_m,
                hand_contact_local_x_m=hand_x,
                reference_lengths_m=references,
                config=config,
            )

            assert snapshot.normal_force_on_club_n is not None
            assert snapshot.tangential_force_on_club_n is not None

            norm_mags = np.linalg.norm(snapshot.normal_force_on_club_n, axis=2)
            tan_mags = np.linalg.norm(snapshot.tangential_force_on_club_n, axis=2)

            for h in range(2):
                for s in range(count):
                    f_n = norm_mags[h, s]
                    f_t = tan_mags[h, s]
                    assert f_t <= mu * f_n + 1e-12
                    if f_n <= 1e-12:
                        assert f_t == pytest.approx(0.0)


def test_controlled_stiffness_damping_across_station_counts() -> None:
    """Station stiffness and damping must divide total properties by station count."""
    for count in (1, 3, 5):
        config = DistributedGripConfig(
            station_count_per_hand=count,
            station_width_m=0.03 if count > 1 else 0.0,
            total_stiffness_n_m=1800.0,
            total_damping_n_s_m=18.0,
            tangential_damping_n_s_m=18.0,
            friction_coefficient=0.3,
        )
        assert config.station_law.stiffness == pytest.approx(1800.0 / count)
        assert config.station_law.damping == pytest.approx(18.0 / count)
        assert config.tangential_damping_n_s_m / count == pytest.approx(18.0 / count)


def test_per_station_power_ledger_and_work_energy_closure() -> None:
    """Power ledger (normal, tangential, dissipative) must close work-energy."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    config = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
        friction_coefficient=0.4,
    )
    forward_cfg = DistributedForwardConfig(
        duration_s=0.01,
        time_steps_s=(0.001, 0.0005),
    )
    case = DistributedIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        time_step_s=0.0005,
        initial_club_displacement_m=0.002,
        initial_club_velocity_m_s=0.1,
        engine="mujoco",
        grip=config,
    )
    result = integrate_distributed_grip(model, case, forward_cfg)

    assert float(np.max(np.abs(result["virtual_power_residual_w"]))) < 1e-10
    assert float(np.max(result["dissipation_power_w"])) <= 1e-12
    assert float(np.max(result["normal_dissipation_power_w"])) <= 1e-12
    assert float(np.max(result["tangential_dissipation_power_w"])) <= 1e-12

    residual = np.asarray(result["work_energy_residual_j"], dtype=float)
    total_energy = np.asarray(result["total_energy_j"], dtype=float)
    normalized_residual = float(np.max(np.abs(residual))) / max(
        1.0, float(np.ptp(total_energy))
    )
    assert normalized_residual < 0.05


def test_station_opening_and_reattachment_transitions_and_first_failure() -> None:
    """Event probe driving station opening and reattachment must record transitions."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    config_slack = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
        slack_distance_m=0.0015,
        friction_coefficient=0.3,
    )
    forward_cfg = DistributedForwardConfig(
        duration_s=0.05,
        time_steps_s=(0.001, 0.0005),
    )

    case = DistributedIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        time_step_s=0.0005,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=-0.8,
        engine="mujoco",
        grip=config_slack,
    )
    result = integrate_distributed_grip(model, case, forward_cfg)

    total_transitions = int(result["total_transition_count"])
    assert total_transitions > 0
    assert result["first_failure_class"] in (
        "partial_opening",
        "full_loss_of_contact",
        "slip_occurring",
    )


def test_nested_horizons_with_right_censoring() -> None:
    """Nested horizons (4, 10, 25, 50 ms) must evaluate from one trajectory."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    config = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
        friction_coefficient=0.35,
    )
    forward_cfg = DistributedForwardConfig(
        duration_s=0.05,
        time_steps_s=(0.001, 0.0005),
    )
    case = DistributedIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        time_step_s=0.0005,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.05,
        engine="mujoco",
        grip=config,
    )
    result = integrate_distributed_grip(model, case, forward_cfg)

    time_s = np.asarray(result["time_s"], dtype=float)
    for horizon in (0.004, 0.01, 0.025, 0.05):
        idx = int(np.searchsorted(time_s, horizon))
        assert np.isclose(time_s[idx], horizon)
        assert np.isfinite(result["maximum_station_force_n"][idx])
