"""Unit tests for physically constrained contact identification and identifiability (MS-20 #10335)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_identification import (
    MAX_CONDITION_NUMBER,
    STATUS_CALIBRATED_IDENTIFIABLE,
    STATUS_ILL_CONDITIONED,
    STATUS_RANK_DEFICIENT,
    STATUS_SYNTHETIC_SENSITIVITY_ONLY,
    ContactGridConfig,
    ContactPrior,
    IdentifiabilityAnalysis,
    PhaseContactMetrics,
    analyze_identifiability,
    classify_identifiability,
    compute_momentum_balance_residual,
    evaluate_contact_phases,
    main,
    run_contact_identification,
)
from src.shared.python.motion_matching.contact_law import ContactParameters


@pytest.mark.unit
def test_contact_prior_validation() -> None:
    prior = ContactPrior(
        nominal=50000.0,
        min_val=10000.0,
        max_val=200000.0,
        unit="N/m",
        description="Turf stiffness",
    )
    assert prior.nominal == 50000.0
    assert prior.contains(50000.0)
    assert not prior.contains(5000.0)
    assert not prior.contains(300000.0)

    with pytest.raises(ValueError, match="min_val must be <= max_val"):
        ContactPrior(
            nominal=10.0, min_val=20.0, max_val=5.0, unit="N/m", description=""
        )


@pytest.mark.unit
def test_contact_grid_config_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "test_grid.json"
    data = {
        "schema_version": "1.0.0",
        "name": "test_grid",
        "priors": {
            "stiffness_n_m": {
                "nominal": 50000.0,
                "min": 10000.0,
                "max": 200000.0,
                "unit": "N/m",
                "description": "",
            },
            "dissipation_s_m": {
                "nominal": 2.0,
                "min": 0.1,
                "max": 5.0,
                "unit": "s/m",
                "description": "",
            },
            "static_friction": {
                "nominal": 0.8,
                "min": 0.4,
                "max": 1.5,
                "unit": "ratio",
                "description": "",
            },
            "dynamic_friction": {
                "nominal": 0.6,
                "min": 0.2,
                "max": 1.2,
                "unit": "ratio",
                "description": "",
            },
            "viscous_friction": {
                "nominal": 0.05,
                "min": 0.0,
                "max": 0.5,
                "unit": "s/m",
                "description": "",
            },
            "transition_velocity_m_s": {
                "nominal": 0.02,
                "min": 0.005,
                "max": 0.1,
                "unit": "m/s",
                "description": "",
            },
        },
        "grid": {
            "stiffness_n_m": [50000.0, 100000.0],
            "dissipation_s_m": [1.0, 2.0],
            "static_friction": [0.8],
            "dynamic_friction": [0.6],
            "viscous_friction": [0.05],
            "transition_velocity_m_s": [0.02],
        },
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")

    cfg = ContactGridConfig.from_file(config_path)
    assert cfg.name == "test_grid"
    assert len(cfg.combinations()) == 4
    for p in cfg.combinations():
        assert isinstance(p, ContactParameters)


@pytest.mark.unit
def test_identifiability_analysis_synthetic_jacobian() -> None:
    # 3 parameters: p1, p2, p3 where p3 has near-zero sensitivity (non-identifiable)
    n_obs = 100
    param_names = ("stiffness", "dissipation", "viscous")
    J = np.zeros((n_obs, 3))
    J[:, 0] = np.linspace(1.0, 2.0, n_obs)
    J[:, 1] = np.sin(np.linspace(0, np.pi, n_obs))
    J[:, 2] = 1e-9 * np.random.default_rng(42).standard_normal(n_obs)  # near zero

    weights = np.ones(n_obs)
    analysis = analyze_identifiability(J, weights, param_names)

    assert isinstance(analysis, IdentifiabilityAnalysis)
    assert len(analysis.eigenvalues) == 3
    assert analysis.condition_number > 1e6
    assert analysis.identifiable_rank == 2
    assert "viscous" in analysis.non_identifiable_directions[0]


@pytest.mark.unit
def test_evaluate_contact_phases() -> None:
    n_frames = 200
    times = np.linspace(0.0, 2.0, n_frames)
    forces_n = np.zeros((n_frames, 3))
    forces_n[:, 2] = 780.0  # 1 BW normal force
    penetrations_m = np.ones(n_frames) * 0.005  # 5 mm
    slips_m_s = np.zeros(n_frames)
    cop_inside = np.ones(n_frames, dtype=bool)
    body_mass_kg = 78.0

    metrics = evaluate_contact_phases(
        times=times,
        forces_n=forces_n,
        penetrations_m=penetrations_m,
        slips_m_s=slips_m_s,
        cop_inside=cop_inside,
        body_mass_kg=body_mass_kg,
    )

    for phase in ("address", "backswing", "downswing", "impact", "follow_through"):
        assert phase in metrics
        pm = metrics[phase]
        assert isinstance(pm, PhaseContactMetrics)
        assert pm.mean_normal_force_n == pytest.approx(780.0)
        assert pm.mean_penetration_m == pytest.approx(0.005)
        assert pm.cop_inside_polygon_fraction == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_momentum_balance_residual() -> None:
    n_frames = 50
    dt = 0.01
    mass = 78.0
    # Steady state standing: CoM stationary, vertical force = m*g
    com_pos = np.zeros((n_frames, 3))
    com_pos[:, 2] = 0.85
    total_grf = np.zeros((n_frames, 3))
    total_grf[:, 2] = mass * 9.80665

    res = compute_momentum_balance_residual(com_pos, total_grf, mass=mass, dt=dt)
    assert res["linear_momentum_rmse_n"] == pytest.approx(0.0, abs=1e-3)
    assert res["max_force_imbalance_n"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_run_contact_identification_pipeline(tmp_path: Path) -> None:
    # Setup mock grid and run directory
    config_path = tmp_path / "contact_grid.json"
    grid_dict = {
        "schema_version": "1.0.0",
        "name": "mock_grid",
        "priors": {
            "stiffness_n_m": {
                "nominal": 50000.0,
                "min": 20000.0,
                "max": 100000.0,
                "unit": "N/m",
                "description": "",
            },
            "dissipation_s_m": {
                "nominal": 2.0,
                "min": 0.5,
                "max": 4.0,
                "unit": "s/m",
                "description": "",
            },
            "static_friction": {
                "nominal": 0.8,
                "min": 0.4,
                "max": 1.2,
                "unit": "ratio",
                "description": "",
            },
            "dynamic_friction": {
                "nominal": 0.6,
                "min": 0.3,
                "max": 1.0,
                "unit": "ratio",
                "description": "",
            },
            "viscous_friction": {
                "nominal": 0.05,
                "min": 0.01,
                "max": 0.2,
                "unit": "s/m",
                "description": "",
            },
            "transition_velocity_m_s": {
                "nominal": 0.02,
                "min": 0.01,
                "max": 0.05,
                "unit": "m/s",
                "description": "",
            },
        },
        "grid": {
            "stiffness_n_m": [50000.0],
            "dissipation_s_m": [2.0],
            "static_friction": [0.8],
            "dynamic_friction": [0.6],
            "viscous_friction": [0.05],
            "transition_velocity_m_s": [0.02],
        },
    }
    config_path.write_text(json.dumps(grid_dict), encoding="utf-8")

    out_dir = tmp_path / "evidence" / "contact_id"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_contact_identification(
        grid_config_path=config_path,
        out_dir=out_dir,
        quick=True,
    )

    assert result["calibrated_parameters"]["stiffness_n_m"] == pytest.approx(50000.0)
    assert result["status"] == STATUS_SYNTHETIC_SENSITIVITY_ONLY
    assert result["schema_version"] == "matched-swing-contact-id/v2"
    assert result["kind"] == "synthetic-contact-sweep"
    assert (out_dir / "receipt.json").exists()

    # Verify parquet file is readable if pyarrow is installed
    try:
        import pyarrow.parquet as pq

        assert (out_dir / "sweep.parquet").exists()
        table = pq.read_table(out_dir / "sweep.parquet")
        assert table.num_rows >= 1
        assert "stiffness_n_m" in table.column_names
        assert "condition_number" in table.column_names
    except ImportError:
        pass


@pytest.mark.unit
def test_classify_identifiability_synthetic_gives_synthetic_sensitivity_only() -> None:
    """Synthetic data (measured=False) must never be CALIBRATED_IDENTIFIABLE (#10960 P0-7)."""
    # Even with full rank and low condition number, synthetic simulation yields SYNTHETIC_SENSITIVITY_ONLY
    status = classify_identifiability(
        rank=6,
        n_params=6,
        condition_number=10.0,
        measured=False,
    )
    assert status == STATUS_SYNTHETIC_SENSITIVITY_ONLY
    assert status != STATUS_CALIBRATED_IDENTIFIABLE


@pytest.mark.unit
def test_classify_identifiability_rank_deficient_is_not_calibrated_identifiable() -> (
    None
):
    """Rank-deficient identification is not CALIBRATED_IDENTIFIABLE (#10960 P0-7)."""
    status = classify_identifiability(
        rank=4,
        n_params=6,
        condition_number=100.0,
        measured=True,
    )
    assert status != STATUS_CALIBRATED_IDENTIFIABLE
    assert status == STATUS_RANK_DEFICIENT


@pytest.mark.unit
def test_classify_identifiability_ill_conditioned_is_not_calibrated_identifiable() -> (
    None
):
    """Ill-conditioned identification is not CALIBRATED_IDENTIFIABLE (#10960 P0-7)."""
    # Above threshold limit
    status_high = classify_identifiability(
        rank=6,
        n_params=6,
        condition_number=MAX_CONDITION_NUMBER + 100.0,
        measured=True,
    )
    assert status_high != STATUS_CALIBRATED_IDENTIFIABLE
    assert status_high == STATUS_ILL_CONDITIONED

    # Infinite condition number
    status_inf = classify_identifiability(
        rank=6,
        n_params=6,
        condition_number=float("inf"),
        measured=True,
    )
    assert status_inf != STATUS_CALIBRATED_IDENTIFIABLE
    assert status_inf == STATUS_ILL_CONDITIONED


@pytest.mark.unit
def test_classify_identifiability_measured_well_conditioned_is_calibrated_identifiable() -> (
    None
):
    """Measured observations with full rank and well-conditioned FIM qualify as CALIBRATED_IDENTIFIABLE."""
    status = classify_identifiability(
        rank=6,
        n_params=6,
        condition_number=50.0,
        measured=True,
        condition_limit=MAX_CONDITION_NUMBER,
    )
    assert status == STATUS_CALIBRATED_IDENTIFIABLE


@pytest.mark.unit
def test_classify_identifiability_preconditions() -> None:
    """DbC preconditions on classify_identifiability inputs."""
    with pytest.raises(ValueError, match="rank must be a non-negative integer"):
        classify_identifiability(
            rank=-1, n_params=6, condition_number=10.0, measured=True
        )

    with pytest.raises(ValueError, match="n_params must be a positive integer"):
        classify_identifiability(
            rank=0, n_params=0, condition_number=10.0, measured=True
        )

    with pytest.raises(ValueError, match="rank cannot exceed n_params"):
        classify_identifiability(
            rank=7, n_params=6, condition_number=10.0, measured=True
        )

    with pytest.raises(ValueError, match="condition_number must be non-negative"):
        classify_identifiability(
            rank=6, n_params=6, condition_number=-1.0, measured=True
        )

    with pytest.raises(ValueError, match="condition_limit must be positive"):
        classify_identifiability(
            rank=6,
            n_params=6,
            condition_number=10.0,
            measured=True,
            condition_limit=-5.0,
        )


@pytest.mark.unit
def test_run_contact_identification_run_dir_raises(tmp_path: Path) -> None:
    """Providing run_dir must fail closed with NotImplementedError (#10960 P0-7)."""
    config_path = tmp_path / "contact_grid.json"
    grid_dict = {
        "schema_version": "1.0.0",
        "name": "mock_grid",
        "priors": {
            "stiffness_n_m": {
                "nominal": 50000.0,
                "min": 20000.0,
                "max": 100000.0,
                "unit": "N/m",
            },
            "dissipation_s_m": {"nominal": 2.0, "min": 0.5, "max": 4.0, "unit": "s/m"},
            "static_friction": {
                "nominal": 0.8,
                "min": 0.4,
                "max": 1.2,
                "unit": "ratio",
            },
            "dynamic_friction": {
                "nominal": 0.6,
                "min": 0.3,
                "max": 1.0,
                "unit": "ratio",
            },
            "viscous_friction": {
                "nominal": 0.05,
                "min": 0.01,
                "max": 0.2,
                "unit": "s/m",
            },
            "transition_velocity_m_s": {
                "nominal": 0.02,
                "min": 0.01,
                "max": 0.05,
                "unit": "m/s",
            },
        },
        "grid": {
            "stiffness_n_m": [50000.0],
            "dissipation_s_m": [2.0],
            "static_friction": [0.8],
            "dynamic_friction": [0.6],
            "viscous_friction": [0.05],
            "transition_velocity_m_s": [0.02],
        },
    }
    config_path.write_text(json.dumps(grid_dict), encoding="utf-8")
    out_dir = tmp_path / "out"

    with pytest.raises(
        NotImplementedError,
        match=r"measured contact calibration not wired \(#10960 P0-7\)",
    ):
        run_contact_identification(
            grid_config_path=config_path,
            out_dir=out_dir,
            run_dir=tmp_path / "measured_run",
        )


@pytest.mark.unit
def test_cli_run_dir_flags_raise(tmp_path: Path) -> None:
    """CLI flags --run and --run-dir must fail closed instead of silently ignoring (#10960 P0-7)."""
    with pytest.raises(
        NotImplementedError,
        match=r"measured contact calibration not wired \(#10960 P0-7\)",
    ):
        main(["--run", "anthro_driver"])

    with pytest.raises(
        NotImplementedError,
        match=r"measured contact calibration not wired \(#10960 P0-7\)",
    ):
        main(["--run-dir", str(tmp_path / "run_data")])
