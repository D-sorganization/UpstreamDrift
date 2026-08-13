"""Evidence-contract tests for the fixed-support GRF drift study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_grf_drift_study import (
    FIGURE_STEMS,
    build_study,
    write_study,
)

pytestmark = pytest.mark.scientific


def test_study_is_explicitly_a_model_benchmark() -> None:
    record, arrays = build_study()

    assert record["schema_version"] == "grf-drift-attribution-v1"
    assert record["evidence_class"] == "model_internal_falsification_benchmark"
    assert record["human_validation"] is False
    assert any("bilateral" in item for item in record["non_identifiable"])
    assert record["counterfactual_contract"]["zvcf_validity"] == (
        "autonomous holonomic constraint with velocity_bias(q, 0) = 0 "
        "and constraint_bias(q, 0) = 0"
    )
    assert record["net_force_consistency"]["status"] == "required_identity"
    assert arrays["total"].shape[1] == 2


def test_study_closes_all_pointwise_identities() -> None:
    record, arrays = build_study()

    np.testing.assert_allclose(
        arrays["total"],
        arrays["configuration"] + arrays["velocity"] + arrays["control"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        arrays["ztcf"], arrays["configuration"] + arrays["velocity"], atol=1e-10
    )
    np.testing.assert_allclose(
        arrays["zvcf"], arrays["configuration"] + arrays["control"], atol=1e-10
    )
    assert record["closure"]["max_abs_total_N"] < 1e-9
    assert record["closure"]["max_abs_zvcf_N"] < 1e-9


def test_drift_only_prediction_has_declared_falsification_metrics() -> None:
    record, _ = build_study()
    metrics = record["drift_only_prediction"]

    assert metrics["target"] == "modeled_total_support_reaction"
    assert metrics["predictor"] == "pointwise_ZTCF_support_reaction"
    assert len(metrics["rmse_N"]) == 2
    assert len(metrics["nrmse_planar_weight"]) == 2
    assert len(metrics["r_squared"]) == 2
    assert metrics["r_squared_definition"] == "1 - fixed_prediction_SSE / target_TSS"
    assert metrics["r_squared_is_squared_correlation"] is False
    assert len(metrics["bias_N"]) == 2
    assert np.all(np.isfinite(metrics["rmse_N"]))
    assert np.all(np.asarray(metrics["nrmse_planar_weight"]) >= 0.0)


def test_svg_evidence_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_study(first)
    write_study(second)

    for stem in FIGURE_STEMS:
        assert (first / "figures" / f"{stem}.svg").read_bytes() == (
            second / "figures" / f"{stem}.svg"
        ).read_bytes()
