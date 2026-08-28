"""Independent committed-evidence checks for articulated attribution (#9151)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_committed_attribution_evidence_is_complete_and_bounded() -> None:
    record = json.loads(
        (DATA / "articulated_drift_contact_attribution.json").read_text("utf-8")
    )

    assert record["schema_version"] == "articulated-drift-contact-attribution/v1"
    assert record["design"]["state_count"] == 234
    assert record["design"]["forward_steps"] == 0
    assert record["design"]["applied_input"] == "exactly_zero_in_baseline"
    assert record["results"]["failed_engine_state_count"] == 0
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["claim_boundary"]["forward_persistence_impulse_or_work"] == (
        "not_executed"
    )
    assert record["claim_boundary"]["human_transfer_or_strategy"] == "untested"


def test_committed_attribution_arrays_recompute_signed_headlines() -> None:
    with np.load(DATA / "articulated_drift_contact_attribution.npz") as arrays:
        assert arrays["mass_metric_acceleration_share"].shape == (18, 13, 2, 4)
        assert arrays["generalized_power_contribution_w"].shape == (18, 13, 2, 4)
        assert arrays["all_gates_passed"].shape == (18, 13, 2)
        assert np.all(arrays["all_gates_passed"])
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key

        shares = arrays["mass_metric_acceleration_share"]
        powers = arrays["generalized_power_contribution_w"]
        names = arrays["contribution_names"].tolist()
        configuration = names.index("configuration")
        velocity = names.index("velocity")
        contact = names.index("contact")
        active = names.index("active")

        assert np.min(shares[..., configuration]) == pytest.approx(0.7552369464)
        assert np.max(shares[..., configuration]) == pytest.approx(0.9095865987)
        assert np.min(shares[..., contact]) == pytest.approx(0.0940588965)
        assert np.max(shares[..., contact]) == pytest.approx(0.2349858343)
        assert np.min(shares[..., velocity]) < 0.0 < np.max(shares[..., velocity])
        assert np.all(shares[..., contact] > 0.0)
        assert np.all(powers[..., contact] < 0.0)
        assert np.all(shares[..., active] == 0.0)
        assert np.all(powers[..., active] == 0.0)
