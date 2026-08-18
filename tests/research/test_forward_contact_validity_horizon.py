"""Tests for the cross-engine closed-state validity-horizon evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.forward_contact_validity_horizon import (
    HorizonStudyConfig,
    registered_variants,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def test_horizon_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="increasing"):
        HorizonStudyConfig(horizons_s=(0.010, 0.004))
    with pytest.raises(ValueError, match="bracket"):
        HorizonStudyConfig(low_factor=1.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        HorizonStudyConfig(energy_closure_limit=1.0)


def test_registered_variants_are_one_factor_at_a_time() -> None:
    variants = registered_variants(HorizonStudyConfig())
    assert len(variants) == 10
    assert variants[0].variant_id == "nominal"
    assert variants[-1].variant_id == "driver_off"
    assert sum(not variant.driver_enabled for variant in variants) == 1
    assert len({variant.variant_id for variant in variants}) == len(variants)


def test_committed_horizon_evidence_is_complete_and_bounded() -> None:
    record = json.loads(
        (DATA_DIR / "forward_contact_validity_horizon.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "forward-contact-validity-horizon/v1"
    assert record["design"]["evaluated_trace_count"] == 1080
    assert record["design"]["evaluated_horizon_case_count"] == 2160
    assert len(record["cases"]) == 2160
    assert len(record["results"]["variants"]) == 10
    assert record["claim_boundary"]["articulated_anatomy"] == "not_established"
    for row in record["cases"]:
        assert row["horizon_s"] in (0.004, 0.010, 0.025, 0.050)
        assert isinstance(row["all_gates_passed"], bool)
        assert all(np.isfinite(value) for value in row["observed_metrics"].values())
        assert all(
            np.isfinite(value) for value in row["energy_closure_by_engine"].values()
        )
    for path, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_committed_horizon_arrays_match_the_registered_axes() -> None:
    with np.load(DATA_DIR / "forward_contact_validity_horizon.npz") as archive:
        assert archive["result_matrix"].shape == (10, 18, 3, 4, 8)
        assert archive["horizons_s"].tolist() == [0.004, 0.010, 0.025, 0.050]
        assert archive["phase_indices"].tolist() == [0, 6, 12]
        assert archive["variant_ids"].shape == (10,)
        assert np.all(np.isfinite(archive["result_matrix"]))
