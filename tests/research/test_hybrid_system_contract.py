from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.hybrid_system_contract import (
    EXPECTED_TIER_IDS,
    HybridSystemContract,
)

pytestmark = pytest.mark.unit


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CONTRACT_PATH = DATA / "hybrid_system_contract_v1.json"


def _raw_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_registered_contract_covers_the_complete_model_ladder() -> None:
    contract = HybridSystemContract.from_json(CONTRACT_PATH, repository_root=ROOT)

    assert contract.schema_version == "proximal-distal-hybrid-system/v1"
    assert tuple(tier.tier_id for tier in contract.tiers) == EXPECTED_TIER_IDS
    assert (
        contract.issue == "https://github.com/D-sorganization/UpstreamDrift/issues/9027"
    )
    assert contract.inference_boundary.startswith("This contract does not establish")


def test_each_tier_declares_every_hybrid_system_component() -> None:
    contract = HybridSystemContract.from_json(CONTRACT_PATH, repository_root=ROOT)

    for tier in contract.tiers:
        assert tier.state_blocks
        assert tier.controls
        assert tier.algebraic_constraints
        assert tier.modes
        assert tier.guards
        assert tier.resets
        assert tier.impacts
        assert tier.actuator_dynamics
        assert tier.uncertain_event_surfaces
        assert tier.observables
        assert tier.limitations
        assert tier.falsifiers
        assert tier.comparison_blockers or tier.comparison_eligibility != "unavailable"


def test_available_tiers_bind_existing_sources_and_unavailable_tiers_fail_closed() -> (
    None
):
    contract = HybridSystemContract.from_json(CONTRACT_PATH, repository_root=ROOT)

    for tier in contract.tiers:
        if tier.authority_status == "unavailable":
            assert not tier.source_paths
            assert tier.comparison_eligibility == "unavailable"
            assert tier.comparison_blockers
        else:
            assert tier.source_paths
            assert all((ROOT / path).is_file() for path in tier.source_paths)


def test_guard_reset_and_uncertainty_references_fail_closed() -> None:
    raw = _raw_contract()
    broken = deepcopy(raw)
    broken["tiers"][1]["resets"][0]["guard_id"] = "missing-guard"
    with pytest.raises(ValueError, match="unknown guard"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)

    broken = deepcopy(raw)
    broken["tiers"][1]["guards"][0]["uncertainty_id"] = "missing-uncertainty"
    with pytest.raises(ValueError, match="unknown uncertainty"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)


def test_actuator_control_and_impact_references_fail_closed() -> None:
    raw = _raw_contract()
    broken = deepcopy(raw)
    broken["tiers"][1]["controls"][0]["actuator_id"] = "missing-actuator"
    with pytest.raises(ValueError, match="unknown actuator"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)

    broken = deepcopy(raw)
    broken["tiers"][1]["actuator_dynamics"][0]["state_block_ids"] = ["missing-state"]
    with pytest.raises(ValueError, match="unknown state block"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)

    broken = deepcopy(raw)
    broken["tiers"][1]["impacts"][0]["guard_id"] = "missing-guard"
    with pytest.raises(ValueError, match="unknown guard"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)


def test_numerical_contracts_reject_invalid_bounds_and_tolerances() -> None:
    raw = _raw_contract()
    broken = deepcopy(raw)
    broken["tiers"][1]["controls"][0]["bounds"] = [1.0, -1.0]
    with pytest.raises(ValueError, match="finite and ordered"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)

    broken = deepcopy(raw)
    tolerance_name = next(iter(broken["numerical_tolerances"]))
    broken["numerical_tolerances"][tolerance_name] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)


def test_duplicate_identifiers_and_source_escape_fail_closed() -> None:
    raw = _raw_contract()
    broken = deepcopy(raw)
    broken["tiers"][0]["modes"].append(deepcopy(broken["tiers"][0]["modes"][0]))
    with pytest.raises(ValueError, match="mode_id values must be unique"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)

    broken = deepcopy(raw)
    broken["tiers"][0]["source_paths"] = ["../outside.py"]
    with pytest.raises(ValueError, match="repository-relative"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)


def test_unavailable_components_require_specific_reasons() -> None:
    raw = _raw_contract()
    broken = deepcopy(raw)
    broken["tiers"][6]["controls"][0]["reason"] = ""
    with pytest.raises(ValueError, match="reason must be non-empty"):
        HybridSystemContract.from_dict(broken, repository_root=ROOT)


def test_contract_is_canonical_json() -> None:
    raw = _raw_contract()
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert _mapping_keys_are_sorted(raw)


def _mapping_keys_are_sorted(value: object) -> bool:
    if isinstance(value, dict):
        return list(value) == sorted(value) and all(
            _mapping_keys_are_sorted(item) for item in value.values()
        )
    if isinstance(value, list):
        return all(_mapping_keys_are_sorted(item) for item in value)
    return True
