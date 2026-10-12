"""Provenance gate for the cross-engine parity baseline (#12013).

A baseline labelled as the Simscape engine must have been *produced* by
Simscape. The producing engine is recorded by the producer (generator
receipt), never inferred from the label a consumer attaches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cross_engine.baseline_provenance import (
    BaselineProvenanceError,
    load_provenance,
    require_producing_engine,
)

pytestmark = [pytest.mark.unit, pytest.mark.cross_engine]


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_mujoco_produced_baseline_labelled_simscape_is_rejected(
    tmp_path: Path,
) -> None:
    receipt = _write(
        tmp_path / "p.json",
        {"producing_engine": "mujoco", "receipt": "gen_mujoco.py@abc"},
    )
    prov = load_provenance(receipt)
    with pytest.raises(BaselineProvenanceError, match="mujoco.*simscape"):
        require_producing_engine(prov, claimed_engine="simscape")


def test_matching_engine_is_accepted(tmp_path: Path) -> None:
    receipt = _write(
        tmp_path / "p.json",
        {"producing_engine": "Simscape", "receipt": "MATLAB R2025b, model sha256:x"},
    )
    prov = load_provenance(receipt)
    require_producing_engine(prov, claimed_engine="simscape")  # no raise


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(BaselineProvenanceError, match="no provenance"):
        load_provenance(tmp_path / "absent.json")


def test_provenance_without_producing_engine_is_rejected(tmp_path: Path) -> None:
    receipt = _write(tmp_path / "p.json", {"receipt": "hand label only"})
    with pytest.raises(BaselineProvenanceError, match="producing_engine"):
        load_provenance(receipt)


def test_committed_simscape_fixture_is_not_simscape_produced() -> None:
    """The committed fixture is a MuJoCo artefact; it must not pass as Simscape."""
    from tests.cross_engine.baseline_provenance import COMMITTED_PROVENANCE_PATH

    prov = load_provenance(COMMITTED_PROVENANCE_PATH)
    assert prov.producing_engine == "mujoco"
    with pytest.raises(BaselineProvenanceError):
        require_producing_engine(prov, claimed_engine="simscape")


def test_parity_harness_withholds_simscape_baseline() -> None:
    from tests.cross_engine.test_four_engine_parity import (
        load_simscape_golden_baseline,
    )

    with pytest.raises(BaselineProvenanceError):
        load_simscape_golden_baseline()
