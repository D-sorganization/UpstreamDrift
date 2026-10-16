"""TDD: the receipt hashes the pre-seeding base spec (#12820).

Foot-progression seeding (OSV-4 #11730, R3 #12683) rewrites the document's
``address_seed_deg`` before calibration. The receipt's ``base_spec_sha256``
must still chain to the committed spec file, and the seeding must be
reproducible from the receipt's explicit ``foot_progression_mode`` and
``foot_seed_deg`` fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.pipeline.receipt import (
    BaseSpecProvenance,
    GroundSupportReceiptInputs,
    build_ground_support_receipt,
)
from src.shared.python.motion_matching.pipeline.receipt_provenance import (
    ProvenanceChainInputs,
    validate_receipt_provenance_chain,
)
from src.shared.python.motion_matching.pipeline.receipt_schema import (
    validate_receipt,
)

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs" / "development" / "full_body_models"
DRIVER_SPEC = DOCS / "full_body_spec_anthro_driver.json"
DRIVER_RECEIPT = DOCS / "evidence" / "ground_support" / "anthro_driver" / "receipt.json"
SEED = {"hip_rotation_r": -31.25, "hip_rotation_l": 12.5}


def _committed() -> dict[str, Any]:
    return json.loads(DRIVER_SPEC.read_text(encoding="utf-8"))


def _seeded(document: dict[str, Any]) -> dict[str, Any]:
    """What ``seed_document_feet`` does to the document: new address seeds."""
    out = dict(document)
    out["address_seed_deg"] = {**(document.get("address_seed_deg") or {}), **SEED}
    return out


def _receipt(base_spec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    lane = MagicMock()
    lane.bounds = {}
    lane.labels = ("m1",)
    lane.ground_cal.height_m = 0.0
    lane.ground_cal.lowest_marker_height_m = 0.03
    lane.stance = [("heel_r",)] * 4
    c3d = Path(__file__)
    with patch(
        "src.shared.python.motion_matching.pipeline.receipt.posture_summary",
        return_value={"spine_bend_deg": {}},
    ):
        return build_ground_support_receipt(
            GroundSupportReceiptInputs(
                base_spec=base_spec,
                spec_path=DRIVER_SPEC,
                scaled_path=Path("scaled.json"),
                hipcal_path=Path("hipcal.json"),
                recalibrate_upper=False,
                anthropometric=None,
                qualification_note="test",
                spec_bytes=b"{}",
                hip_report={},
                candidate_bytes=b"{}",
                c3d_path=c3d,
                capture_name="driver",
                lane=lane,
                address_report={},
                ik_report={},
                dynamics_report={},
                kin=MagicMock(),
                q_ref=np.zeros((300, 4)),
                elapsed_s=1.0,
                validate=False,
                **kwargs,
            )
        )


def test_style_seeded_receipt_chains_to_the_committed_spec(tmp_path: Path) -> None:
    """Acceptance 1: a style-feet receipt passes the provenance chain."""
    committed = _committed()
    seeded = _seeded(committed)
    assert canonical_sha256(seeded) != canonical_sha256(committed)
    provenance = BaseSpecProvenance.of(committed, "style", SEED)
    receipt = _receipt(seeded, provenance=provenance)

    assert receipt["base_spec_sha256"] == canonical_sha256(committed)
    assert receipt["foot_progression_mode"] == "style"
    assert receipt["foot_seed_deg"] == SEED

    final = {"de_leva_table_sha256": committed["de_leva_table_sha256"], "v": 1}
    final_bytes = (json.dumps(final, sort_keys=True) + "\n").encode("utf-8")
    final_path = tmp_path / "final.json"
    final_path.write_bytes(final_bytes)
    chained = {
        **receipt,
        "de_leva_table_sha256": committed["de_leva_table_sha256"],
        "spec_sha256": __import__("hashlib").sha256(final_bytes).hexdigest(),
        "spec_canonical_sha256": canonical_sha256(final),
    }
    validate_receipt_provenance_chain(
        ProvenanceChainInputs(
            receipt=chained,
            base_spec_path=DRIVER_SPEC,
            final_spec_path=final_path,
        )
    )


def test_seeding_is_reproducible_from_the_receipt_fields() -> None:
    """Acceptance 2: committed spec + recorded seed rebuild the seeded document."""
    committed = _committed()
    receipt = _receipt(
        _seeded(committed), provenance=BaseSpecProvenance.of(committed, "style", SEED)
    )
    rebuilt = dict(committed)
    rebuilt["address_seed_deg"] = {
        **(committed.get("address_seed_deg") or {}),
        **receipt["foot_seed_deg"],
    }
    assert canonical_sha256(rebuilt) == canonical_sha256(_seeded(committed))


def test_without_provenance_the_receipt_hashes_its_base_spec() -> None:
    """Legacy callers keep today's behaviour and record feet off."""
    committed = _committed()
    receipt = _receipt(committed)
    assert receipt["base_spec_sha256"] == canonical_sha256(committed)
    assert receipt["foot_progression_mode"] == "off"
    assert receipt["foot_seed_deg"] is None


def test_off_provenance_matches_the_legacy_hash() -> None:
    """Feet off: no seeding, so committed receipts keep their hash."""
    committed = _committed()
    off = BaseSpecProvenance.of(committed)
    assert off.receipt_fields() == {
        "base_spec_sha256": canonical_sha256(committed),
        "foot_progression_mode": "off",
        "foot_seed_deg": None,
    }


@pytest.mark.parametrize(
    ("mode", "seed", "match"),
    [
        ("sideways", SEED, "foot_progression_mode"),
        ("off", SEED, "off"),
        ("style", None, "seed"),
        ("style", {}, "seed"),
        ("style", {"hip_rotation_r": math.nan}, "finite"),
        ("style", {"hip_rotation_r": True}, "finite"),
    ],
)
def test_provenance_rejects_inconsistent_fields(
    mode: str, seed: dict[str, float] | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BaseSpecProvenance.of(_committed(), mode, seed)


def test_provenance_rejects_a_non_mapping_document() -> None:
    with pytest.raises(TypeError, match="document"):
        BaseSpecProvenance.of([1, 2])  # type: ignore[arg-type]


def test_committed_receipts_without_the_fields_still_validate() -> None:
    """Acceptance 3: receipts written before #12820 still validate."""
    doc = json.loads(DRIVER_RECEIPT.read_text(encoding="utf-8"))
    doc.pop("foot_progression_mode", None)
    doc.pop("foot_seed_deg", None)
    receipt = validate_receipt(doc)
    assert receipt.foot_progression_mode is None
    assert receipt.foot_seed_deg is None


def test_schema_accepts_and_checks_the_new_fields() -> None:
    doc = json.loads(DRIVER_RECEIPT.read_text(encoding="utf-8"))
    doc["foot_progression_mode"] = "style"
    doc["foot_seed_deg"] = SEED
    receipt = validate_receipt(doc)
    assert receipt.foot_progression_mode == "style"
    assert receipt.foot_seed_deg == SEED
    doc["foot_progression_mode"] = "sideways"
    with pytest.raises(ValueError):
        validate_receipt(doc)


@pytest.mark.slow
@pytest.mark.live_simulation
@pytest.mark.timeout(900)
def test_canonical_style_run_records_the_committed_hash(tmp_path: Path) -> None:
    """End to end: the default (style) calibrated run seeds the feet but its
    provenance still hashes the committed driver spec."""
    from src.shared.python.motion_matching.pipeline import cli

    args = cli.build_parser().parse_args(
        ["--spec", str(DRIVER_SPEC), "--capture", "driver", "--out", str(tmp_path)]
    )
    assert args.foot_progression == "style"
    run = cli.calibrate_run(args)
    committed = _committed()
    assert run.provenance is not None
    fields = run.provenance.receipt_fields()
    assert fields["base_spec_sha256"] == canonical_sha256(committed)
    assert fields["foot_progression_mode"] == "style"
    rebuilt = dict(committed)
    rebuilt["address_seed_deg"] = {
        **(committed.get("address_seed_deg") or {}),
        **fields["foot_seed_deg"],
    }
    assert canonical_sha256(rebuilt) == canonical_sha256(run.base_spec)
    assert canonical_sha256(run.base_spec) != fields["base_spec_sha256"]
