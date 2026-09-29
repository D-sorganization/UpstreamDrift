"""Qualify the experimental intake contract without fabricating human evidence."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .experimental_protocol import ProtocolManifest, evaluate_dataset_readiness


ROOT = Path(__file__).resolve().parents[3]
ARTICLE_DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_readiness_record() -> dict[str, object]:
    """Return deterministic synthetic qualification evidence."""
    protocol_path = ARTICLE_DATA / "experimental_protocol_v1.json"
    fixture_path = ARTICLE_DATA / "experimental_protocol_dry_run.json"
    protocol = ProtocolManifest.from_json(protocol_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    readiness = evaluate_dataset_readiness(protocol, fixture)
    readiness_record = asdict(readiness)
    readiness_record["limitations"] = list(readiness.limitations)
    return {
        "schema_version": "experimental-falsification-readiness-v1",
        "protocol_id": protocol.protocol_id,
        "readiness": readiness_record,
        "human_data_evaluation": "not_executed",
        "claim_status": {
            prediction.prediction_id: "untested_no_governed_human_data"
            for prediction in protocol.predictions
        },
        "source_sha256": {
            str(protocol_path.relative_to(ROOT)).replace("\\", "/"): _sha256(
                protocol_path
            ),
            str(fixture_path.relative_to(ROOT)).replace("\\", "/"): _sha256(
                fixture_path
            ),
        },
        "interpretation_boundary": (
            "This artifact qualifies schema, split, modality, provenance, and "
            "fail-closed behavior only; it contains no human measurements."
        ),
    }


def main() -> None:
    """Write the committed readiness record."""
    output = ARTICLE_DATA / "experimental_protocol_readiness.json"
    output.write_text(
        json.dumps(build_readiness_record(), indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
