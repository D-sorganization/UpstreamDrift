"""Generate the published launch-monitor analysis contract v2 JSON Schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_monitor_conformance_fixture import (  # noqa: E402
    launch_monitor_conformance_bundle as _build_conformance_bundle,
)
from src.tools.launch_monitor_model import (
    AnalysisContextV2,
    PlayerCovariationRequestV1,
    PlayerCovariationScanRequestV1,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    analyze_player_covariation_v1,
    contract_v2_json_schema,
    dataset_job_contract_json_schema,
    longitudinal_session_contract_json_schema,
    launch_monitor_conformance_bundle_json_schema,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
    strokes_gained_contract_json_schema,
)


def launch_monitor_conformance_bundle():
    """Return the deterministic aggregate-only consumer conformance bundle."""

    return _build_conformance_bundle(Path(__file__).resolve().parents[1])


def _write_schema(destination: Path, schema: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing == schema:
            return
    destination.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def player_covariation_golden_fixture() -> dict[str, object]:
    """Return a deterministic synthetic consumer fixture and expected result."""

    records = [
        {
            "shot_id": f"shot-{index}",
            "source_id": "golden-source",
            "source_row": index,
            "player_id": "A" if index < 5 else "B",
            "face_angle": index if index < 5 else index + 5,
            "club_path": 4 - index if index < 5 else 19 - index,
            "ball_speed": 100 + 2 * index,
            "monitor_vendor": "TrackMan",
            "monitor_model": "golden-comparable",
            "software_version": "fixture-1",
        }
        for index in range(10)
    ]
    request = PlayerCovariationRequestV1(
        x_column="face_angle",
        y_column="club_path",
        player_column="player_id",
    )
    context = AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_id",
            evidence="Synthetic golden labels are stable by construction.",
        ),
        sources=(
            SourceFileReferenceV2(
                source_id="golden-source",
                file_sha256="3" * 64,
                rights_status="public_redistributable",
            ),
        ),
    )
    result = analyze_player_covariation_v1(
        pd.DataFrame.from_records(records), request, context=context
    )
    scan_request = PlayerCovariationScanRequestV1(
        player_column="player_id",
        numeric_columns=("face_angle", "club_path", "ball_speed"),
    )
    scan_result = scan_player_covariation_v1(
        pd.DataFrame.from_records(records), scan_request, context=context
    )
    return {
        "fixture_version": "launch-monitor-player-covariation-golden/1.0.0",
        "description": (
            "Synthetic aggregation-reversal fixture; no observed player data."
        ),
        "records": records,
        "request": request.model_dump(mode="json"),
        "scan_request": scan_request.model_dump(mode="json"),
        "context": context.model_dump(mode="json", exclude_none=True),
        "expected_result": result.model_dump(mode="json", exclude_none=False),
        "expected_scan_result": scan_result.model_dump(mode="json", exclude_none=False),
    }


def main() -> None:
    """Write the deterministic schema artifact from the Python authority."""

    root = Path(__file__).resolve().parents[1]
    contract_root = root / "docs" / "api" / "contracts"
    _write_schema(
        contract_root / "launch-monitor-analysis-v2.schema.json",
        contract_v2_json_schema(),
    )
    _write_schema(
        contract_root / "launch-monitor-strokes-gained-v1.schema.json",
        strokes_gained_contract_json_schema(),
    )
    _write_schema(
        contract_root / "launch-monitor-dataset-job-v1.schema.json",
        dataset_job_contract_json_schema(),
    )
    _write_schema(
        contract_root / "launch-monitor-player-covariation-v1.schema.json",
        player_covariation_contract_json_schema(),
    )
    _write_schema(
        contract_root / "fixtures" / "launch-monitor-player-covariation-v1.golden.json",
        player_covariation_golden_fixture(),
    )
    _write_schema(
        contract_root / "launch-monitor-longitudinal-session-v1.schema.json",
        longitudinal_session_contract_json_schema(),
    )
    _write_schema(
        contract_root / "launch-monitor-conformance-bundle-v1.schema.json",
        launch_monitor_conformance_bundle_json_schema(),
    )
    _write_schema(
        contract_root / "fixtures" / "launch-monitor-conformance-bundle-v1.golden.json",
        launch_monitor_conformance_bundle().model_dump(mode="json"),
    )


if __name__ == "__main__":
    main()
