"""Generate the published launch-monitor analysis contract v2 JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from src.shared.python.launch_monitor import (
    contract_v2_json_schema,
    dataset_job_contract_json_schema,
    strokes_gained_contract_json_schema,
)


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


if __name__ == "__main__":
    main()
