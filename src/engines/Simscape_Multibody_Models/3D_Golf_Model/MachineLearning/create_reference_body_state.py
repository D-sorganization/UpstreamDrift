"""Create a reference body-state CSV for club torque sequence optimization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "processed" / "club_direct_dynamics.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "reference_body_state.csv"


def _input_columns(dataset: Path) -> list[str]:
    metadata = pq.ParquetFile(dataset).schema_arrow.metadata or {}
    return json.loads(metadata[b"input_columns"].decode("utf-8"))


def create_reference_body_state(dataset: Path, output: Path, row_index: int) -> None:
    input_columns = _input_columns(dataset)
    table = pq.read_table(dataset, columns=input_columns)
    frame = table.to_pandas()
    if row_index < 0 or row_index >= len(frame):
        raise ValueError(
            f"row_index {row_index} is outside dataset length {len(frame)}"
        )

    reference = frame.iloc[[row_index]].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output, index=False)
    LOGGER.info("Wrote reference body state to %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    create_reference_body_state(
        dataset=args.dataset,
        output=args.output,
        row_index=args.row_index,
    )


if __name__ == "__main__":
    main()
