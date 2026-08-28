"""Prove checkpoint corruption fails closed without mutating campaign evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    load_available_checkpoints,
    plan_sha256,
)

SCHEMA = "articulated-structural-factorial-corruption-audit/1.0.0"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_REJECTION = "completed checkpoint parity sidecar is missing or corrupt"


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_checkpoint_corruption(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    audit_revision: str,
) -> dict[str, object]:
    """Corrupt a copied sidecar and retain the expected rejection evidence."""

    if _SHA40.fullmatch(audit_revision) is None:
        raise ValueError("audit_revision must be a lowercase 40-character SHA")
    checkpoints = load_available_checkpoints(
        plan=plan, launch=launch, checkpoint_dir=checkpoint_dir
    )
    try:
        source = next(row for row in checkpoints if row.status == "completed")
    except StopIteration as exc:
        raise ValueError("corruption audit requires one completed checkpoint") from exc
    source_sidecar = source.path.with_suffix(".npz")
    original_json_sha = _sha256(source.path)
    original_npz_sha = _sha256(source_sidecar)
    with tempfile.TemporaryDirectory(
        prefix="structural-corruption-audit-"
    ) as temporary:
        copied_dir = Path(temporary)
        copied_json = copied_dir / source.path.name
        copied_npz = copied_dir / source_sidecar.name
        shutil.copy2(source.path, copied_json)
        shutil.copy2(source_sidecar, copied_npz)
        corrupted = bytearray(copied_npz.read_bytes())
        if not corrupted:
            raise ValueError("completed parity sidecar must be nonempty")
        index = len(corrupted) // 2
        corrupted[index] ^= 0x01
        copied_npz.write_bytes(corrupted)
        try:
            load_available_checkpoints(
                plan=plan, launch=launch, checkpoint_dir=copied_dir
            )
        except ValueError as exc:
            observed_rejection = str(exc)
        else:
            raise AssertionError("planted checkpoint corruption was not rejected")
    source_unchanged = bool(
        _sha256(source.path) == original_json_sha
        and _sha256(source_sidecar) == original_npz_sha
    )
    if observed_rejection != _EXPECTED_REJECTION:
        raise AssertionError(
            "planted corruption produced an unexpected rejection boundary: "
            + observed_rejection
        )
    if not source_unchanged:
        raise RuntimeError("corruption audit mutated source campaign evidence")
    return {
        "schema_version": SCHEMA,
        "classification": "corruption_killswitch_not_scientific_evidence",
        "identity": {
            "plan_sha256": plan_sha256(plan),
            "execution_revision": launch.get("execution_revision"),
            "audit_revision": audit_revision,
        },
        "source_checkpoint": {
            "case_key": source.case.case_key,
            "json_file": source.path.name,
            "json_sha256": original_json_sha,
            "parity_sidecar_file": source_sidecar.name,
            "parity_sidecar_sha256": original_npz_sha,
        },
        "sentinel": {
            "operation": "flip_middle_byte_in_copied_parity_sidecar",
            "source_checkpoint_unchanged": source_unchanged,
            "observed_rejection": observed_rejection,
            "passes": True,
        },
    }


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the copied-checkpoint sentinel and atomically retain its audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--audit-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = _mapping(json.loads(args.plan.read_text(encoding="utf-8")), name="plan")
    launch = _mapping(
        json.loads(args.launch.read_text(encoding="utf-8")), name="launch"
    )
    result = audit_checkpoint_corruption(
        plan=plan,
        launch=launch,
        checkpoint_dir=args.checkpoint_dir,
        audit_revision=args.audit_revision,
    )
    _write_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_checkpoint_corruption"]
