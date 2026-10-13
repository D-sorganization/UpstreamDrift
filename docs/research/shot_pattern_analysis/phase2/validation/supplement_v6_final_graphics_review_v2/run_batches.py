"""Run all 36 qualified final-figure inventory batches sequentially."""

import hashlib
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


python = "/tmp/shot-pattern-validation-venv-20261009/bin/python"  # nosec B108
main = Path("/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift")
producer = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift"
)
phase = main / "docs/research/shot_pattern_analysis/phase2"
output = phase / "validation/supplement_v6_final_graphics_review_v2"
harness = phase / "validation/v6_output_review_harness_v1/review_batch.py"
expected_harness = "1db7be642e097377afeb44874687e41639a29c32c0440f13b3055eb3457d0bf6"
if sha(harness) != expected_harness:
    raise ValueError("qualified harness source changed")
if not (output / "strict_loader_receipt.json").exists():
    raise ValueError("strict public loader receipt missing")

schedule: list[tuple[str, str | None, int, int]] = []
for pair_kind, count in (
    ("across_lean", 56),
    ("curve_strength", 164),
    ("shaft_rotation_sd", 164),
):
    for offset in range(0, count, 12):
        schedule.append(("pairs", pair_kind, offset, min(12, count - offset)))
for offset in range(0, 26, 12):
    schedule.append(("cells", None, offset, min(12, 26 - offset)))
assert len(schedule) == 36

for group, kind, offset, limit in schedule:
    name = f"{kind or 'cells'}_{offset:03}"
    target = output / name
    if target.exists():
        raise ValueError(f"fresh output requirement violated: {target}")
    command = [
        python,
        str(harness),
        "--repo",
        str(producer),
        "--report",
        str(phase / "supplement_report_v1"),
        "--completion-receipt",
        str(phase / "validation/supplement_rendering_revision_v6/exit.json"),
        "--start-receipt",
        str(phase / "validation/supplement_rendering_revision_v6/start.json"),
        "--start-sha256",
        "ab2d3b97b8be26a0fbe4bfa7f2fabaccefb8de602493f76a6d07d2682595bfc0",
        "--expected-numeric",
        str(
            phase / "validation/protocol_name_recovery_v6/expected_numeric_hashes.json"
        ),
        "--output",
        str(target),
        "--group",
        group,
        "--offset",
        str(offset),
        "--limit",
        str(limit),
    ]
    if kind:
        command.extend(("--kind", kind))
    receipt: dict[str, object] = {
        "name": name,
        "command": command,
        "harness_sha256": expected_harness,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    command_path = output / f"{name}_command.json"
    command_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = subprocess.run(command, cwd=main, capture_output=True, text=True)
    receipt.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "batch_inventory_sha256": sha(target / "batch_inventory.json")
            if (target / "batch_inventory.json").exists()
            else None,
        }
    )
    command_path.write_text(json.dumps(receipt, indent=2) + "\n")
    sys.stdout.write(f"{name}: exit {result.returncode} {result.stdout.strip()}" + "\n")
    sys.stdout.flush()
    if result.returncode:
        raise SystemExit(result.returncode)
