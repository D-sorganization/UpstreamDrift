"""Continue qualified batches after the two completed v2 batches."""

import hashlib
import json
import sys
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


python = "/tmp/shot-pattern-validation-venv-20261009/bin/python"
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
for name in ("across_lean_000", "across_lean_012"):
    receipt = json.loads((output / f"{name}_command.json").read_text())
    if receipt.get("exit_code") != 0 or sha(
        output / name / "batch_inventory.json"
    ) != receipt.get("batch_inventory_sha256"):
        raise ValueError(f"previous completed batch is unqualified: {name}")

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
schedule = schedule[2:]

runner_start = {
    "pid": os.getpid(),
    "start_ticks": int(
        Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    ),
    "started_utc": utc(),
    "script_sha256": sha(Path(__file__)),
    "schedule": [
        {"group": g, "kind": k, "offset": o, "limit": n} for g, k, o, n in schedule
    ],
    "prior_partial": str(output / "across_lean_024"),
}
(output / "remaining_runner_start.json").write_text(
    json.dumps(runner_start, indent=2) + "\n"
)

try:
    for group, kind, offset, limit in schedule:
        name = f"{kind or 'cells'}_{offset:03}"
        if name == "across_lean_024":
            name += "_retry"
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
                phase
                / "validation/protocol_name_recovery_v6/expected_numeric_hashes.json"
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
        receipt = {
            "name": name,
            "command": command,
            "harness_sha256": expected_harness,
            "started_utc": utc(),
        }
        command_path = output / f"{name}_command.json"
        command_path.write_text(json.dumps(receipt, indent=2) + "\n")
        result = subprocess.run(command, cwd=main, capture_output=True, text=True)
        receipt.update(
            {
                "finished_utc": utc(),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "batch_inventory_sha256": sha(target / "batch_inventory.json")
                if (target / "batch_inventory.json").exists()
                else None,
            }
        )
        command_path.write_text(json.dumps(receipt, indent=2) + "\n")
        sys.stdout.write(
            f"{name}: exit {result.returncode} {result.stdout.strip()}" + "\n"
        )
        sys.stdout.flush()
        if result.returncode:
            raise RuntimeError(f"batch failed: {name}: {result.returncode}")
except BaseException as exc:
    (output / "remaining_runner_exit.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "finished_utc": utc(),
                "pid": os.getpid(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
        + "\n"
    )
    raise
else:
    (output / "remaining_runner_exit.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "finished_utc": utc(),
                "pid": os.getpid(),
                "batch_count": len(schedule),
            },
            indent=2,
        )
        + "\n"
    )
