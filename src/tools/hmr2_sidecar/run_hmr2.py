"""4D-Humans / HMR 2.0 sidecar runner.

Spawns the user-installed `4D-Humans <https://github.com/shubham-goel/4D-Humans>`_
demo pipeline as an **isolated subprocess** so that UpstreamDrift's main
environment never imports CC-BY-NC-licensed code at runtime. Called
either as a CLI (``python -m src.tools.hmr2_sidecar.run_hmr2
--video ... --output ...``) or programmatically via
:func:`run_hmr2_sidecar`.

Architecture
------------

UpstreamDrift is MIT-licensed; 4D-Humans is CC-BY-NC and the SMPL body
model weights it requires are research-restricted. To keep the licenses
cleanly separated we never import 4D-Humans (or SMPL model files) from
this codebase. Instead:

1. The user installs 4D-Humans into a separate Python environment and
   registers its SMPL weights there.
2. Our sidecar invokes the command configured via the ``HMR2_COMMAND``
   environment variable (or an explicit ``hmr2_command`` argument) as a
   subprocess, appending ``--video <path> --out_folder <dir>``.
3. The subprocess writes results to a known output directory.
4. UpstreamDrift reads those results back via stdlib only.

No symbol from 4D-Humans is ever bound in our process. CC-BY-NC code and
research-restricted weights stay entirely on the other side of the
subprocess boundary.

Output contract
---------------

After a successful run the output directory contains at minimum:

- ``joints3d.csv`` -- frame-by-frame 3D joint positions in meters with
  columns ``frame,time`` followed by ``<joint>_x,<joint>_y,<joint>_z``
  for each of the 22 SMPL body joints in :data:`SMPL_BODY_JOINTS` order.
- ``betas.json`` -- ``{"betas": [<10 floats>], "gender": "neutral"}``
  SMPL shape coefficients for the tracked subject.
- ``metadata.json`` -- source video path, fps, and tool version
  (``"stub"`` when no real tool ran).

When real 4D-Humans is unavailable (no command configured, command not
found, dry-run mode) the sidecar still writes minimum stub artifacts so
downstream consumers have a stable contract to test against.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Environment variable naming the external 4D-Humans command to invoke.
HMR2_COMMAND_ENV = "HMR2_COMMAND"

#: The 22 SMPL body joints, in canonical SMPL kinematic-tree order.
#: This tuple defines the ``joints3d.csv`` column contract. The adapter
#: (``src.shared.python.motion_pipeline.sources.hmr2_adapter``) carries
#: its own copy — adapters may not import ``src.tools`` per the
#: Law-of-Demeter gate — kept in sync by
#: ``tests/unit/motion_pipeline/sources/test_hmr2_adapter.py``.
SMPL_BODY_JOINTS: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

#: Number of SMPL shape coefficients written to ``betas.json``.
NUM_BETAS = 10

#: Full ``joints3d.csv`` header, in order.
JOINTS3D_COLUMNS: tuple[str, ...] = (
    "frame",
    "time",
    *(f"{joint}_{axis}" for joint in SMPL_BODY_JOINTS for axis in ("x", "y", "z")),
)


class HMR2SidecarError(RuntimeError):
    """Raised when the 4D-Humans subprocess fails or its output is malformed."""


@dataclass(frozen=True)
class HMR2Result:
    """Result of a 4D-Humans sidecar invocation.

    Attributes:
        success: True iff artifacts satisfying the output contract exist.
        output_dir: Directory containing joints3d.csv, betas.json, and
            metadata.json.
        return_code: Exit code from the subprocess (0 for stub modes).
        used_real_hmr2: True iff a real 4D-Humans install was invoked.
            False when running in dry-run / stub mode.
        joints3d_csv: Absolute path to the joints file (None on failure).
        betas_json: Absolute path to the betas file (None on failure).
        metadata_json: Absolute path to the metadata file (None on failure).
        stderr_tail: Last few KB of subprocess stderr (truncated for log
            sanity).
    """

    success: bool
    output_dir: Path
    return_code: int
    used_real_hmr2: bool
    joints3d_csv: Path | None = None
    betas_json: Path | None = None
    metadata_json: Path | None = None
    stderr_tail: str = ""
    extra: dict[str, str] = field(default_factory=dict)


# Maximum bytes of stderr to retain in the result (avoid blowing context).
_STDERR_TAIL_BYTES = 4096

#: Default stub frame rate (Hz) recorded in stub artifacts.
_STUB_FPS = 30.0


def _write_stub_artifacts(
    output_dir: Path,
    video_path: Path | None = None,
    fps: float = _STUB_FPS,
) -> tuple[Path, Path, Path]:
    """Write minimum stub artifacts when real 4D-Humans is unavailable.

    Stubs follow the same shape as real sidecar output so downstream
    consumers (motion_pipeline adapter, character builder) can
    integration-test without needing the actual 4D-Humans install.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    joints3d = output_dir / "joints3d.csv"
    betas = output_dir / "betas.json"
    metadata = output_dir / "metadata.json"

    header = ",".join(JOINTS3D_COLUMNS)
    zeros = ",".join("0.0" for _ in range(3 * len(SMPL_BODY_JOINTS)))
    rows = [header]
    for frame in range(2):
        rows.append(f"{frame},{frame / fps},{zeros}")
    joints3d.write_text("\n".join(rows) + "\n", encoding="utf-8")

    betas.write_text(
        json.dumps({"betas": [0.0] * NUM_BETAS, "gender": "neutral"}, indent=2),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "stub": True,
                "tool": "4D-Humans",
                "tool_version": "stub",
                "source_video": str(video_path) if video_path is not None else None,
                "fps": fps,
                "n_frames": 2,
                "joint_names": list(SMPL_BODY_JOINTS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return joints3d, betas, metadata


def _validate_output(
    output_dir: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Return (joints3d, betas, metadata) paths if all exist, else Nones."""
    joints3d = output_dir / "joints3d.csv"
    betas = output_dir / "betas.json"
    metadata = output_dir / "metadata.json"
    if joints3d.exists() and betas.exists() and metadata.exists():
        return joints3d, betas, metadata
    return None, None, None


def _resolve_command(
    hmr2_command: str | Sequence[str] | None,
) -> list[str] | None:
    """Resolve the external command to invoke, or None for stub mode.

    Precedence: explicit ``hmr2_command`` argument, then the
    :data:`HMR2_COMMAND_ENV` environment variable. A string is split
    with :func:`shlex.split`; a sequence is used verbatim.
    """
    if hmr2_command is not None:
        if isinstance(hmr2_command, str):
            return shlex.split(hmr2_command)
        return [str(part) for part in hmr2_command]
    env_command = os.environ.get(HMR2_COMMAND_ENV, "").strip()
    if env_command:
        return shlex.split(env_command)
    return None


def run_hmr2_sidecar(
    video_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    hmr2_command: str | Sequence[str] | None = None,
    dry_run: bool = False,
    timeout_s: float = 1800.0,
) -> HMR2Result:
    """Run the 4D-Humans / HMR2 sidecar.

    Args:
        video_path: Path to the monocular input video.
        output_dir: Path where joints3d.csv, betas.json, and
            metadata.json will be written. Created if it does not exist.
        hmr2_command: External command invoking a user-installed
            4D-Humans demo wrapper. If None, the :data:`HMR2_COMMAND_ENV`
            environment variable is consulted; if that is also unset the
            sidecar writes stub artifacts instead of spawning anything.
        dry_run: If True, skip the subprocess invocation and write only
            stub artifacts. Useful for tests and CI matrices that don't
            have 4D-Humans installed.
        timeout_s: Subprocess timeout. Defaults to 30 minutes.

    Returns:
        :class:`HMR2Result` with paths and metadata.
    """
    video = Path(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = None if dry_run else _resolve_command(hmr2_command)
    if cmd is None:
        mode = "dry-run" if dry_run else "unconfigured"
        logger.info(
            "HMR2 sidecar in %s mode (no %s configured); writing stub artifacts to %s",
            mode,
            HMR2_COMMAND_ENV,
            output_path,
        )
        joints3d, betas, metadata = _write_stub_artifacts(output_path, video)
        return HMR2Result(
            success=True,
            output_dir=output_path,
            return_code=0,
            used_real_hmr2=False,
            joints3d_csv=joints3d,
            betas_json=betas,
            metadata_json=metadata,
            extra={"mode": mode},
        )

    full_cmd = [*cmd, "--video", str(video), "--out_folder", str(output_path)]
    logger.info("Invoking 4D-Humans subprocess: %s", full_cmd)

    try:
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "HMR2 command %s not found (%s); falling back to stub artifacts",
            cmd[0],
            exc,
        )
        joints3d, betas, metadata = _write_stub_artifacts(output_path, video)
        return HMR2Result(
            success=False,
            output_dir=output_path,
            return_code=127,
            used_real_hmr2=False,
            joints3d_csv=joints3d,
            betas_json=betas,
            metadata_json=metadata,
            stderr_tail=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("4D-Humans subprocess timed out after %.1fs", timeout_s)
        return HMR2Result(
            success=False,
            output_dir=output_path,
            return_code=-1,
            used_real_hmr2=True,
            stderr_tail=f"timeout after {timeout_s}s: {exc}",
        )

    stderr_tail = (proc.stderr or "")[-_STDERR_TAIL_BYTES:]

    if proc.returncode != 0:
        logger.error("4D-Humans subprocess exited with code %d", proc.returncode)
        return HMR2Result(
            success=False,
            output_dir=output_path,
            return_code=proc.returncode,
            used_real_hmr2=True,
            stderr_tail=stderr_tail,
        )

    joints3d, betas, metadata = _validate_output(output_path)
    if joints3d is None or betas is None or metadata is None:
        logger.error(
            "4D-Humans exited 0 but expected output files are missing in %s",
            output_path,
        )
        return HMR2Result(
            success=False,
            output_dir=output_path,
            return_code=proc.returncode,
            used_real_hmr2=True,
            stderr_tail=stderr_tail,
        )

    logger.info(
        "HMR2 sidecar completed successfully; joints3d=%s betas=%s metadata=%s",
        joints3d,
        betas,
        metadata,
    )
    return HMR2Result(
        success=True,
        output_dir=output_path,
        return_code=proc.returncode,
        used_real_hmr2=True,
        joints3d_csv=joints3d,
        betas_json=betas,
        metadata_json=metadata,
        stderr_tail=stderr_tail,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "4D-Humans/HMR2 sidecar runner "
            "(subprocess isolation for CC-BY-NC 4D-Humans)"
        ),
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--output", required=True, help="Path to output artifact directory"
    )
    parser.add_argument(
        "--command",
        required=False,
        default=None,
        help=(
            "External 4D-Humans command to invoke "
            f"(default: the {HMR2_COMMAND_ENV} environment variable)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the subprocess and write stub artifacts only",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Subprocess timeout in seconds (default 1800)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the result as JSON to stdout"
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    args = parser.parse_args(argv)

    result = run_hmr2_sidecar(
        video_path=args.video,
        output_dir=args.output,
        hmr2_command=args.command,
        dry_run=args.dry_run,
        timeout_s=args.timeout,
    )

    if args.json:
        payload = asdict(result)
        for k, v in payload.items():
            if isinstance(v, Path):
                payload[k] = str(v)
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")

    return 0 if result.success else max(1, result.return_code)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
