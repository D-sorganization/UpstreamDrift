"""FreeMoCap sidecar runner.

Spawns FreeMoCap as an **isolated subprocess** so that UpstreamDrift's
main environment never imports AGPL-licensed code at runtime. Called
either as a CLI (``python -m src.tools.freemocap_sidecar.run_freemocap
--input ... --output ...``) or programmatically via
:func:`run_freemocap_sidecar`.

Architecture
------------

UpstreamDrift is MIT-licensed; FreeMoCap is AGPL. To keep the licenses
cleanly separated we never import freemocap from this codebase. Instead:

1. The user installs freemocap into a separate Python environment (e.g.
   ``~/.venvs/freemocap``).
2. Our sidecar invokes ``<that env>/python -m freemocap ...`` as a
   subprocess.
3. The subprocess writes results to a known output directory.
4. UpstreamDrift reads those results back via stdlib only.

No symbol from freemocap is ever bound in our process. AGPL code stays
entirely on the other side of the subprocess boundary.

Output contract
---------------

After a successful run the output directory contains at minimum:

- ``landmarks.csv`` -- frame-by-frame 3D landmark positions
  ``frame,landmark_id,x,y,z``
- ``metadata.json`` -- session metadata (camera count, fps, duration,
  freemocap version)

When real freemocap is unavailable (no env, no install, dry-run mode)
the sidecar still writes minimum stub artifacts so downstream consumers
have a stable contract to test against.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class FreeMoCapSidecarError(RuntimeError):
    """Raised when the freemocap subprocess fails or its output is malformed."""


@dataclass(frozen=True)
class FreeMoCapResult:
    """Result of a freemocap sidecar invocation.

    Attributes:
        success: True iff the subprocess exited 0 and the expected
            output files exist.
        output_dir: Directory containing landmarks.csv and metadata.json.
        landmarks_csv: Absolute path to the landmarks file (None on failure).
        metadata_json: Absolute path to the metadata file (None on failure).
        return_code: Exit code from the subprocess.
        stderr_tail: Last few KB of subprocess stderr (truncated for log
            sanity).
        used_real_freemocap: True iff a real freemocap install was
            invoked. False when running in dry-run / scaffold mode.
    """

    success: bool
    output_dir: Path
    return_code: int
    used_real_freemocap: bool
    landmarks_csv: Path | None = None
    metadata_json: Path | None = None
    stderr_tail: str = ""
    extra: dict[str, str] = field(default_factory=dict)


# Maximum bytes of stderr to retain in the result (avoid blowing context).
_STDERR_TAIL_BYTES = 4096


def _write_stub_artifacts(output_dir: Path) -> tuple[Path, Path]:
    """Write minimum stub artifacts when real freemocap is unavailable.

    Stubs follow the same shape as real freemocap output so downstream
    consumers (motion_pipeline, tests) can integration-test without
    needing the actual freemocap install.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    landmarks = output_dir / "landmarks.csv"
    metadata = output_dir / "metadata.json"
    landmarks.write_text("frame,landmark_id,x,y,z\n0,0,0.0,0.0,0.0\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "stub": True,
                "n_frames": 1,
                "n_landmarks": 1,
                "fps": 0,
                "freemocap_version": "stub",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return landmarks, metadata


def _validate_output(output_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (landmarks, metadata) paths if both exist, else (None, None)."""
    landmarks = output_dir / "landmarks.csv"
    metadata = output_dir / "metadata.json"
    if landmarks.exists() and metadata.exists():
        return landmarks, metadata
    return None, None


def run_freemocap_sidecar(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    freemocap_env_python: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    timeout_s: float = 1800.0,
) -> FreeMoCapResult:
    """Run the freemocap sidecar.

    Args:
        input_dir: Path to the recording session videos.
        output_dir: Path where landmarks.csv and metadata.json will be
            written. Created if it does not exist.
        freemocap_env_python: Path to the python interpreter in the
            isolated freemocap venv. If None, defaults to ``sys.executable``.
        dry_run: If True, skip the subprocess invocation and write only
            stub artifacts. Useful for tests and CI matrices that don't
            have freemocap installed.
        timeout_s: Subprocess timeout. Defaults to 30 minutes.

    Returns:
        :class:`FreeMoCapResult` with paths and metadata.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    landmarks: Path | None = None
    metadata: Path | None = None

    if dry_run:
        logger.info("Dry-run mode: writing stub artifacts to %s", output_path)
        landmarks, metadata = _write_stub_artifacts(output_path)
        return FreeMoCapResult(
            success=True,
            output_dir=output_path,
            return_code=0,
            used_real_freemocap=False,
            landmarks_csv=landmarks,
            metadata_json=metadata,
        )

    python_exe = (
        str(freemocap_env_python)
        if freemocap_env_python is not None
        else sys.executable
    )

    cmd: list[str] = [
        python_exe,
        "-m",
        "freemocap",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]

    logger.info(
        "Invoking freemocap subprocess: python=%s input=%s output=%s",
        python_exe,
        input_path,
        output_path,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "freemocap interpreter %s not found (%s); falling back to stub artifacts",
            python_exe,
            exc,
        )
        landmarks, metadata = _write_stub_artifacts(output_path)
        return FreeMoCapResult(
            success=False,
            output_dir=output_path,
            return_code=127,
            used_real_freemocap=False,
            landmarks_csv=landmarks,
            metadata_json=metadata,
            stderr_tail=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("freemocap subprocess timed out after %.1fs", timeout_s)
        return FreeMoCapResult(
            success=False,
            output_dir=output_path,
            return_code=-1,
            used_real_freemocap=True,
            stderr_tail=f"timeout after {timeout_s}s: {exc}",
        )

    stderr_tail = (proc.stderr or "")[-_STDERR_TAIL_BYTES:]

    if proc.returncode != 0:
        if "No module named 'freemocap'" in (proc.stderr or ""):
            logger.warning(
                "freemocap module not installed at %s; writing stub artifacts",
                python_exe,
            )
            landmarks, metadata = _write_stub_artifacts(output_path)
            return FreeMoCapResult(
                success=False,
                output_dir=output_path,
                return_code=proc.returncode,
                used_real_freemocap=False,
                landmarks_csv=landmarks,
                metadata_json=metadata,
                stderr_tail=stderr_tail,
            )

        logger.error("freemocap subprocess exited with code %d", proc.returncode)
        return FreeMoCapResult(
            success=False,
            output_dir=output_path,
            return_code=proc.returncode,
            used_real_freemocap=True,
            stderr_tail=stderr_tail,
        )

    landmarks, metadata = _validate_output(output_path)
    if landmarks is None or metadata is None:
        logger.error(
            "freemocap exited 0 but expected output files are missing in %s",
            output_path,
        )
        return FreeMoCapResult(
            success=False,
            output_dir=output_path,
            return_code=proc.returncode,
            used_real_freemocap=True,
            stderr_tail=stderr_tail,
        )

    logger.info(
        "freemocap sidecar completed successfully; landmarks=%s metadata=%s",
        landmarks,
        metadata,
    )
    return FreeMoCapResult(
        success=True,
        output_dir=output_path,
        return_code=proc.returncode,
        used_real_freemocap=True,
        landmarks_csv=landmarks,
        metadata_json=metadata,
        stderr_tail=stderr_tail,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="FreeMoCap sidecar runner (subprocess isolation for AGPL freemocap)",
    )
    parser.add_argument("--input", required=True, help="Path to input session videos")
    parser.add_argument(
        "--output", required=True, help="Path to output landmarks directory"
    )
    parser.add_argument(
        "--env-python",
        required=False,
        default=None,
        help="Path to the python executable in the freemocap venv",
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

    result = run_freemocap_sidecar(
        input_dir=args.input,
        output_dir=args.output,
        freemocap_env_python=args.env_python,
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
