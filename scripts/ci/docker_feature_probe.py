#!/usr/bin/env python3
"""Probe a feature inside a Docker image and print its registry status.

The Docker smoke workflow used to pipe registry JSON directly through a
one-line parser while discarding container stderr. That made a valid status
indistinguishable from import noise or an early interpreter crash: both showed
up as ``PROBE_ERROR``. This helper keeps the contract narrow:

* stdout is exactly the parsed status when a status can be recovered.
* stderr includes the Docker command output when no status can be parsed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError


class ProbeParseError(ValueError):
    """Raised when registry output does not contain a JSON status."""


@dataclass(frozen=True)
class ProbeResult:
    """Captured Docker probe process result."""

    returncode: int
    stdout: str
    stderr: str


def extract_status(stdout: str) -> str:
    """Return the first JSON object's ``status`` field from stdout.

    Preconditions:
        * ``stdout`` is the registry command's stdout text.

    Postconditions:
        * Returned status is a non-empty string.

    Raises:
        ProbeParseError: when no JSON object with a string ``status`` is found.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("status"), str):
            return payload["status"]
    raise ProbeParseError("probe stdout did not contain a JSON object with status")


def run_probe(image: str, feature: str) -> ProbeResult:
    """Run the in-image feature registry check."""
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image,
            "python",
            "-m",
            "src.shared.python.feature_registry",
            "--check",
            feature,
            "--json",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    return ProbeResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a feature registry status from a Docker image."
    )
    parser.add_argument("image", help="Docker image tag to probe.")
    parser.add_argument("feature", help="Feature registry key to check.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = run_probe(args.image, args.feature)
    try:
        status = extract_status(result.stdout)
    except ProbeParseError as exc:
        sys.stderr.write(
            f"::error::Feature probe failed for {args.feature!r} in {args.image!r}: "
            f"{exc}; docker exit={result.returncode}\n"
        )
        if result.stdout:
            sys.stderr.write("--- probe stdout ---\n")
            sys.stderr.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if result.stderr:
            sys.stderr.write("--- probe stderr ---\n")
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")
        return 2
    sys.stdout.write(status)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
