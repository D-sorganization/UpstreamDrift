"""File-level calibration algorithm provenance, independent of a clean checkout."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import cv2
import numpy
import scipy
import shared.python.sidekick.lab.mocap as mocap

from src.motion_capture.provenance import sha256_of


def algorithm_evidence() -> dict[str, Any]:
    """Identify provider and adapter source bytes actually available to this worker.

    These digests describe source inputs, not physical or publication approval.
    Runtime versions cover the numerical libraries; this is not a full SBOM.
    """
    if mocap.__file__ is None:
        raise ValueError("Cannot identify the calibration provider source")
    provider = Path(mocap.__file__).resolve().parent
    consumer = Path(__file__).resolve().parent
    return {
        "provider": "D-sorganization/Tools:shared.python.sidekick.lab.mocap",
        "provider_source_sha256": {
            path.relative_to(provider).as_posix(): sha256_of(path)
            for path in sorted(provider.rglob("*.py"))
        },
        "consumer_source_sha256": {
            path.name: sha256_of(path) for path in sorted(consumer.glob("*.py"))
        },
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
    }
