"""Frozen capture-A / canonical-v2 reference pins for OpenSim catch-up A1.

Epic #15793 package A1. The public capture, marker schema, timebase, and
canonical-v2 frame contract live in this repository. The best-match trajectory
does not: callers pass the external same-input bundle named by
``UD_DEFAULT_MATCH_BUNDLE``. The committed trunk-visual demo is pinned only so
it cannot be mistaken for that best match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.motion_capture.capture_registry import capture_info
from src.shared.python.contracts import ensure, require
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.swing_comparison.motion import CAPTURE_A_MARKER_LABELS

FIXTURE_RELATIVE = "src/config/opensim_catchup/a1_reference_fixture.json"
FIXTURE_SCHEMA = "opensim-catchup-a1/v1"
CAPTURE_ID = "capture-A"
BUNDLE_SCHEMA = "same-input-bundle/v1"
BEST_MATCH_ENV = "UD_DEFAULT_MATCH_BUNDLE"
GRAVITY_M_S2 = (0.0, 0.0, -GRAVITY_FLOAT)


class A1FixtureError(ValueError):
    """The frozen A1 reference pin does not match the tree or a bundle."""


def _pin(condition: bool, message: str) -> None:
    """Enforce a pin even when contract checks are silenced."""
    require(condition, message)
    if not condition:
        raise A1FixtureError(message)


def repo_root_from(start: Path | None = None) -> Path:
    """Return the checkout that contains ``data/capture_registry.json``."""
    origin = Path(__file__).resolve() if start is None else start.resolve()
    candidates = [origin, *origin.parents]
    for candidate in candidates:
        if (candidate / "data" / "capture_registry.json").is_file():
            return candidate
    raise A1FixtureError("capture registry not found")


def fixture_path(repo_root: Path | None = None) -> Path:
    """Absolute path of the committed A1 fixture document."""
    root = repo_root or repo_root_from()
    return root / FIXTURE_RELATIVE


def load_fixture(repo_root: Path | None = None) -> dict[str, Any]:
    """Load the A1 fixture. Pre: the document uses ``FIXTURE_SCHEMA``."""
    path = fixture_path(repo_root)
    _pin(path.is_file(), f"A1 fixture missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    _pin(isinstance(document, dict), "A1 fixture must be a JSON object")
    _pin(document.get("schema") == FIXTURE_SCHEMA, "unexpected A1 fixture schema")
    return document


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of ``path``."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _close(left: float, right: float, message: str, tol: float = 1e-12) -> None:
    """Pin two floats to within ``tol``; raise A1FixtureError with ``message`` otherwise."""
    _pin(abs(left - right) <= tol, message)


def verify_public_pins(repo_root: Path | None = None) -> dict[str, Any]:
    """Check committed pins against capture-A, markers, and the demo bundle.

    Post: the capture file digest equals both the fixture and the registry.
    """
    root = repo_root or repo_root_from()
    fixture = load_fixture(root)
    info = capture_info(CAPTURE_ID, repo_root=root)
    capture = fixture["capture"]
    _pin(capture["id"] == info.id == CAPTURE_ID, "capture id drifted")
    _pin(capture["sha256"] == info.sha256, "fixture sha drifted from registry")
    _pin(capture["relative_path"] == info.relative_path, "capture path drifted")
    _pin(int(capture["bytes"]) == int(info.bytes or -1), "capture size drifted")
    _pin(int(capture["frame_count"]) == int(info.frame_count or -1), "frames drifted")
    _close(
        float(capture["sample_rate_hz"]),
        float(info.sample_rate_hz or -1),
        "sample rate drifted",
    )
    capture_path = root / str(capture["relative_path"])
    _pin(capture_path.is_file(), "capture-A file is not in the checkout")
    _pin(sha256_file(capture_path) == capture["sha256"], "capture-A bytes drifted")

    labels = tuple(fixture["markers"]["labels"])
    _pin(labels == CAPTURE_A_MARKER_LABELS, "marker schema drifted")
    _pin(int(fixture["markers"]["count"]) == len(labels) == 38, "marker count drifted")

    timebase = fixture["timebase"]
    rate = float(info.sample_rate_hz or 0.0)
    _close(float(timebase["sample_rate_hz"]), rate, "timebase rate drifted")
    _pin(int(timebase["frame_count"]) == int(info.frame_count or -1), "timebase frames")
    _close(float(timebase["dt_s"]), 1.0 / rate, "timebase dt drifted")
    _pin(float(timebase["t0_s"]) == 0.0, "timebase must start at t=0")

    model = fixture["model"]
    _pin(model["convention"] == "canonical-v2", "model convention drifted")
    _pin(model["frame"] == "world_Zup", "world frame drifted")
    _pin(model["units"] == "SI", "units drifted")
    _pin(model["quaternion_order"] == "wxyz", "quaternion order drifted")
    _pin(model["base_linear_velocity_frame"] == "world", "linear velocity frame")
    _pin(model["base_angular_velocity_frame"] == "body", "angular velocity frame")
    gravity = tuple(float(item) for item in model["gravity_m_s2"])
    _pin(len(gravity) == 3, "gravity must be a 3-vector")
    for got, expected in zip(gravity, GRAVITY_M_S2, strict=True):
        _close(got, expected, "gravity drifted")

    frames = fixture["frame_transforms"]
    _pin(frames["world"] == "Z-up-right-handed", "world transform drifted")
    _pin(frames["length_unit"] == "m", "length unit drifted")
    _pin(frames["angle_unit"] == "rad", "angle unit drifted")

    trajectory = fixture["reference_trajectory"]
    _pin(
        trajectory["best_match_location"] == "external", "best match must stay external"
    )
    _pin(trajectory["best_match_env"] == BEST_MATCH_ENV, "best-match env drifted")
    _pin(trajectory["bundle_schema"] == BUNDLE_SCHEMA, "bundle schema drifted")
    _pin(
        trajectory["in_repo_demo_is_best_match"] is False, "demo is not the best match"
    )
    demo_rel = str(trajectory["in_repo_demo_relative_path"])
    demo_path = root / demo_rel
    _pin(demo_path.is_file(), "in-repo demo bundle missing")
    _pin(
        sha256_file(demo_path) == trajectory["in_repo_demo_sha256"],
        "in-repo demo bundle bytes drifted",
    )
    ensure(capture["sha256"] == sha256_file(capture_path), "capture digest confirmed")
    return fixture


def verify_best_match_bundle(
    path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Check an external same-input bundle against the frozen capture-A timebase.

    Pre: ``path`` is a ``same-input-bundle/v1`` npz. The bundle is not copied
    into the repository. Post: ``dt_s`` matches capture-A and ``reference_q``
    has at least two rows.
    """
    fixture = verify_public_pins(repo_root)
    _pin(path.is_file(), f"best-match bundle missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        reference_q = np.asarray(data["reference_q"])
    expected_schema = fixture["reference_trajectory"]["bundle_schema"]
    _pin(manifest.get("schema") == expected_schema, "bundle schema is not A1's")
    _close(
        float(manifest["dt_s"]),
        float(fixture["timebase"]["dt_s"]),
        "bundle dt does not match capture-A",
        tol=1e-9,
    )
    _pin(
        reference_q.ndim == 2 and int(reference_q.shape[0]) >= 2,
        "reference_q must be a trajectory",
    )
    spec_sha = str(manifest["spec_sha256"])
    _pin(len(spec_sha) == 64, "bundle spec_sha256 must be a sha256 hex digest")
    steps = int(reference_q.shape[0])
    result: dict[str, str | float | int] = {
        "schema": str(manifest["schema"]),
        "spec_sha256": spec_sha,
        "dt_s": float(manifest["dt_s"]),
        "reference_steps": steps,
    }
    ensure(steps >= 2, "bundle trajectory confirmed")
    return result
