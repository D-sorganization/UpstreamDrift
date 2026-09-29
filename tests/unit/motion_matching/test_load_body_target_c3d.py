"""Unit tests for the C3D body-marker loader producing :class:`BodyTarget`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import (
    AlignOptions,
    BodyTarget,
    load_body_target_c3d,
    load_club_target_c3d,
)
from src.shared.python.motion_matching.load_body_target import load_body_target
from src.shared.python.motion_matching.loaders.c3d_body import (
    DEFAULT_BODY_MARKER_EXCLUDES,
    default_anatomical_marker_set,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


pytestmark = pytest.mark.skipif(
    not DRIVER_C3D.exists(),
    reason=f"reference C3D fixture not available at {DRIVER_C3D}",
)


def _opts() -> AlignOptions:
    return AlignOptions(
        sample_rate_hz=1000.0,
        simulation_time_s=0.3,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )


def test_load_body_target_c3d_happy_path() -> None:
    """Default load returns 28 anatomical markers minus the excluded one."""
    bt = load_body_target_c3d(DRIVER_C3D, _opts())
    assert isinstance(bt, BodyTarget)
    # Default 28-marker set minus 1 excluded-by-default = 27 markers.
    expected = default_anatomical_marker_set()
    assert bt.marker_names == expected
    assert len(bt.marker_names) == 27

    # Sample count matches the AlignOptions grid: int(0.3*1000)+1 = 301.
    assert bt.time.shape == (301,)
    assert bt.marker_xyz.shape == (301, len(expected), 3)

    # Impact lands within +-5 frames of the documented sample (251 == idx 250).
    assert abs(int(bt.impact_idx) - 250) <= 5

    # source provenance is populated
    assert bt.source.format == "c3d"
    assert bt.source.filename == DRIVER_C3D.name
    assert bt.source.sha256


def test_default_excludes_known_occluded_marker() -> None:
    """Known-occluded marker must not be silently included by default."""
    assert "RShoulderTop" in DEFAULT_BODY_MARKER_EXCLUDES
    bt = load_body_target_c3d(DRIVER_C3D, _opts())
    assert "RShoulderTop" not in bt.marker_names


def test_explicit_marker_set_honoured_in_order() -> None:
    """Explicit marker_set is returned in the requested order."""
    requested = ("HeadTop", "WaistLeft", "RWristTop")
    bt = load_body_target_c3d(DRIVER_C3D, _opts(), marker_set=requested)
    assert bt.marker_names == requested
    assert bt.marker_xyz.shape[1] == 3


def test_shared_clock_with_club_target() -> None:
    """When impact_source is supplied, body and club share the timegrid exactly."""
    opts = _opts()
    club = load_club_target_c3d(DRIVER_C3D, opts)
    body = load_body_target_c3d(DRIVER_C3D, opts, impact_source=club)
    assert body.time.shape == club.time.shape
    np.testing.assert_array_equal(body.time, club.time)


def test_unknown_path_raises_filenotfound() -> None:
    """Nonexistent path produces a descriptive FileNotFoundError."""
    with pytest.raises((FileNotFoundError, Exception)) as excinfo:
        load_body_target_c3d(Path("does_not_exist.c3d"), _opts())
    # Either a precondition AssertionError-style failure or FileNotFoundError
    # may be raised depending on the contracts decorator. Both are acceptable
    # so long as the failure is loud.
    assert excinfo.type is FileNotFoundError or "exist" in str(excinfo.value).lower()


def test_dispatcher_routes_c3d() -> None:
    """The top-level dispatcher routes ``.c3d`` to the C3D loader."""
    bt = load_body_target(DRIVER_C3D, opts=_opts())
    assert isinstance(bt, BodyTarget)


def test_dispatcher_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported extensions raise ValueError mentioning the extension."""
    bogus = tmp_path / "fake.unknownext"
    bogus.write_bytes(b"not a real file")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_body_target(bogus, opts=_opts())


def test_explicit_empty_marker_set_rejected() -> None:
    """Empty explicit marker_set raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        load_body_target_c3d(DRIVER_C3D, _opts(), marker_set=())


def test_unknown_marker_in_marker_set_rejected() -> None:
    """Markers not present in the C3D file raise ValueError."""
    with pytest.raises(ValueError, match="not present"):
        load_body_target_c3d(
            DRIVER_C3D, _opts(), marker_set=("HeadTop", "NoSuchMarker")
        )
