"""Static-trial placement can hold the club cluster at its document seeds (#12221)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import address
from src.shared.python.motion_matching.pipeline.constants import STATIC_FRAMES

pytestmark = pytest.mark.unit

SEEDS = {
    "Head": ("head", (0.0, 0.0, 0.0)),
    "Club1": ("Clubhead", (0.0, 0.0, 0.0)),
}
#: Measured body-frame offsets (identity poses, so world = offset).
MEASURED = {"Head": (0.1, 0.2, 0.3), "Club1": (-0.4, 0.5, 0.6)}


def _lane() -> SimpleNamespace:
    labels = list(MEASURED)
    points = np.array([[MEASURED[k] for k in labels]] * STATIC_FRAMES, dtype=float)
    return SimpleNamespace(
        labels=labels,
        times=np.arange(STATIC_FRAMES) * 0.01,
        points=points,
        valid=np.ones(points.shape[:2], dtype=bool),
    )


class _Kin:
    def body_poses(self, q, bodies):  # noqa: ANN001, ANN201, ARG002
        return {b: (np.eye(3), np.zeros(3)) for b in bodies}


def test_default_places_every_seed_marker() -> None:
    placed = address.static_offsets(_lane(), _Kin(), np.zeros(1), SEEDS)
    for label, offset in MEASURED.items():
        assert placed[label][0] == SEEDS[label][0]
        np.testing.assert_allclose(placed[label][1], offset, atol=1e-12)


def test_fixed_bodies_keep_their_seed_offsets() -> None:
    placed = address.static_offsets(
        _lane(), _Kin(), np.zeros(1), SEEDS, fixed_bodies=("Clubhead",)
    )
    assert placed["Club1"] == SEEDS["Club1"]
    np.testing.assert_allclose(placed["Head"][1], MEASURED["Head"], atol=1e-12)


def test_fix_club_markers_needs_static_seeds() -> None:
    inputs = address.AddressStageInputs(
        lane=None,
        base_spec={},
        hip_spec={},
        hip_bytes=b"",
        upper={},
        fixed={},
        seeds_all={},
        labels=(),
        fix_club_markers=True,
    )
    with pytest.raises(ValueError, match="--fix-club-markers needs --static-seeds"):
        address.solve_address_stage(inputs)


def test_cli_parses_fix_club_markers() -> None:
    from src.shared.python.motion_matching.pipeline import cli

    parser = cli.build_parser()
    assert parser.parse_args(["--fix-club-markers"]).fix_club_markers is True
    assert parser.parse_args([]).fix_club_markers is False


def test_club_marker_body_is_the_fixed_club_body() -> None:
    from src.shared.python.motion_matching.execution.club_markers import (
        CLUB_MARKER_BODY,
    )

    assert address.club_fixed_bodies(True) == (CLUB_MARKER_BODY,)
    assert address.club_fixed_bodies(False) == ()
