"""#12702: the pipeline's default spec carries a recognised club body.

The qualified v2 document is the native Simscape export: its grip is the
segmented ``1.5" Bottom`` / ``1.5" Top`` / ``2.5" Butt`` stack, with no single
``Grip`` solid and no ``club`` record. ``assembly_from_spec`` used to return
``None`` for it, so ``Lane.set_club_targets`` raised ``spec has no club body``
on the MuJoCo-native driver run (the slow smoke test), 4.7 minutes in.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching.pipeline.constants import SPEC

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def default_spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _without_club(spec: dict) -> dict:
    doc = copy.deepcopy(spec)
    doc["bodies"] = [
        b for b in doc["bodies"] if not b["name"].endswith(ca.CLUB_BODY_SUFFIX)
    ]
    return doc


def test_default_spec_has_club_body_and_assembly(default_spec: dict) -> None:
    assert ca.club_body_name(default_spec) is not None
    assembly = ca.assembly_from_spec(default_spec)
    assert assembly is not None
    assert assembly.head_alias == "driver"
    assert 1.10 < assembly.length_m < 1.20  # native butt end, about 1.14 m


def test_default_spec_face_axis_is_unit_and_finite(default_spec: dict) -> None:
    axis = cft.face_axis_in_frame(default_spec)
    assert axis.shape == (3,)
    assert np.isfinite(axis).all()
    assert math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1e-9)
    assert np.isfinite(cft.face_centre_in_frame(default_spec)).all()


def test_explicit_club_record_still_wins_over_native_inference(
    default_spec: dict,
) -> None:
    doc = copy.deepcopy(default_spec)
    doc["club"] = {"name": "iron7", "length_m": 0.95}
    assembly = ca.assembly_from_spec(doc)
    assert assembly is not None
    assert (assembly.head_alias, assembly.length_m) == ("iron7", 0.95)


def test_require_club_assembly_names_spec_source(default_spec: dict) -> None:
    with pytest.raises(ValueError, match="spec has no club body") as exc:
        cft.require_club_assembly(_without_club(default_spec))
    message = str(exc.value)
    assert "schema_version='full-body-v1'" in message
    assert "provenance=" in message
    assert ca.CLUB_BODY_SUFFIX in message


def test_set_club_targets_fails_fast_without_club(default_spec: dict) -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    lane = object.__new__(Lane)  # the guard must fire before any lane state is read
    with pytest.raises(ValueError, match="spec has no club body.*schema_version"):
        lane.set_club_targets({}, _without_club(default_spec), face_weight=1.0)
    assert not hasattr(lane, "face_weight")  # no partial state was written
