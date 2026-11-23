"""New specs are built on the current golfer model (#15648).

``build_anthropometric_spec`` used to write specs that drew the deprecated
capsule hands and loft body. It now applies the registry's current model: the
skinned hand wrist revision and the skinned body. A left-handed spec keeps the
deprecated parts (the skinned pair is posed right-handed) and says so.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest

from src.shared.python.model_appearance import golfer_model_versions as gmv
from src.shared.python.motion_matching.execution import spec_builder
from src.shared.python.motion_matching.execution.assets import (
    get_candidate_geometry_spec,
    get_native_geometry_spec,
    get_opensim_model,
)
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_golfer_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def built() -> spec_builder.AnthropometricDocument:
    return spec_builder.compose_anthropometric_document()


def test_a_right_handed_spec_moves_to_the_current_model(built: Any) -> None:
    document, subject = with_current_golfer_model(built.document, built.subject)
    current = gmv.current_golfer_model()
    assert gmv.deprecated_parts(document) == ()
    assert document["spec_revision"]["configuration"] == (
        current.spec_revision_configuration
    )
    assert subject["hand_visual_backend"] == current.hand_visual_backend
    assert subject["body_visual_backend"] == current.body_visual_backend


def test_the_revision_moves_placements_only(built: Any) -> None:
    before = copy.deepcopy(built.document)
    document, _ = with_current_golfer_model(built.document, built.subject)
    assert built.document == before
    masses = [(b["name"], b.get("mass")) for b in document["bodies"]]
    assert masses == [(b["name"], b.get("mass")) for b in before["bodies"]]


def test_a_left_handed_spec_keeps_the_deprecated_parts_and_says_so(
    built: Any, caplog: pytest.LogCaptureFixture
) -> None:
    left = copy.deepcopy(built.document)
    left["subject"]["handedness"] = "left"
    gmv.reset_deprecation_warnings()
    with caplog.at_level(logging.WARNING):
        document, _ = with_current_golfer_model(left, built.subject)
    assert document == left
    assert "capsule" in caplog.text and "loft" in caplog.text


def test_the_builder_writes_a_current_model_spec(tmp_path: Any) -> None:
    spec_path, _receipt = spec_builder.build_anthropometric_spec(
        native_path=get_native_geometry_spec(),
        osim_path=get_opensim_model(),
        native_candidate_path=get_candidate_geometry_spec(),
        stature_m=1.80,
        mass_kg=80.0,
        output_dir=tmp_path,
    )
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    assert gmv.deprecated_parts(document) == ()


def test_physical_grip_picks_keep_their_hands_and_say_so(
    built: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The skinned pair is posed with the owner grip, not ``grip_config``.

    Moving a spec with physical grip picks onto it would silently drop the
    picks, so it keeps the deprecated hands, logged, until the skinned pair is
    posed from ``grip_config``.
    """
    picked = copy.deepcopy(built.document)
    picked["subject"]["grip_config"] = {"style": "interlock"}
    gmv.reset_deprecation_warnings()
    with caplog.at_level(logging.WARNING):
        document, _ = with_current_golfer_model(picked, built.subject)
    assert document == picked
    assert "capsule" in caplog.text
