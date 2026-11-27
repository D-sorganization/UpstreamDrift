"""Contracts for hash-bound review poses used by the preview exporter."""

import hashlib

import pytest

from scripts.export_player_models import _validated_review_pose
from src.shared.python.model_appearance.schema import physics_spec_sha256

pytestmark = pytest.mark.unit


def test_pose_validation_checks_hashes_and_maps_coordinates_by_name():
    """Accept exact target hashes and return q mapped to its declared names."""
    spec = {"coordinate_order": ["x", "z"]}
    xml = b"review target"
    pose = {
        "target_physics_spec_sha256": physics_spec_sha256(spec),
        "target_model_xml_sha256": hashlib.sha256(xml).hexdigest(),
        "coordinate_order": ["z", "x"],
        "q": [2.0, 1.0],
    }
    assert _validated_review_pose(pose, spec, xml) == {"z": 2.0, "x": 1.0}
    with pytest.raises(ValueError, match="XML hash"):
        _validated_review_pose({**pose, "target_model_xml_sha256": "0" * 64}, spec, xml)
