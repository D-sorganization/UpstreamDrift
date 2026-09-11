"""Unit tests for canonical golfer model schema validator (Issue #9965)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.schema_validator import (
    CanonicalModel,
    Inertia,
    ValidationError,
    validate_canonical_model,
)

CANONICAL_YAML = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)


def _minimal_valid_dict() -> dict:
    """Return a minimal valid canonical dictionary."""
    return {
        "description": "Minimal test golfer",
        "units": {
            "length": "meters",
            "mass": "kilograms",
            "time": "seconds",
            "angle": "radians",
        },
        "coordinate_system": {"x": "forward", "y": "left", "z": "up"},
        "root": {
            "name": "pelvis",
            "frame": "pelvis_frame",
            "position": [0.0, 0.0, 0.9],
            "orientation": [0.0, 0.0, 0.0],
            "mass": 11.7,
            "inertia": {
                "ixx": 0.1337,
                "iyy": 0.1337,
                "izz": 0.1337,
                "ixy": 0.0,
                "ixz": 0.0,
                "iyz": 0.0,
            },
            "geometry": {
                "type": "capsule",
                "size": [0.15, 0.2],
                "visual_rgba": [0.8, 0.6, 0.4, 1.0],
            },
        },
        "segments": [
            {
                "name": "torso",
                "parent": "pelvis",
                "joint": {
                    "type": "revolute",
                    "axis": [0, 0, 1],
                    "limits": [-1.0, 1.0],
                    "damping": 1.5,
                },
                "origin": {"xyz": [0.0, 0.0, 0.2], "rpy": [0.0, 0.0, 0.0]},
                "mass": 18.0,
                "inertia": {
                    "ixx": 0.22,
                    "iyy": 0.20,
                    "izz": 0.10,
                    "ixy": 0.0,
                    "ixz": 0.0,
                    "iyz": 0.0,
                },
                "geometry": {
                    "type": "capsule",
                    "size": [0.1, 0.3],
                    "visual_rgba": [0.7, 0.5, 0.3, 1.0],
                },
            }
        ],
    }


class TestSchemaValidatorCanonicalFile:
    """Validate that the canonical specification file in the repository is valid."""

    def test_canonical_yaml_exists_and_parses(self) -> None:
        assert CANONICAL_YAML.exists(), f"Missing canonical YAML at {CANONICAL_YAML}"
        model = validate_canonical_model(CANONICAL_YAML)
        assert isinstance(model, CanonicalModel)
        assert model.root.name == "pelvis"
        assert len(model.segments) == 27
        # Total mass should match ~77.95 kg (77.60 kg body + 0.35 kg club)
        assert np.isclose(model.total_mass, 77.95, atol=0.1)

    def test_canonical_file_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_canonical_model(tmp_path / "nonexistent.yaml")


class TestSchemaValidatorPhysics:
    """Validate physical constraints (masses, inertias, triangle inequality)."""

    def test_negative_mass_raises_validation_error(self) -> None:
        data = _minimal_valid_dict()
        data["root"]["mass"] = -5.0
        with pytest.raises(ValidationError, match="mass must be positive"):
            validate_canonical_model(data)

    def test_segment_negative_mass_raises_validation_error(self) -> None:
        data = _minimal_valid_dict()
        data["segments"][0]["mass"] = 0.0
        with pytest.raises(ValidationError, match="mass must be positive"):
            validate_canonical_model(data)

    def test_triangle_inequality_violation_raises(self) -> None:
        data = _minimal_valid_dict()
        # ixx + iyy < izz: 0.01 + 0.01 = 0.02 < 0.10
        data["segments"][0]["inertia"] = {
            "ixx": 0.01,
            "iyy": 0.01,
            "izz": 0.10,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
        with pytest.raises(ValidationError, match="triangle inequality"):
            validate_canonical_model(data)

    def test_inertia_positive_definiteness(self) -> None:
        inertia = Inertia(ixx=0.1, iyy=0.1, izz=0.1, ixy=0.2, ixz=0.0, iyz=0.0)
        assert not inertia.is_positive_definite()


class TestSchemaValidatorTopology:
    """Validate kinematic tree topology checks."""

    def test_nonexistent_parent_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"][0]["parent"] = "ghost_link"
        with pytest.raises(ValidationError, match="nonexistent parent"):
            validate_canonical_model(data)

    def test_duplicate_segment_name_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"].append(data["segments"][0].copy())
        with pytest.raises(ValidationError, match="Duplicate segment name"):
            validate_canonical_model(data)

    def test_cycle_detection_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"].append(
            {
                "name": "arm",
                "parent": "hand",
                "joint": {"type": "fixed"},
                "origin": {"xyz": [0, 0, 0]},
                "mass": 1.0,
                "inertia": {"ixx": 0.01, "iyy": 0.01, "izz": 0.01},
                "geometry": {"type": "box", "size": [0.1, 0.1, 0.1]},
            }
        )
        data["segments"].append(
            {
                "name": "hand",
                "parent": "arm",
                "joint": {"type": "fixed"},
                "origin": {"xyz": [0, 0, 0]},
                "mass": 0.5,
                "inertia": {"ixx": 0.005, "iyy": 0.005, "izz": 0.005},
                "geometry": {"type": "box", "size": [0.05, 0.05, 0.05]},
            }
        )
        with pytest.raises(ValidationError):
            validate_canonical_model(data)


class TestSchemaValidatorJoints:
    """Validate joint articulation rules."""

    def test_invalid_joint_type_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"][0]["joint"]["type"] = "hyperdrive"
        with pytest.raises(ValidationError, match="invalid joint type"):
            validate_canonical_model(data)

    def test_inverted_joint_limits_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"][0]["joint"]["limits"] = [1.0, -1.0]
        with pytest.raises(ValidationError, match="lower limit.*upper"):
            validate_canonical_model(data)

    def test_negative_damping_raises(self) -> None:
        data = _minimal_valid_dict()
        data["segments"][0]["joint"]["damping"] = -1.0
        with pytest.raises(ValidationError, match="damping must be non-negative"):
            validate_canonical_model(data)
