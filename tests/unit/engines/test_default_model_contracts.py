"""Tests for cross-engine default model contracts and body naming (Issue #12985).

Covers:
- Canonical list of required golfer body names across core engines.
- Default model files on disk (MuJoCo, Drake, Pinocchio) containing required bodies.
- Equivalence of link and joint naming between Drake and Pinocchio generated URDFs.
- Loaders loading the canonical golfer models by default.
- Unknown body name raising KeyError with valid names in compute_jacobian.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import defusedxml.ElementTree as ET
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.loaders import (
    load_drake_engine,
    load_mujoco_engine,
    load_pinocchio_engine,
)
from src.shared.python.engine_core.model_contracts import REQUIRED_GOLFER_BODIES

MUJOCO_DEFAULT_MODEL = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "mujoco"
    / "models"
    / "generated"
    / "golfer.xml"
)

DRAKE_DEFAULT_MODEL = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "drake"
    / "models"
    / "generated"
    / "golfer.urdf"
)

PINOCCHIO_DEFAULT_MODEL = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "generated"
    / "golfer.urdf"
)


@pytest.mark.unit
def test_required_golfer_bodies_defined() -> None:
    """The canonical list of required golfer body names must contain the key segments."""
    expected = {
        "pelvis",
        "forearm_left",
        "forearm_right",
        "hand_left",
        "hand_right",
        "club_shaft",
        "club_head",
    }
    assert expected.issubset(set(REQUIRED_GOLFER_BODIES))


@pytest.mark.unit
def test_default_models_on_disk_contain_required_bodies() -> None:
    """Default models for MuJoCo, Drake, and Pinocchio must contain all required bodies."""
    # 1. MuJoCo XML
    assert MUJOCO_DEFAULT_MODEL.exists(), f"Missing {MUJOCO_DEFAULT_MODEL}"
    mujoco_tree = ET.parse(MUJOCO_DEFAULT_MODEL)
    mujoco_bodies = {
        b.get("name") for b in mujoco_tree.findall(".//body") if b.get("name")
    }
    for body in REQUIRED_GOLFER_BODIES:
        assert body in mujoco_bodies, (
            f"MuJoCo golfer.xml missing required body '{body}'"
        )

    # 2. Pinocchio URDF
    assert PINOCCHIO_DEFAULT_MODEL.exists(), f"Missing {PINOCCHIO_DEFAULT_MODEL}"
    pin_tree = ET.parse(PINOCCHIO_DEFAULT_MODEL)
    pin_links = {
        link.get("name") for link in pin_tree.findall(".//link") if link.get("name")
    }
    for body in REQUIRED_GOLFER_BODIES:
        assert body in pin_links, (
            f"Pinocchio golfer.urdf missing required body '{body}'"
        )

    # 3. Drake URDF
    assert DRAKE_DEFAULT_MODEL.exists(), f"Missing {DRAKE_DEFAULT_MODEL}"
    drake_tree = ET.parse(DRAKE_DEFAULT_MODEL)
    drake_links = {
        link.get("name") for link in drake_tree.findall(".//link") if link.get("name")
    }
    for body in REQUIRED_GOLFER_BODIES:
        assert body in drake_links, f"Drake golfer.urdf missing required body '{body}'"


@pytest.mark.unit
def test_generated_urdfs_link_and_joint_naming_equivalence() -> None:
    """Drake and Pinocchio generated URDFs must share identical link and joint naming."""
    assert DRAKE_DEFAULT_MODEL.exists()
    assert PINOCCHIO_DEFAULT_MODEL.exists()

    drake_tree = ET.parse(DRAKE_DEFAULT_MODEL)
    pin_tree = ET.parse(PINOCCHIO_DEFAULT_MODEL)

    drake_links = sorted(
        link.get("name") for link in drake_tree.findall(".//link") if link.get("name")
    )
    pin_links = sorted(
        link.get("name") for link in pin_tree.findall(".//link") if link.get("name")
    )

    drake_joints = sorted(
        joint.get("name")
        for joint in drake_tree.findall(".//joint")
        if joint.get("name")
    )
    pin_joints = sorted(
        joint.get("name") for joint in pin_tree.findall(".//joint") if joint.get("name")
    )

    assert drake_links == pin_links, "URDF links must have identical names and counts"
    assert drake_joints == pin_joints, (
        "URDF joints must have identical names and counts"
    )


@pytest.mark.unit
def test_loader_default_model_paths() -> None:
    """Loaders must resolve to the canonical golfer models by default."""
    import inspect
    from src.engines import loaders

    mujoco_src = inspect.getsource(loaders.load_mujoco_engine)
    assert "golfer.xml" in mujoco_src, "MuJoCo loader should default to golfer.xml"
    assert "simple_pendulum.xml" not in mujoco_src, (
        "MuJoCo loader must not default to pendulum"
    )

    drake_src = inspect.getsource(loaders.load_drake_engine)
    assert "drake/models/generated/golfer.urdf" in drake_src


@pytest.mark.unit
def test_mujoco_engine_exposes_required_bodies_and_computes_jacobian() -> None:
    """MuJoCo engine with default model must expose required bodies and raise KeyError on unknown."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_path(str(MUJOCO_DEFAULT_MODEL))

    body_names = engine.get_body_names()
    for body in REQUIRED_GOLFER_BODIES:
        assert body in body_names, f"MuJoCo get_body_names() missing '{body}'"
        jac = engine.compute_jacobian(body)
        assert jac is not None
        assert "spatial" in jac
        assert "linear" in jac
        assert "angular" in jac

    with pytest.raises(
        KeyError, match=r"Body 'unknown_segment' not found in MuJoCo model"
    ):
        engine.compute_jacobian("unknown_segment")


@pytest.mark.unit
def test_pinocchio_compute_jacobian_unknown_body_raises_key_error() -> None:
    """Pinocchio compute_jacobian must raise KeyError when body/frame is not found."""
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    engine = PinocchioPhysicsEngine()
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.model.existFrame.return_value = False
    mock_frame = MagicMock()
    mock_frame.name = "pelvis"
    engine.model.frames = [mock_frame]

    with pytest.raises(
        KeyError, match=r"Body/Frame 'unknown_segment' not found in Pinocchio model"
    ):
        engine.compute_jacobian("unknown_segment")


@pytest.mark.unit
def test_drake_compute_jacobian_unknown_body_raises_key_error() -> None:
    """Drake compute_jacobian must raise KeyError when body is not found."""
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )

    engine = DrakePhysicsEngine.__new__(DrakePhysicsEngine)
    engine.plant = MagicMock()
    engine.plant_context = MagicMock()
    engine.plant.HasBodyNamed.return_value = False
    mock_body = MagicMock()
    mock_body.name.return_value = "pelvis"
    engine.plant.num_bodies.return_value = 1
    engine.plant.get_body.return_value = mock_body

    with pytest.raises(
        KeyError, match=r"Body 'unknown_segment' not found in Drake model"
    ):
        engine.compute_jacobian("unknown_segment")
