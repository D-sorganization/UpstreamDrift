"""Sanity tests on model_generation.core.constants."""

from __future__ import annotations

import math

from model_generation.core import constants as C


def test_gravity_and_densities_positive() -> None:
    assert C.GRAVITY_M_S2 > 0
    assert C.TISSUE_DENSITY_KG_M3 > 0
    assert C.WATER_DENSITY_KG_M3 > 0
    assert C.BONE_DENSITY_KG_M3 > C.WATER_DENSITY_KG_M3
    assert C.FAT_DENSITY_KG_M3 < C.WATER_DENSITY_KG_M3


def test_default_density_matches_tissue() -> None:
    assert C.DEFAULT_DENSITY_KG_M3 == C.TISSUE_DENSITY_KG_M3


def test_angle_constants() -> None:
    assert 2 * math.pi == C.FULL_ROTATION_RAD
    assert C.JOINT_LIMIT_SMALL < C.JOINT_LIMIT_MEDIUM < C.JOINT_LIMIT_LARGE
    assert C.JOINT_LIMIT_LARGE < C.JOINT_LIMIT_FULL


def test_humanoid_defaults() -> None:
    assert C.HUMANOID_SEGMENT_COUNT > 0
    assert C.DEFAULT_HEIGHT_M > 0
    assert C.DEFAULT_MASS_KG > 0


def test_mesh_thresholds() -> None:
    assert 0 < C.COLLISION_MESH_SIMPLIFICATION <= 1
    assert C.MIN_COLLISION_FACES < C.MAX_VISUAL_FACES


def test_tolerances() -> None:
    assert C.FLOAT_TOLERANCE > 0
    assert C.MIN_MASS_KG > 0
    assert C.MIN_INERTIA_KG_M2 > 0


def test_urdf_constants() -> None:
    assert C.URDF_INDENT == "  "
    assert "<?xml" in C.URDF_XML_DECLARATION
    assert C.DEFAULT_ROBOT_NAME
