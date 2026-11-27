"""Consistency checks for #15016 Part A: ZVCF manual formulas, docstrings, and manual links."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
USER_MANUAL = ROOT / "docs/user_guide/upstream_drift_user_manual"
pytestmark = pytest.mark.unit


def _method_docstring(path: Path, method_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            doc = ast.get_docstring(node)
            if doc is not None:
                return doc
    raise AssertionError(f"Method {method_name!r} with docstring not found in {path}")


def test_frozen_capture_chapter_links_existing_opensim_geometry_chapter() -> None:
    chapter = ROOT / "manuals/upstreamdrift/chapters/64-frozen-capture-registration.qmd"
    target = ROOT / "manuals/upstreamdrift/chapters/60-native-opensim-geometry.qmd"
    text = chapter.read_text(encoding="utf-8")

    assert "26-native-opensim-geometry.qmd" not in text
    assert "(60-native-opensim-geometry.qmd)" in text
    assert target.is_file()


def test_dynamics_interface_counterfactual_q_shape_docstrings_match_nq() -> None:
    path = ROOT / "src/shared/python/engine_core/_dynamics_interface.py"
    for method in ("compute_ztcf", "compute_zvcf"):
        doc = _method_docstring(path, method)
        assert "Joint positions (n_q,)" in doc, f"{method} must document q as (n_q,)"
        assert "Joint positions (n_v,)" not in doc, (
            f"{method} must not document q as (n_v,)"
        )


def test_user_manual_ch05_zvcf_distinguishes_canonical_control_and_legacy_shim() -> (
    None
):
    text = (USER_MANUAL / "ch05_physics_engine_framework.md").read_text(
        encoding="utf-8"
    )
    assert r"\bigl(-g(q) + \tau" not in text
    assert (
        "a_{\\text{full}} - a_{\\text{ZVCF}}$\nisolates Coriolis/centrifugal"
        not in text
    )
    assert "compute_control_acceleration" in text
    assert "M(q)^{-1}" in text


def test_user_manual_ch06_documents_both_ztcf_and_zvcf() -> None:
    section = (USER_MANUAL / "ch06_mujoco_physics_engine.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"### 6\.2\.4 .*?(?=### 6\.2\.5)", section, re.DOTALL)
    assert match is not None
    block = match.group(0)
    assert "compute_zvcf" in block
    assert "ctrl[:] = 0" in block


def test_user_manual_ch08_drake_zvcf_zeros_velocity_and_tau() -> None:
    text = (USER_MANUAL / "ch08_drake_physics_engine.md").read_text(encoding="utf-8")
    assert r"M(q)^{-1}(\tau - g(q))" not in text
    assert r"-M(q)^{-1} g(q)" in text
    assert "compute_control_acceleration" in text


def test_user_manual_ch09_pinocchio_zvcf_zeros_tau_not_preserved() -> None:
    text = (USER_MANUAL / "ch09_pinocchio_physics_engine.md").read_text(
        encoding="utf-8"
    )
    assert r"\text{ABA}(q, \mathbf{0}, \tau)" not in text
    assert "preserving current control" not in text
    assert r"\text{ABA}(q, \mathbf{0}, \mathbf{0})" in text
    assert "compute_control_acceleration_at_state" in text


def test_user_manual_ch10_opensim_zvcf_zeros_controls() -> None:
    text = (USER_MANUAL / "ch10_opensim_physics_engine.md").read_text(encoding="utf-8")
    match = re.search(r"### 10\.6\.2 .*?(?=---)", text, re.DOTALL)
    assert match is not None
    block = match.group(0)
    assert "Controls are preserved" not in block
    assert "zero" in block.casefold()
    assert "restore" in block.casefold()


def test_user_manual_ch11_myosuite_zvcf_comment_notes_ctrl_zeroing() -> None:
    text = (USER_MANUAL / "ch11_myosuite_musculoskeletal_engine.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"### 11\.14\.4 .*?(?=---)", text, re.DOTALL)
    assert match is not None
    block = match.group(0)
    assert "ctrl=0" in block or "zeros ctrl" in block
