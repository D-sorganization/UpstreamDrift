"""Cycle 2 TDD contracts for Shadow hand organic skin envelope and bilateral wrap."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.grip_hands import grip_hand_meshes
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
from src.shared.python.model_appearance.hand_backends import shadow, shadow_solver

pytestmark = pytest.mark.unit
mujoco = pytest.importorskip("mujoco")


def _iron7() -> ClubAssembly:
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


def test_both_lead_and_trail_curl_onto_grip_cylinder() -> None:
    """Both lead and trail hands flex strongly and wrap pads snugly around the grip."""
    spec = GripSpec.neutral()
    cfg = shadow.visual_settings()
    grip_radius_m = 0.0125
    target = grip_radius_m + cfg.pad_clearance_m
    for hand, side in ((Hand.LEAD, "lead"), (Hand.TRAIL, "trail")):
        spec_hand = spec.trail if hand == Hand.TRAIL else spec.lead
        solved = shadow.solve_shadow_hand_qpos(
            hand, spec_hand, grip_spec=spec, grip_radius_m=grip_radius_m
        )
        model, data = shadow_solver._forward_model(hand, solved.qpos, spec.handedness)
        for prefix in ("ff", "mf", "rf"):
            q_vals = [
                float(
                    solved.qpos[
                        shadow_solver._joint_qpos_index(
                            model,
                            f"{shadow_solver._prefix(hand, spec.handedness)}{prefix.upper()}J{j}",
                        )
                    ]
                )
                for j in (3, 2, 1)
            ]
            assert q_vals[0] >= 0.45, f"{side} {prefix} MCP J3 too open: {q_vals}"
            assert max(q_vals) >= 0.70, f"{side} {prefix} too flat: {q_vals}"
            assert sum(q_vals) >= 1.75, (
                f"{side} {prefix} insufficient total curl wrap: sum={sum(q_vals):.3f}"
            )
        vis_mesh = shadow.build_hand_visual(
            spec, side, None, grip_radius_m=grip_radius_m
        ).mesh
        palm_max_r = float(
            np.hypot(vis_mesh.vertices[:, 0], vis_mesh.vertices[:, 1]).max()
        )
        assert palm_max_r <= 0.084, (
            f"{side} skin hand protrudes too far radially: max_r={palm_max_r:.4f} m"
        )
        gaps = shadow.finger_pad_contact_gaps(
            hand, solved, grip_radius_m=grip_radius_m, settings=cfg
        )
        for digit in ("index", "middle", "ring", "thumb"):
            pts = shadow_solver._select_part_vertices(
                model, data, cfg, finger=digit, segments=("middle", "distal")
            )
            mounted = shadow_solver._transform_to_grip_frame(pts, solved.mount_matrix)
            min_r = float(np.hypot(mounted[:, 0], mounted[:, 1]).min())
            clearance = min_r - grip_radius_m
            assert -cfg.club_penetration_tolerance_m <= clearance <= 0.0035, (
                f"{side} {digit} clearance={clearance:.4f} (gap={gaps[digit]:.4f}, target={target:.4f})"
            )


def test_bilateral_club_placement_seats_trail_adjacent_to_lead() -> None:
    """In club coordinates, trail hand sits snugly below lead along the grip shaft."""
    meshes = grip_hand_meshes(_iron7(), DEFAULT_GRIP_POSE, grip_spec=GripSpec.legacy())
    lead_y = float(meshes["lead_hand"].vertices[:, 1].mean())
    trail_y = float(meshes["trail_hand"].vertices[:, 1].mean())
    axial_sep = abs(trail_y - lead_y)
    assert 0.045 <= axial_sep <= 0.095, (
        f"Expected compact bilateral axial separation in [0.045, 0.095] m, got {axial_sep:.4f}"
    )


def test_organic_skin_envelope_watertight_tapered_and_debug_shell_toggle() -> None:
    """Default skin_mode builds a smooth watertight tapered skin envelope with robot debug fallback."""
    cfg = shadow.visual_settings()
    assert cfg.skin_mode == "lofted_envelope"
    spec = GripSpec.legacy()
    result = shadow.build_hand_visual(spec, "lead", None, grip_radius_m=0.012)
    mesh = result.mesh
    assert len(mesh.vertices) > 20_000
    assert len(mesh.faces) > 20_000
    assert mesh.volume() > 0.0

    edge_counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in mesh.faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge_counts[(u, v) if u < v else (v, u)] += 1
    non_manifold = [e for e, cnt in edge_counts.items() if cnt != 2]
    assert non_manifold == [], (
        f"Expected watertight shells, found {len(non_manifold)} open edges"
    )

    tris = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    assert int((areas < 1e-12).sum()) == 0, "Found degenerate zero-area triangles"

    # Debug fallback still produces the posed MuJoCo robot shell.
    robot_cfg = replace(cfg, skin_mode="robot_shell")
    robot_res = shadow.build_hand_visual(
        spec, "lead", None, grip_radius_m=0.012, settings=robot_cfg
    )
    assert len(robot_res.mesh.vertices) > 20_000
    assert robot_res.mesh.vertices.shape != mesh.vertices.shape

    # Organic envelope tapers digits distally and cups palm.
    from src.shared.python.model_appearance.hand_backends.shadow_skin import (
        measure_shadow_skin_envelope,
    )

    metrics = measure_shadow_skin_envelope(
        spec, "lead", grip_radius_m=0.012, settings=cfg
    )
    assert metrics["cup_depth_m"] >= 0.0020
    for finger in ("index", "middle", "ring", "little"):
        assert (
            metrics[f"{finger}_distal_width_m"]
            < metrics[f"{finger}_middle_width_m"]
            < metrics[f"{finger}_proximal_width_m"]
        )
