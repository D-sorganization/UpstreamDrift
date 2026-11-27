#!/usr/bin/env python3
"""Regenerate scripts/config/affine_drift_theory_links.v1.json (Refs #15016)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELP_ROOT = REPO_ROOT / "docs" / "help"
OUT_PATH = REPO_ROOT / "scripts" / "config" / "affine_drift_theory_links.v1.json"
AD = "https://affinedrift.com"


def _url(path: str) -> str:
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    if not path.endswith(".html") and not path.endswith("/"):
        path += ".html"
    return f"{AD}{path}"


THEMES: dict[str, tuple[str, str]] = {
    "intro": ("AffineDrift Theory Part I", "/articles/theory-part1.html"),
    "decomposition": (
        "Drift Is the Past, Control Is the Present",
        "/articles/drift-is-the-past-control-is-the-present.html",
    ),
    "ztcf": (
        "Zero-Torque Counterfactual (ZTCF) Family",
        "/articles/zero-torque-counterfactual.html",
    ),
    "pendulum": (
        "Drift Components on a Wrench Double Pendulum",
        "/articles/drift-components-wrench-double-pendulum.html",
    ),
    "engines": (
        "The Physics of Golf — Forces and Torques",
        "/articles/The_Physics_of_Golf/quarto/ch04_forces_and_torques.html",
    ),
    "impact": (
        "Impact Mechanics and Ball Flight",
        "/articles/impact-mechanics-and-ball-flight.html",
    ),
    "putting": ("Putting Roll Models", "/articles/putting-roll-models.html"),
    "grf": (
        "Ground Reaction Forces",
        "/articles/The_Physics_of_Golf/quarto/ch15_ground_reaction_forces.html",
    ),
    "nullspace": (
        "After One Solution: Null-Space Exploration",
        "/articles/after-one-solution-null-space-exploration.html",
    ),
    "nullspace_jac": (
        "Null-Space Constraint Jacobian",
        "/articles/null-space-constraint-jacobian.html",
    ),
    "shot_pattern": (
        "Shot Pattern Analysis",
        "/articles/shot-pattern-analysis.html",
    ),
    "proximal": (
        "Proximal-to-Distal Energy Transfer",
        "/articles/proximal_distal_energy_transfer/index.html",
    ),
    "id": ("Inverse Dynamics", "/articles/inverse-dynamics.html"),
    "screw": ("Screw Theory Reference", "/articles/screw-theory-reference.html"),
    "mocap": (
        "Markerless Motion Capture Camera Selection",
        "/articles/markerless-mocap-camera-selection.html",
    ),
    "dof": (
        "Degrees of Freedom and Dimensionality",
        "/articles/degrees-of-freedom-and-dimensionality.html",
    ),
    "control": (
        "Passive and Distributed Control",
        "/articles/passive-distributed-control.html",
    ),
    "constraints": (
        "Constraint Forces",
        "/articles/The_Physics_of_Golf/quarto/ch07_constraint_forces.html",
    ),
    "sidekick": (
        "Core Theses Registry",
        "/articles/core-theses-registry.html",
    ),
    "rate_closure": (
        "Rotation-Induced Spin",
        "/articles/rotation-induced-spin.html",
    ),
}

RULES: dict[str, str] = {
    "analysis_tools_api.md": "intro",
    "analysis_tools.md": "intro",
    "anatomical_event_table.md": "engines",
    "ball_flight_simulator.md": "impact",
    "biomech_exercise.md": "id",
    "biomech_gait.md": "id",
    "biomech_sit_to_stand.md": "id",
    "c3d_viewer.md": "mocap",
    "capture_rig.md": "mocap",
    "character_builder.md": "dof",
    "chat_assistant.md": "sidekick",
    "config_setup_wizard.md": "intro",
    "cross_engine_dashboard.md": "decomposition",
    "data_explorer.md": "intro",
    "data_processor.md": "intro",
    "dataset_generator.md": "intro",
    "drake_golf.md": "decomposition",
    "drake_models_shared.md": "engines",
    "engine_selection.md": "engines",
    "golf_environment.md": "engines",
    "golf_simulation_suite.md": "decomposition",
    "golf_simulator.md": "impact",
    "grip_wrench_plots.md": "nullspace_jac",
    "ground_reaction_plots.md": "grf",
    "impact_parameters.md": "impact",
    "launch_monitor_analytics.md": "shot_pattern",
    "library_tool.md": "intro",
    "loop_nullspace.md": "nullspace",
    "loop_nullspace_variability.md": "nullspace",
    "matched_swing_browser.md": "proximal",
    "matlab_suite.md": "engines",
    "mediapipe_analysis.md": "mocap",
    "model_explorer.md": "dof",
    "motion_capture.md": "mocap",
    "motion_matching.md": "proximal",
    "motion_target_preview.md": "proximal",
    "movement_optimizer.md": "id",
    "mujoco_models_shared.md": "engines",
    "mujoco_unified.md": "decomposition",
    "myosim_suite.md": "engines",
    "necromatcher.md": "proximal",
    "openpose_analysis.md": "mocap",
    "opensim_golf.md": "decomposition",
    "opensim_models_shared.md": "engines",
    "pendulum_simulator.md": "pendulum",
    "pinocchio_golf.md": "decomposition",
    "pinocchio_models_shared.md": "engines",
    "pose_studio.md": "dof",
    "project_map.md": "intro",
    "putting_analytics.md": "putting",
    "putting_green.md": "putting",
    "rate_of_closure.md": "rate_closure",
    "shot_pattern_analysis.md": "shot_pattern",
    "shot_tracer.md": "impact",
    "sidekick.md": "sidekick",
    "simulation_backends.md": "ztcf",
    "simulation_controls.md": "control",
    "spine_viewer.md": "id",
    "swing_flight_pipeline.md": "impact",
    "swing_objective_lab.md": "id",
    "terrain_engine.md": "constraints",
    "tools_calculator_hub.md": "intro",
    "tour_matching_viewer.md": "proximal",
    "training_controller.md": "control",
    "video_analyzer.md": "mocap",
    "video_processor.md": "mocap",
    "visualization.md": "intro",
}


def theme(name: str) -> dict[str, str]:
    title, path = THEMES[name]
    return {"title": title, "url": _url(path)}


def main() -> int:
    pages: dict[str, dict[str, str]] = {}
    for path in sorted(HELP_ROOT.glob("*.md")):
        rule = RULES.get(path.name)
        if rule is None:
            raise SystemExit(f"missing theory rule for {path.name}")
        pages[path.stem] = theme(rule)
    payload = {
        "registry_id": "upstreamdrift-affine-drift-theory-links",
        "version": "1.0.0",
        "pages": pages,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
