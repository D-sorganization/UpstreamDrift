"""CI helper: adds the 14 post-#5556 audit tiles to launcher_manifest.json.

Run from repo root: python scripts/_add_audit_tiles.py
This script is idempotent (won't duplicate tiles).
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path("src/config/launcher_manifest.json")

NEW_TILES = [
    {
        "id": "drake_dashboard",
        "name": "Drake Dashboard",
        "description": "Interactive Drake simulation dashboard",
        "category": "physics_engine",
        "type": "special_app",
        "path": "src/launchers/drake_dashboard.py",
        "engine_type": "drake",
        "logo": "drake.svg",
        "status": "experimental",
        "web_route": "/tools/drake-dashboard",
        "capabilities": ["rigid_body", "optimization", "dashboard"],
        "order": 29,
    },
    {
        "id": "mujoco_dashboard",
        "name": "MuJoCo Dashboard",
        "description": "Interactive MuJoCo simulation dashboard",
        "category": "physics_engine",
        "type": "special_app",
        "path": "src/launchers/mujoco_dashboard.py",
        "engine_type": "mujoco",
        "logo": "mujoco_humanoid.svg",
        "status": "experimental",
        "web_route": "/tools/mujoco-dashboard",
        "capabilities": ["rigid_body", "contact", "dashboard"],
        "order": 30,
    },
    {
        "id": "pinocchio_dashboard",
        "name": "Pinocchio Dashboard",
        "description": "Interactive Pinocchio simulation dashboard",
        "category": "physics_engine",
        "type": "special_app",
        "path": "src/launchers/pinocchio_dashboard.py",
        "engine_type": "pinocchio",
        "logo": "pinocchio.svg",
        "status": "experimental",
        "web_route": "/tools/pinocchio-dashboard",
        "capabilities": ["rigid_body", "inverse_kinematics", "dashboard"],
        "order": 31,
    },
    {
        "id": "analysis_tools_api",
        "name": "Analysis Tools",
        "description": "REST API access to 6 analysis endpoints (swing metrics, biomechanics)",
        "category": "analysis",
        "type": "special_app",
        "path": "src/api/routes/analysis_tools.py",
        "web_route": "/api/analysis",
        "logo": "data_explorer.svg",
        "status": "ready",
        "capabilities": ["swing_metrics", "biomechanics", "api"],
        "order": 32,
    },
    {
        "id": "motion_pipeline",
        "name": "Motion Pipeline",
        "description": "Full markerless mocap to forward-dynamics pipeline (video in, motion out)",
        "category": "motion_matching",
        "type": "special_app",
        "path": "src/shared/python/motion_pipeline/api.py",
        "web_route": "/tools/motion-pipeline",
        "logo": "c3d_icon.svg",
        "status": "experimental",
        "capabilities": ["markerless_mocap", "forward_dynamics", "pipeline"],
        "order": 33,
    },
    {
        "id": "perturbation_analysis",
        "name": "Perturbation Analysis",
        "description": "Cross-engine robustness analysis via perturbation testing",
        "category": "analysis",
        "type": "special_app",
        "path": "src/api/routes/physics.py",
        "web_route": "/api/perturbation",
        "logo": "data_explorer.svg",
        "status": "experimental",
        "capabilities": ["perturbation", "robustness", "cross_engine"],
        "order": 34,
    },
    {
        "id": "force_overlays",
        "name": "Force Overlays",
        "description": "Physics visualization: joint forces and torques overlaid on 3D model",
        "category": "tool",
        "type": "special_app",
        "path": "src/api/routes/force_overlays.py",
        "web_route": "/api/force-overlays",
        "logo": "mujoco_humanoid.svg",
        "status": "experimental",
        "capabilities": ["force_visualization", "joint_torques", "3d_overlay"],
        "order": 35,
    },
    {
        "id": "realtime_ws",
        "name": "Realtime WebSocket",
        "description": "Live simulation data stream via WebSocket pub-sub",
        "category": "tool",
        "type": "special_app",
        "path": "src/api/routes/realtime.py",
        "web_route": "/ws/realtime",
        "logo": "data_explorer.svg",
        "status": "experimental",
        "capabilities": ["websocket", "realtime", "pubsub"],
        "order": 36,
    },
    {
        "id": "aip",
        "name": "AI Protocol (AIP)",
        "description": "AI-native simulation control protocol with structured method dispatch",
        "category": "tool",
        "type": "special_app",
        "path": "src/api/routes/aip.py",
        "web_route": "/api/aip",
        "logo": "golf_logo.svg",
        "status": "experimental",
        "capabilities": ["ai_methods", "protocol", "dispatch"],
        "order": 37,
    },
    {
        "id": "actuator_controls",
        "name": "Actuator Controls",
        "description": "Live actuator parameter tuning and control scheme editor",
        "category": "tool",
        "type": "special_app",
        "path": "src/api/routes/actuator_controls.py",
        "web_route": "/api/actuator-controls",
        "logo": "mujoco_humanoid.svg",
        "status": "experimental",
        "capabilities": ["actuator_tuning", "control_schemes", "realtime"],
        "order": 38,
    },
    {
        "id": "unreal_integration",
        "name": "Unreal Integration",
        "description": "Unreal Engine streaming and VR visualization for golf swing simulation",
        "category": "tool",
        "type": "special_app",
        "path": "src/unreal_integration/__init__.py",
        "web_route": "/docs/unreal-integration",
        "logo": "golf_logo.svg",
        "status": "external",
        "capabilities": ["streaming", "vr", "visualization"],
        "order": 39,
    },
    {
        "id": "robotics_module",
        "name": "Robotics Module",
        "description": "Locomotion planning, contact mechanics, and robot control for humanoid models",
        "category": "tool",
        "type": "special_app",
        "path": "src/robotics/__init__.py",
        "web_route": "/docs/robotics",
        "logo": "drake.svg",
        "status": "experimental",
        "capabilities": ["locomotion", "contact", "planning"],
        "order": 40,
    },
    {
        "id": "tools_calculator_hub",
        "name": "Tools Calculator Suite",
        "description": "Engineering process calculators from the Tools repo (50+ calculators)",
        "category": "tool",
        "type": "special_app",
        "path": "src/data_processing/data_processor/launch_pyqt6.py",
        "web_route": "/tools/calculators",
        "logo": "data_explorer.svg",
        "status": "external",
        "capabilities": ["process_calculators", "engineering", "analysis"],
        "order": 41,
    },
    {
        "id": "pid_generator",
        "name": "P&ID Generator",
        "description": "Programmatic P&ID Generator from the Tools repo - create ISA-5.1 piping diagrams",
        "category": "tool",
        "type": "special_app",
        "path": "src/tools/pid_generator/__main__.py",
        "web_route": "/docs/pid-generator",
        "logo": "data_explorer.svg",
        "status": "external",
        "capabilities": ["pid_generation", "isa_5_1", "diagrams"],
        "order": 42,
    },
]


def main() -> None:
    """Add new tiles to the manifest (idempotent)."""
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing_ids = {t["id"] for t in data["tiles"]}

    # Find the starting_pose_matcher (order=99) to insert before it
    spm_index = next(
        i for i, t in enumerate(data["tiles"]) if t["id"] == "starting_pose_matcher"
    )

    added = 0
    for tile in NEW_TILES:
        if tile["id"] not in existing_ids:
            data["tiles"].insert(spm_index, tile)
            spm_index += 1
            added += 1

    MANIFEST_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Added {added} tiles. Total: {len(data['tiles'])}")


if __name__ == "__main__":
    main()
