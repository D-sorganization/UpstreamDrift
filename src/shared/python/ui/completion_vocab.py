"""Centralized completion vocabulary service for AutoCompleteLineEdit.

Provides a bootstrap function that seeds the global text-prediction
dictionary from physics-engine names, common parameter identifiers,
and constants declared in ``src/config/``.
"""

from __future__ import annotations

from pathlib import Path

# Physics engine names surfaced throughout the launcher UI.
_ENGINE_NAMES: list[str] = [
    "mujoco",
    "drake",
    "pinocchio",
    "myosuite",
    "opensim",
]

# Common physics / simulation parameter names that appear in dialogs,
# search queries, and the AI composer.
_PHYSICS_TERMS: list[str] = [
    # Kinematics / dynamics
    "gravity",
    "velocity",
    "acceleration",
    "angular_velocity",
    "angular_acceleration",
    "mass",
    "inertia",
    "torque",
    "force",
    "friction",
    "damping",
    "stiffness",
    "compliance",
    "restitution",
    # Simulation controls
    "timestep",
    "duration",
    "max_simulation_time",
    "default_timestep",
    "num_steps",
    # Biomechanics / motion-capture terms
    "joint",
    "link",
    "body",
    "constraint",
    "contact",
    "tendon",
    "actuator",
    "muscle",
    "pelvis",
    "spine",
    "femur",
    "tibia",
    "foot",
    # Golf-specific
    "swing",
    "clubhead",
    "ball_flight",
    "launch_angle",
    "spin_rate",
    "carry_distance",
    # Visualization
    "live_viz",
    "render",
    "camera",
    "fps",
    # Runtime / infrastructure
    "docker",
    "wsl",
    "native",
    "gpu",
    "cpu",
    "cuda",
]

# Model-registry search shortcuts that users commonly type.
_MODEL_SHORTCUTS: list[str] = [
    "biomechanics",
    "simulation",
    "motion_capture",
    "motion_matching",
    "reinforcement_learning",
    "research",
    "tools",
]


def _load_engine_names_from_config() -> list[str]:
    """Read the preferred engine list from ``interim_config.yaml`` if available.

    Falls back to the hard-coded :data:`_ENGINE_NAMES` list on any error so
    that the vocabulary service never raises at import time.
    """
    try:
        config_path = (
            Path(__file__).resolve().parents[4] / "config" / "interim_config.yaml"
        )
        if not config_path.exists():
            return []
        import re

        text = config_path.read_text(encoding="utf-8")
        in_section = False
        found: list[str] = []
        for line in text.splitlines():
            if re.match(r"\s*preferred_order\s*:", line):
                in_section = True
                continue
            if in_section:
                item = re.match(r'\s*-\s*"?([a-zA-Z0-9_]+)"?', line)
                if item:
                    found.append(item.group(1))
                elif re.match(r"\s{0,4}\S", line):
                    break
        return found
    except Exception:  # noqa: BLE001
        return []


def build_vocabulary() -> list[str]:
    """Return the merged, deduplicated completion vocabulary.

    The vocabulary is assembled from:

    * Physics engine names (config-sourced, with a hard-coded fallback)
    * Common physics / simulation parameter names
    * Model-registry category shortcuts

    Returns
    -------
    list[str]
        Sorted, deduplicated list of completion terms.
    """
    config_engines = _load_engine_names_from_config()
    engines = config_engines or _ENGINE_NAMES

    merged: set[str] = set(engines) | set(_PHYSICS_TERMS) | set(_MODEL_SHORTCUTS)
    return sorted(merged)
