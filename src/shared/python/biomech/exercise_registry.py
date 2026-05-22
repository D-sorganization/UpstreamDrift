"""Exercise registry for discovering biomechanics models across sibling repos."""

from src.shared.python.config.model_source_providers import _MODEL_SOURCES


def discover_exercise(exercise_name: str) -> list[str]:
    """Return a list of engine names that provide the given exercise."""
    engines = []

    mapping = {
        "mujoco_models": "MuJoCo_Models",
        "drake_models": "Drake_Models",
        "pinocchio_models": "Pinocchio_Models",
        "opensim_models": "OpenSim_Models",
    }

    for provider_key, engine_name in mapping.items():
        if provider_key not in _MODEL_SOURCES:
            continue

        try:
            root_path = _MODEL_SOURCES[provider_key]()

            possible_paths = [
                root_path / "exercises" / exercise_name,
                root_path / "src" / provider_key / "exercises" / exercise_name,
            ]

            if any(p.exists() for p in possible_paths):
                engines.append(engine_name)
        except Exception:  # noqa: BLE001
            pass

    return engines
