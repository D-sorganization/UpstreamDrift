from __future__ import annotations

from typing import Any

from src.shared.python.engine_core.engine_availability import MYOSUITE_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

if not MYOSUITE_AVAILABLE:
    logger.warning("MyoSuite not installed. MyoSuitePhysicsEngine will not function.")
else:
    try:
        import gymnasium as gym
    except ImportError:
        import gym  # type: ignore[no-redef]
    import myosuite  # noqa: F401


class EngineInitMixin:
    def __init__(self) -> None:
        self.env: Any = None
        self.sim: Any = None
        self.env_id: str = ""
        if not hasattr(self, "_model_override"):
            self._model_override: Any = None
        self._dt = 0.002
        self._terminated: bool = False

    def _reset_loaded_state(self) -> None:
        self.env = None
        self.sim = None
        self.env_id = ""
        self._model_override = None
        self._dt = 0.002

    @staticmethod
    def _extract_sim_from_env(env: Any) -> Any:
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim

        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            sim = getattr(unwrapped, "sim", None)
            if sim is not None:
                return sim

        raise RuntimeError(
            "Could not extract underlying MuJoCo sim object from MyoSuite env"
        )

    @property
    def model_name(self) -> str:
        return self.env_id or "MyoSuite_NoModel"

    @property
    def model(self) -> Any:
        if self.sim is not None:
            return self.sim.model
        return self._model_override

    @model.setter
    def model(self, value: Any) -> None:
        self._model_override = value

    @property
    def is_initialized(self) -> bool:
        return self.env is not None and self.sim is not None

    def _load_from_path_impl(self, path: str) -> None:
        """Engine-specific path loading (called by BasePhysicsEngine).

        For MyoSuite, 'path' is actually a Gym Environment ID string.
        """
        if not MYOSUITE_AVAILABLE:
            raise ImportError("MyoSuite not installed")

        env_id = path.strip()
        env = gym.make(env_id)
        env.reset()
        sim = self._extract_sim_from_env(env)

        self.env = env
        self.sim = sim
        self.env_id = env_id
        self._dt = self.sim.model.opt.timestep

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """String loading is not supported for MyoSuite."""
        raise RuntimeError(
            "MyoSuite does not support loading from string (requires Env ID registration)"
        )

    def load_from_path(self, path: str) -> None:
        if not MYOSUITE_AVAILABLE:
            raise ImportError("MyoSuite not installed")

        try:
            self._load_from_path_impl(path)
            self._is_initialized = True  # type: ignore[attr-defined]
        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            self._reset_loaded_state()
            logger.error(
                "Failed to load MyoSuite environment '%s': %s", path.strip(), e
            )
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        logger.error(
            "MyoSuite does not support loading from string (requires Env ID registration)"
        )
        raise RuntimeError(
            "MyoSuite does not support loading from string (requires Env ID registration)"
        )
