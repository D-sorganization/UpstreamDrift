"""Unified engine interface with standardized model loading.

This module provides a unified interface for all physics engines with:
- Standardized model loading across all engines
- Automatic model validation and compatibility checking
- Consistent API regardless of underlying engine
- Professional error handling and diagnostics
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from shared.python.engine_manager import EngineManager
from shared.python.engine_registry import EngineType
from shared.python.interfaces import PhysicsEngine
from shared.python.standard_models import StandardModelManager

logger = logging.getLogger(__name__)


class UnifiedEngineInterface:
    """Unified interface for all physics engines with standardized model loading."""

    def __init__(self, suite_root: Path | None = None):
        """Initialize unified engine interface.

        Args:
            suite_root: Root directory of Golf Modeling Suite
        """
        self.suite_root = suite_root or Path(__file__).parent.parent.parent
        self.engine_manager = EngineManager(self.suite_root)
        self.model_manager = StandardModelManager(self.suite_root)

        self.current_engine: PhysicsEngine | None = None
        self.current_engine_type: EngineType | None = None
        self.loaded_model_path: Path | None = None

    def load_engine(
        self, engine_type: str | EngineType, load_standard_model: bool = True
    ) -> bool:
        """Load a physics engine with optional standard model.

        Args:
            engine_type: Type of engine to load
            load_standard_model: Whether to load standard humanoid model

        Returns:
            True if engine loaded successfully
        """
        try:
            # Convert string to enum if needed
            if isinstance(engine_type, str):
                engine_type = EngineType(engine_type.upper())

            # Load engine
            self.engine_manager._load_engine(engine_type)

            # Get active engine to check if loading was successful
            self.current_engine = self.engine_manager.get_active_physics_engine()
            if not self.current_engine:
                logger.error(f"Failed to load engine: {engine_type}")
                return False
            self.current_engine_type = engine_type

            if not self.current_engine:
                logger.error("No active engine after loading")
                return False

            # Load standard model if requested
            if load_standard_model:
                return self.load_standard_humanoid()

            return True

        except Exception as e:
            logger.error(f"Error loading engine {engine_type}: {e}")
            return False

    def load_standard_humanoid(self) -> bool:
        """Load the standard humanoid model into current engine.

        Returns:
            True if model loaded successfully
        """
        if not self.current_engine:
            logger.error("No active engine to load model into")
            return False

        try:
            # Get standard humanoid path
            humanoid_path = self.model_manager.get_standard_humanoid_path()

            # Load model into current engine
            self.current_engine.load_from_path(str(humanoid_path))
            self.loaded_model_path = humanoid_path

            logger.info(f"Loaded standard humanoid into {self.current_engine_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to load standard humanoid: {e}")
            return False

    def load_golf_club(self, club_type: str = "driver") -> bool:
        """Load a golf club model.

        Args:
            club_type: Type of golf club to load

        Returns:
            True if club loaded successfully
        """
        if not self.current_engine:
            logger.error("No active engine to load club into")
            return False

        try:
            club_path = self.model_manager.get_golf_club_path(club_type)

            # For now, we'll need to implement club attachment logic
            # This would involve creating a composite model with human + club
            logger.info(f"Golf club {club_type} available at {club_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load golf club {club_type}: {e}")
            return False

    def validate_current_model(self) -> dict[str, Any]:
        """Validate the currently loaded model.

        Returns:
            Validation results with compatibility and diagnostics
        """
        if not self.loaded_model_path:
            return {"valid": False, "error": "No model loaded"}

        try:
            # Run model validation
            compatibility = self.model_manager.validate_model_compatibility(
                self.loaded_model_path
            )

            # Get model info
            model_info = {
                "path": str(self.loaded_model_path),
                "engine": (
                    self.current_engine_type.value if self.current_engine_type else None
                ),
                "compatibility": compatibility,
            }

            # Check if current engine is compatible
            current_engine_name = (
                self.current_engine_type.value.lower()
                if self.current_engine_type
                else None
            )
            is_compatible = compatibility.get(current_engine_name or "", False)

            return {
                "valid": is_compatible,
                "model_info": model_info,
                "compatibility": compatibility,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Model information dictionary
        """
        if not self.current_engine:
            return {"error": "No active engine"}

        try:
            # Get basic model info
            info: dict[str, Any] = {
                "engine": (
                    self.current_engine_type.value if self.current_engine_type else None
                ),
                "model_name": self.current_engine.model_name,
                "model_path": (
                    str(self.loaded_model_path) if self.loaded_model_path else None
                ),
            }

            # Get state information if available
            try:
                q, v = self.current_engine.get_state()
                info["dof"] = len(q)
                info["state_size"] = {"positions": len(q), "velocities": len(v)}
            except Exception:
                pass

            return info

        except Exception as e:
            return {"error": str(e)}

    def reset_simulation(self) -> bool:
        """Reset the current simulation to initial state.

        Returns:
            True if reset successful
        """
        if not self.current_engine:
            return False

        try:
            self.current_engine.reset()
            return True
        except Exception as e:
            logger.error(f"Failed to reset simulation: {e}")
            return False

    def step_simulation(self, dt: float | None = None) -> bool:
        """Step the simulation forward.

        Args:
            dt: Time step (uses engine default if None)

        Returns:
            True if step successful
        """
        if not self.current_engine:
            return False

        try:
            self.current_engine.step(dt)
            return True
        except Exception as e:
            logger.error(f"Failed to step simulation: {e}")
            return False

    def get_state(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get current simulation state.

        Returns:
            Tuple of (positions, velocities) or None if error
        """
        if not self.current_engine:
            return None

        try:
            return self.current_engine.get_state()
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> bool:
        """Set simulation state.

        Args:
            positions: Joint positions
            velocities: Joint velocities

        Returns:
            True if state set successfully
        """
        if not self.current_engine:
            return False

        try:
            self.current_engine.set_state(positions, velocities)
            return True
        except Exception as e:
            logger.error(f"Failed to set state: {e}")
            return False

    def apply_control(self, control_inputs: np.ndarray) -> bool:
        """Apply control inputs to the simulation.

        Args:
            control_inputs: Control vector (torques/forces)

        Returns:
            True if control applied successfully
        """
        if not self.current_engine:
            return False

        try:
            self.current_engine.set_control(control_inputs)
            return True
        except Exception as e:
            logger.error(f"Failed to apply control: {e}")
            return False

    def get_available_engines(self) -> list[str]:
        """Get list of available physics engines.

        Returns:
            List of available engine names
        """
        available = self.engine_manager.get_available_engines()
        return [engine.value for engine in available]

    def get_available_models(self) -> dict[str, Any]:
        """Get list of available standard models.

        Returns:
            Dictionary of available models
        """
        return self.model_manager.list_available_models()

    def setup_all_models(self) -> bool:
        """Download and setup all standard models.

        Returns:
            True if all models setup successfully
        """
        return self.model_manager.setup_all_models()


# Convenience functions for common operations
def create_unified_interface(suite_root: Path | None = None) -> UnifiedEngineInterface:
    """Create a unified engine interface.

    Args:
        suite_root: Root directory of Golf Modeling Suite

    Returns:
        Configured UnifiedEngineInterface instance
    """
    return UnifiedEngineInterface(suite_root)


def quick_setup(
    engine_type: str = "mujoco", suite_root: Path | None = None
) -> UnifiedEngineInterface:
    """Quick setup with standard configuration.

    Args:
        engine_type: Physics engine to use
        suite_root: Root directory of Golf Modeling Suite

    Returns:
        Configured interface with engine and standard model loaded
    """
    interface = UnifiedEngineInterface(suite_root)

    # Load engine with standard model
    success = interface.load_engine(engine_type, load_standard_model=True)

    if not success:
        logger.warning(f"Failed to load {engine_type}, trying MuJoCo as fallback")
        interface.load_engine("mujoco", load_standard_model=True)

    return interface
