from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class MuscleInterfaceMixin:
    # Attributes provided by EngineInitMixin.__init__; declared here for type checking.
    if TYPE_CHECKING:
        sim: Any

    def get_muscle_analyzer(self) -> Any | None:
        if not self.sim:
            logger.warning("Cannot create muscle analyzer - simulation not initialized")
            return None

        try:
            from .muscle_analysis import MyoSuiteMuscleAnalyzer

            return MyoSuiteMuscleAnalyzer(self.sim)

        except ImportError as e:
            logger.error(f"Failed to import muscle analyzer: {e}")
            return None

    def create_grip_model(self) -> Any | None:
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            logger.warning("Cannot create grip model - muscle analyzer not available")
            return None

        try:
            from .muscle_analysis import MyoSuiteGripModel

            return MyoSuiteGripModel(self.sim, analyzer)

        except ImportError as e:
            logger.error(f"Failed to import grip model: {e}")
            return None

    def set_muscle_activations(self, activations: dict[str, float]) -> None:
        if activations is None:
            raise ValueError("activations must be provided")
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            logger.warning("Cannot set activations - muscle analyzer unavailable")
            return

        for muscle_name, activation in activations.items():
            try:
                idx = analyzer.muscle_names.index(muscle_name)
                actuator_id = analyzer.muscle_actuator_ids[idx]
                activation_clamped = max(0.0, min(1.0, activation))

                try:
                    ctrl = self.sim.data.ctrl

                    if (
                        hasattr(ctrl, "__len__")
                        and actuator_id < len(ctrl)
                        or hasattr(ctrl, "__setitem__")
                    ):
                        self.sim.data.ctrl[actuator_id] = activation_clamped

                except (TypeError, AttributeError, IndexError):
                    pass

            except ValueError:
                logger.warning(f"Muscle '{muscle_name}' not found")

            except (RuntimeError, OSError) as e:
                logger.error(f"Failed to set activation for '{muscle_name}': {e}")

    def compute_muscle_induced_accelerations(self) -> dict[str, np.ndarray]:
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            return {}

        return dict(analyzer.compute_muscle_induced_accelerations())

    def analyze_muscle_contributions(self) -> Any | None:
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            logger.warning("Cannot analyze muscles - analyzer not available")
            return None

        return analyzer.analyze_all()

    def get_muscle_state(self) -> Any | None:
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            return None

        from .muscle_analysis import MyoSuiteMuscleState

        return MyoSuiteMuscleState(
            muscle_names=analyzer.muscle_names,
            activations=analyzer.get_muscle_activations(),
            forces=analyzer.get_muscle_forces(),
            lengths=analyzer.get_muscle_lengths(),
            velocities=analyzer.get_muscle_velocities(),
        )

    def get_muscle_names(self) -> list[str]:
        analyzer = self.get_muscle_analyzer()

        if analyzer is None:
            return []

        return list(analyzer.muscle_names)
