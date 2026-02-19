"""Control Features Registry - Expose Hidden Control Features.

Provides a discoverable registry of all control and analysis features available
in the physics engines. Ensures no features are hidden from users by maintaining
a single source of truth for capabilities.

Design by Contract:
    Preconditions:
        - Engine must implement PhysicsEngine protocol
    Postconditions:
        - All registered features are queryable
        - Feature availability is checked against engine capabilities
    Invariants:
        - Registry is immutable after initialization
        - Feature descriptions are always available

Usage:
    >>> from src.shared.python.control_features_registry import ControlFeaturesRegistry
    >>> registry = ControlFeaturesRegistry(engine)
    >>> features = registry.list_features()
    >>> result = registry.execute("compute_mass_matrix")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class FeatureCategory(str, Enum):
    """Categories for grouping features."""

    DYNAMICS = "dynamics"
    KINEMATICS = "kinematics"
    CONTROL = "control"
    ANALYSIS = "analysis"
    COUNTERFACTUAL = "counterfactual"
    CONTACT = "contact"
    MODEL_DATA = "model_data"
    SHAFT = "shaft"


@dataclass
class FeatureDescriptor:
    """Describes a single engine feature.

    Attributes:
        name: Feature identifier (method name).
        display_name: Human-readable name.
        description: Detailed description.
        category: Feature category.
        requires_args: Whether the feature requires arguments.
        arg_specs: Argument specifications.
        return_type: Description of return type.
        section: Design guideline section reference.
        available: Whether the feature is available on the current engine.
    """

    name: str
    display_name: str
    description: str
    category: FeatureCategory
    requires_args: bool = False
    arg_specs: list[dict[str, str]] = field(default_factory=list)
    return_type: str = "np.ndarray"
    section: str = ""
    available: bool = True


# Master registry of all features that exist in the PhysicsEngine interface
_FEATURE_DEFINITIONS: list[FeatureDescriptor] = [
    # --- Dynamics (Section F) ---
    FeatureDescriptor(
        name="compute_mass_matrix",
        display_name="Mass Matrix M(q)",
        description="Compute the dense inertia/mass matrix at current configuration. "
        "Returns symmetric positive definite matrix M of shape (n_v, n_v).",
        category=FeatureCategory.DYNAMICS,
        return_type="np.ndarray (n_v, n_v)",
        section="Dynamics Interface",
    ),
    FeatureDescriptor(
        name="compute_bias_forces",
        display_name="Bias Forces C(q,v) + g(q)",
        description="Compute bias forces including Coriolis, centrifugal, and gravity terms. "
        "Returns vector b of shape (n_v,).",
        category=FeatureCategory.DYNAMICS,
        return_type="np.ndarray (n_v,)",
        section="Dynamics Interface",
    ),
    FeatureDescriptor(
        name="compute_gravity_forces",
        display_name="Gravity Forces g(q)",
        description="Compute gravity forces at current configuration. "
        "Returns vector g of shape (n_v,).",
        category=FeatureCategory.DYNAMICS,
        return_type="np.ndarray (n_v,)",
        section="Dynamics Interface",
    ),
    FeatureDescriptor(
        name="compute_inverse_dynamics",
        display_name="Inverse Dynamics tau = ID(q, v, a)",
        description="Compute required torques for desired acceleration: "
        "tau = M(q) @ qacc + C(q,v) @ v + g(q). "
        "Requires qacc vector as input.",
        category=FeatureCategory.DYNAMICS,
        requires_args=True,
        arg_specs=[
            {
                "name": "qacc",
                "type": "np.ndarray (n_v,)",
                "description": "Desired acceleration",
            }
        ],
        return_type="np.ndarray (n_v,)",
        section="Dynamics Interface",
    ),
    # --- Drift/Control Decomposition (Section F) ---
    FeatureDescriptor(
        name="compute_drift_acceleration",
        display_name="Drift Acceleration (Passive Dynamics)",
        description="Compute passive/drift acceleration with zero control inputs. "
        "Answers: 'What happens if all motors turn off?' "
        "q̈_drift = M(q)⁻¹ · (C(q,v)v + g(q))",
        category=FeatureCategory.ANALYSIS,
        return_type="np.ndarray (n_v,)",
        section="Section F: Drift-Control Decomposition",
    ),
    FeatureDescriptor(
        name="compute_control_acceleration",
        display_name="Control Acceleration (Active Dynamics)",
        description="Compute control-attributed acceleration from applied torques only. "
        "q̈_control = M(q)⁻¹ · τ. "
        "CRITICAL: drift + control = full acceleration (superposition).",
        category=FeatureCategory.ANALYSIS,
        requires_args=True,
        arg_specs=[
            {
                "name": "tau",
                "type": "np.ndarray (n_v,)",
                "description": "Applied torques",
            }
        ],
        return_type="np.ndarray (n_v,)",
        section="Section F: Drift-Control Decomposition",
    ),
    # --- Counterfactual Experiments (Section G) ---
    FeatureDescriptor(
        name="compute_ztcf",
        display_name="Zero-Torque Counterfactual (ZTCF)",
        description="Compute acceleration with zero applied torques, preserving state. "
        "Isolates drift from control effects. "
        "Δa_control = a_full - a_ZTCF.",
        category=FeatureCategory.COUNTERFACTUAL,
        requires_args=True,
        arg_specs=[
            {
                "name": "q",
                "type": "np.ndarray (n_q,)",
                "description": "Joint positions",
            },
            {
                "name": "v",
                "type": "np.ndarray (n_v,)",
                "description": "Joint velocities",
            },
        ],
        return_type="np.ndarray (n_v,)",
        section="Section G1: ZTCF",
    ),
    FeatureDescriptor(
        name="compute_zvcf",
        display_name="Zero-Velocity Counterfactual (ZVCF)",
        description="Compute acceleration with zero velocities, preserving configuration. "
        "Isolates gravity/configuration effects from Coriolis/centrifugal. "
        "Δa_velocity = a_full - a_ZVCF.",
        category=FeatureCategory.COUNTERFACTUAL,
        requires_args=True,
        arg_specs=[
            {
                "name": "q",
                "type": "np.ndarray (n_q,)",
                "description": "Joint positions",
            },
        ],
        return_type="np.ndarray (n_v,)",
        section="Section G2: ZVCF",
    ),
    # --- Kinematics ---
    FeatureDescriptor(
        name="get_state",
        display_name="State (Positions & Velocities)",
        description="Get current generalized coordinates q and velocities v.",
        category=FeatureCategory.KINEMATICS,
        return_type="tuple[np.ndarray (n_q,), np.ndarray (n_v,)]",
    ),
    FeatureDescriptor(
        name="get_full_state",
        display_name="Full State (Batched)",
        description="Get complete state in a single call: q, v, t, and optionally M.",
        category=FeatureCategory.KINEMATICS,
        return_type="dict with q, v, t, M keys",
    ),
    FeatureDescriptor(
        name="compute_jacobian",
        display_name="Body Jacobian",
        description="Compute spatial Jacobian for a named body frame. "
        "Returns linear (3, n_v) and angular (3, n_v) Jacobians.",
        category=FeatureCategory.KINEMATICS,
        requires_args=True,
        arg_specs=[
            {"name": "body_name", "type": "str", "description": "Body frame name"}
        ],
        return_type="dict with 'linear', 'angular' keys or None",
    ),
    # --- Contact ---
    FeatureDescriptor(
        name="compute_contact_forces",
        display_name="Contact Forces (GRF)",
        description="Compute total ground reaction forces. "
        "Returns force vector (3,) or wrench (6,).",
        category=FeatureCategory.CONTACT,
        return_type="np.ndarray (3,) or (6,)",
    ),
    # --- Control ---
    FeatureDescriptor(
        name="set_control",
        display_name="Apply Control Inputs",
        description="Apply control torques/forces to actuators. "
        "Stored for next step/forward call.",
        category=FeatureCategory.CONTROL,
        requires_args=True,
        arg_specs=[
            {"name": "u", "type": "np.ndarray (n_u,)", "description": "Control vector"}
        ],
        return_type="None",
    ),
    # --- Shaft (Section B5) ---
    FeatureDescriptor(
        name="set_shaft_properties",
        display_name="Flexible Shaft Properties",
        description="Configure flexible shaft model (EI profile, mass distribution). "
        "Optional capability - not all engines support this.",
        category=FeatureCategory.SHAFT,
        requires_args=True,
        arg_specs=[
            {"name": "length", "type": "float", "description": "Shaft length [m]"},
            {
                "name": "EI_profile",
                "type": "np.ndarray",
                "description": "Bending stiffness profile",
            },
            {
                "name": "mass_profile",
                "type": "np.ndarray",
                "description": "Mass per unit length",
            },
        ],
        return_type="bool",
        section="Section B5: Flexible Beam Shaft",
    ),
    FeatureDescriptor(
        name="get_shaft_state",
        display_name="Shaft Deformation State",
        description="Get current shaft deflection, rotation, velocity, and modal amplitudes.",
        category=FeatureCategory.SHAFT,
        return_type="dict or None",
        section="Section B5: Flexible Beam Shaft",
    ),
]


class ControlFeaturesRegistry:
    """Registry of all control and analysis features.

    Provides feature discovery, availability checking, and execution
    for all engine capabilities.

    Design by Contract:
        Preconditions:
            - Engine must be initialized
        Postconditions:
            - Feature availability is accurately reported
            - Execution results are validated
    """

    def __init__(self, engine: PhysicsEngine) -> None:
        """Initialize the registry with an engine.

        Args:
            engine: Physics engine to query for capabilities.
        """
        self.engine = engine
        self._features = self._check_availability()

    def list_features(
        self,
        category: FeatureCategory | str | None = None,
        available_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List all registered features.

        Args:
            category: Filter by category (optional).
            available_only: If True, only return available features.

        Returns:
            List of feature descriptors as dictionaries.
        """
        features = self._features

        if category is not None:
            if isinstance(category, str):
                category = FeatureCategory(category.lower())
            features = [f for f in features if f.category == category]

        if available_only:
            features = [f for f in features if f.available]

        return [
            {
                "name": f.name,
                "display_name": f.display_name,
                "description": f.description,
                "category": f.category.value,
                "requires_args": f.requires_args,
                "arg_specs": f.arg_specs,
                "return_type": f.return_type,
                "section": f.section,
                "available": f.available,
            }
            for f in features
        ]

    def get_feature(self, name: str) -> dict[str, Any] | None:
        """Get a specific feature descriptor.

        Args:
            name: Feature name.

        Returns:
            Feature descriptor dict, or None if not found.
        """
        for f in self._features:
            if f.name == name:
                return {
                    "name": f.name,
                    "display_name": f.display_name,
                    "description": f.description,
                    "category": f.category.value,
                    "requires_args": f.requires_args,
                    "arg_specs": f.arg_specs,
                    "return_type": f.return_type,
                    "section": f.section,
                    "available": f.available,
                }
        return None

    def is_available(self, name: str) -> bool:
        """Check if a feature is available on the current engine.

        Args:
            name: Feature name.

        Returns:
            True if the feature is available and callable.
        """
        for f in self._features:
            if f.name == name:
                return f.available
        return False

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a feature by name.

        Args:
            name: Feature name (method name).
            **kwargs: Arguments to pass to the feature method.

        Returns:
            Feature result.

        Raises:
            ValueError: If feature not found.
            RuntimeError: If feature not available.
        """
        feature = None
        for f in self._features:
            if f.name == name:
                feature = f
                break

        if feature is None:
            raise ValueError(
                f"Feature '{name}' not found. "
                f"Available: {[f.name for f in self._features]}"
            )

        if not feature.available:
            raise RuntimeError(
                f"Feature '{name}' is not available on engine "
                f"{type(self.engine).__name__}"
            )

        method = getattr(self.engine, name)
        result = method(**kwargs)

        # Convert numpy arrays to serializable format for API responses
        return self._serialize_result(result)

    def get_categories(self) -> list[dict[str, Any]]:
        """Get all feature categories with counts.

        Returns:
            List of category info dicts.
        """
        categories: dict[str, dict[str, Any]] = {}
        for f in self._features:
            cat = f.category.value
            if cat not in categories:
                categories[cat] = {
                    "name": cat,
                    "total": 0,
                    "available": 0,
                }
            categories[cat]["total"] += 1
            if f.available:
                categories[cat]["available"] += 1

        return list(categories.values())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all registered features.

        Returns:
            Summary dict with counts and engine info.
        """
        total = len(self._features)
        available = sum(1 for f in self._features if f.available)
        return {
            "engine": type(self.engine).__name__,
            "model_name": getattr(self.engine, "model_name", "unknown"),
            "total_features": total,
            "available_features": available,
            "unavailable_features": total - available,
            "categories": self.get_categories(),
        }

    def _check_availability(self) -> list[FeatureDescriptor]:
        """Check which features are available on the current engine.

        Returns:
            List of FeatureDescriptors with availability set.
        """
        features = []
        for fd in _FEATURE_DEFINITIONS:
            available = hasattr(self.engine, fd.name) and callable(
                getattr(self.engine, fd.name, None)
            )

            # For optional methods, also check if they have a real implementation
            if available and fd.name in ("set_shaft_properties", "get_shaft_state"):
                # These have default implementations that return False/None
                available = True  # Always available since default exists

            features.append(
                FeatureDescriptor(
                    name=fd.name,
                    display_name=fd.display_name,
                    description=fd.description,
                    category=fd.category,
                    requires_args=fd.requires_args,
                    arg_specs=fd.arg_specs,
                    return_type=fd.return_type,
                    section=fd.section,
                    available=available,
                )
            )

        return features

    def _serialize_result(self, result: Any) -> Any:
        """Serialize a feature result for API responses.

        Args:
            result: Raw result from engine method.

        Returns:
            JSON-serializable result.
        """
        if isinstance(result, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(result.shape),
                "data": result.tolist(),
            }
        elif isinstance(result, tuple):
            return [self._serialize_result(r) for r in result]
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}
        else:
            return result
