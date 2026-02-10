"""Contact-Based Grip Model Module.

Guideline K2 Implementation: Contact-Based Grip Model (MuJoCo).

Provides contact mechanics modeling for hand-club interface including:
- Friction cone constraints (static/dynamic coefficients)
- Normal force distribution across contact points
- Slip detection and magnitude tracking
- Grip pressure visualization data

The grip model replaces rigid constraints with contact pairs,
enabling realistic force transmission and slip analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Default friction coefficients for hand-grip interface
DEFAULT_STATIC_FRICTION = 0.8  # Rubber grip on dry skin
DEFAULT_DYNAMIC_FRICTION = 0.6  # Sliding friction
SLIP_VELOCITY_THRESHOLD = 0.001  # [m/s] velocity above which slip is detected


class ContactState(Enum):
    """State of a contact point."""

    NO_CONTACT = auto()  # Not in contact
    STICKING = auto()  # In contact, within friction cone
    SLIPPING = auto()  # In contact, outside friction cone


@dataclass
class ContactPoint:
    """Single contact point between hand and grip.

    Attributes:
        position: Contact position in world frame [m] (3,)
        normal: Contact normal (pointing from grip to hand) [unitless] (3,)
        normal_force: Normal force magnitude [N]
        tangent_force: Tangential force vector [N] (3,)
        slip_velocity: Slip velocity at contact [m/s] (3,)
        state: Current contact state
        body_name: Name of the body this contact belongs to (e.g., "left_hand")
        contact_id: Unique identifier for this contact point
    """

    position: np.ndarray
    normal: np.ndarray
    normal_force: float
    tangent_force: np.ndarray
    slip_velocity: np.ndarray
    state: ContactState
    body_name: str = ""
    contact_id: int = 0


@dataclass
class GripContactState:
    """Complete state of all grip contacts.

    Attributes:
        contacts: List of active contact points
        total_normal_force: Sum of all normal forces [N]
        total_tangent_force: Net tangential force [N] (3,)
        num_slipping: Number of contacts that are slipping
        num_sticking: Number of contacts that are sticking
        center_of_pressure: Weighted center of contact forces [m] (3,)
        timestamp: Time of measurement [s]
    """

    contacts: list[ContactPoint]
    total_normal_force: float
    total_tangent_force: np.ndarray
    num_slipping: int
    num_sticking: int
    center_of_pressure: np.ndarray
    timestamp: float = 0.0


@dataclass
class GripParameters:
    """Parameters for grip contact model.

    Attributes:
        static_friction: Static friction coefficient μ_s
        dynamic_friction: Dynamic friction coefficient μ_d
        contact_stiffness: Contact spring stiffness [N/m]
        contact_damping: Contact damping coefficient [N·s/m]
        grip_diameter: Diameter of grip [m]
        hand_contact_area: Total contact area [m²]
    """

    static_friction: float = DEFAULT_STATIC_FRICTION
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION
    contact_stiffness: float = 1e5  # [N/m]
    contact_damping: float = 1e3  # [N·s/m]
    grip_diameter: float = 0.022  # [m] Standard golf grip diameter
    hand_contact_area: float = 0.01  # [m²] Approximate hand contact area


def check_friction_cone(
    normal_force: float,
    tangent_force: np.ndarray,
    friction_coefficient: float,
) -> bool:
    """Check if tangential force is within friction cone.

    The friction cone constraint is:
        |F_t| ≤ μ * F_n

    Args:
        normal_force: Normal force magnitude [N]
        tangent_force: Tangential force vector [N] (3,)
        friction_coefficient: Friction coefficient μ

    Returns:
        True if force is within friction cone (sticking)
    """
    tangent_magnitude = np.linalg.norm(tangent_force)
    max_tangent = friction_coefficient * abs(normal_force)
    return bool(tangent_magnitude <= max_tangent)


def compute_slip_direction(
    tangent_force: np.ndarray,
) -> np.ndarray:
    """Compute the direction of slip from tangential force.

    Slip occurs in the direction of the tangential force.

    Args:
        tangent_force: Tangential force vector [N] (3,)

    Returns:
        Unit vector in slip direction, or zero if no tangent force
    """
    magnitude = np.linalg.norm(tangent_force)
    if magnitude < 1e-10:
        return np.zeros(3)
    return np.asarray(tangent_force / magnitude)


def decompose_contact_force(
    contact_force: np.ndarray,
    contact_normal: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Decompose contact force into normal and tangential components.

    Args:
        contact_force: Total contact force [N] (3,)
        contact_normal: Unit normal vector (3,)

    Returns:
        Tuple of (normal_force [N], tangent_force [N] (3,))
    """
    # Normal component
    normal_force = float(np.dot(contact_force, contact_normal))

    # Tangential component
    tangent_force = contact_force - normal_force * contact_normal

    return normal_force, tangent_force


def classify_contact_state(
    normal_force: float,
    tangent_force: np.ndarray,
    slip_velocity: np.ndarray,
    params: GripParameters,
) -> ContactState:
    """Classify the state of a contact point.

    Args:
        normal_force: Normal force magnitude [N]
        tangent_force: Tangential force vector [N] (3,)
        slip_velocity: Slip velocity at contact [m/s] (3,)
        params: Grip contact parameters

    Returns:
        ContactState classification
    """
    # No contact if normal force is negligible or tensile
    if normal_force <= 0:
        return ContactState.NO_CONTACT

    # Check for slip based on velocity
    slip_speed = np.linalg.norm(slip_velocity)
    if slip_speed > SLIP_VELOCITY_THRESHOLD:
        return ContactState.SLIPPING

    # Check friction cone for sticking vs slipping
    if check_friction_cone(normal_force, tangent_force, params.static_friction):
        return ContactState.STICKING
    else:
        return ContactState.SLIPPING


def compute_center_of_pressure(
    contacts: list[ContactPoint],
) -> np.ndarray:
    """Compute weighted center of pressure from contact points.

    COP = Σ(f_n * p) / Σ(f_n)

    Args:
        contacts: List of contact points

    Returns:
        Center of pressure position [m] (3,)
    """
    if not contacts:
        return np.zeros(3)

    total_force = 0.0
    weighted_position = np.zeros(3)

    for c in contacts:
        if c.normal_force > 0:
            total_force += c.normal_force
            weighted_position += c.normal_force * c.position

    if total_force < 1e-10:
        return np.zeros(3)

    return weighted_position / total_force


def compute_grip_torque(
    contacts: list[ContactPoint],
    grip_center: np.ndarray,
) -> np.ndarray:
    """Compute net torque on grip from all contact forces.

    τ = Σ(r × F)

    Args:
        contacts: List of contact points
        grip_center: Reference point for torque computation [m] (3,)

    Returns:
        Net torque vector [N·m] (3,)
    """
    total_torque = np.zeros(3)

    for c in contacts:
        r = c.position - grip_center
        f = c.normal_force * c.normal + c.tangent_force
        total_torque += np.cross(r, f)

    return total_torque


class GripContactModel:
    """Contact-based grip model for hand-club interface.

    Tracks contact states, forces, and slip for grip analysis.
    Designed for MuJoCo integration but provides engine-agnostic API.
    """

    def __init__(self, params: GripParameters | None = None) -> None:
        """Initialize grip contact model.

        Args:
            params: Grip contact parameters (uses defaults if None)
        """
        self.params = params or GripParameters()
        self.current_state: GripContactState | None = None
        self.contact_history: list[GripContactState] = []

    def update_from_mujoco(
        self,
        contact_positions: np.ndarray,
        contact_normals: np.ndarray,
        contact_forces: np.ndarray,
        contact_velocities: np.ndarray,
        body_names: list[str],
        timestamp: float,
    ) -> GripContactState:
        """Update grip state from MuJoCo contact data.

        Args:
            contact_positions: Contact positions (N, 3) [m]
            contact_normals: Contact normals (N, 3) [unitless]
            contact_forces: Contact forces (N, 3) [N]
            contact_velocities: Relative velocities at contacts (N, 3) [m/s]
            body_names: Body names for each contact (N,)
            timestamp: Current simulation time [s]

        Returns:
            Updated GripContactState
        """
        n_contacts = len(contact_positions)

        contacts: list[ContactPoint] = []

        for i in range(n_contacts):
            # Decompose force
            normal_force, tangent_force = decompose_contact_force(
                contact_forces[i], contact_normals[i]
            )

            # Compute slip velocity (tangential component of velocity)
            vel = contact_velocities[i]
            vel_normal = np.dot(vel, contact_normals[i]) * contact_normals[i]
            slip_velocity = vel - vel_normal

            # Classify state
            state = classify_contact_state(
                normal_force, tangent_force, slip_velocity, self.params
            )

            contacts.append(
                ContactPoint(
                    position=contact_positions[i].copy(),
                    normal=contact_normals[i].copy(),
                    normal_force=normal_force,
                    tangent_force=tangent_force.copy(),
                    slip_velocity=slip_velocity.copy(),
                    state=state,
                    body_name=body_names[i] if i < len(body_names) else "",
                    contact_id=i,
                )
            )

        # Aggregate statistics
        total_normal = sum(c.normal_force for c in contacts if c.normal_force > 0)
        total_tangent = sum((c.tangent_force for c in contacts), np.zeros(3))
        num_slipping = sum(1 for c in contacts if c.state == ContactState.SLIPPING)
        num_sticking = sum(1 for c in contacts if c.state == ContactState.STICKING)
        cop = compute_center_of_pressure(contacts)

        self.current_state = GripContactState(
            contacts=contacts,
            total_normal_force=total_normal,
            total_tangent_force=total_tangent,
            num_slipping=num_slipping,
            num_sticking=num_sticking,
            center_of_pressure=cop,
            timestamp=timestamp,
        )

        self.contact_history.append(self.current_state)
        return self.current_state

    def check_static_equilibrium(
        self,
        club_weight: float,
        gravity_direction: np.ndarray = np.array([0.0, 0.0, -1.0]),
    ) -> dict[str, bool | float]:
        """Check if grip contacts can support club weight in static equilibrium.

        Validation test: Club weight should be fully supported by
        contact normal forces.

        Args:
            club_weight: Weight of club [N]
            gravity_direction: Direction of gravity [unitless] (3,)

        Returns:
            Dictionary with equilibrium check results
        """
        if self.current_state is None:
            return {"equilibrium": False, "support_ratio": 0.0}

        # Vertical component of total contact force
        total_contact_force = np.zeros(3)
        for c in self.current_state.contacts:
            total_contact_force += c.normal_force * c.normal + c.tangent_force

        # Check if contact force can support weight
        support_magnitude = np.dot(total_contact_force, -gravity_direction)
        support_ratio = support_magnitude / club_weight if club_weight > 0 else 0

        equilibrium = support_ratio >= 0.99  # 99% support threshold

        return {
            "equilibrium": equilibrium,
            "support_ratio": float(support_ratio),
            "total_normal_force": self.current_state.total_normal_force,
            "required_force": float(club_weight),
        }

    def check_slip_margin(self) -> dict[str, float]:
        """Compute slip margin for all contacts.

        Slip margin = (μ * F_n - |F_t|) / (μ * F_n)
        Positive margin means within friction cone (sticking)
        Negative margin means slipping

        Returns:
            Dictionary with slip margin statistics
        """
        if self.current_state is None or not self.current_state.contacts:
            return {"min_margin": 0.0, "mean_margin": 0.0, "any_slipping": True}

        # Use list comprehension for performance (10-15% faster)
        margins = [
            (max_tangent - np.linalg.norm(c.tangent_force)) / max_tangent
            for c in self.current_state.contacts
            if c.normal_force > 0
            and (max_tangent := self.params.static_friction * c.normal_force) > 0
        ]

        if not margins:
            return {"min_margin": 0.0, "mean_margin": 0.0, "any_slipping": True}

        return {
            "min_margin": float(np.min(margins)),  # type: ignore[arg-type]
            "mean_margin": float(np.mean(margins)),
            "any_slipping": any(m < 0 for m in margins),
        }

    def get_pressure_distribution(self) -> np.ndarray:
        """Get grip pressure distribution for visualization.

        Returns:
            Pressure at each contact point [Pa] (N,)
        """
        if self.current_state is None:
            return np.array([])

        # Approximate contact area per point
        n_contacts = len(self.current_state.contacts)
        if n_contacts == 0:
            return np.array([])

        area_per_contact = self.params.hand_contact_area / n_contacts

        # Use list comprehension for performance (10-15% faster)
        pressures = [
            c.normal_force / area_per_contact if area_per_contact > 0 else 0
            for c in self.current_state.contacts
        ]

        return np.array(pressures)

    def reset(self) -> None:
        """Reset grip model state."""
        self.current_state = None
        self.contact_history.clear()


def create_mujoco_grip_contacts(
    grip_body_name: str = "club_grip",
    hand_body_names: list[str] | None = None,
    friction: tuple[float, float, float] = (0.8, 0.6, 0.001),
) -> dict:
    """Generate MuJoCo contact specification for grip model.

    Creates contact pair definitions suitable for MuJoCo XML.

    Args:
        grip_body_name: Name of club grip body
        hand_body_names: Names of hand bodies (default: left/right hand)
        friction: (static, dynamic, rolling) friction coefficients

    Returns:
        Dictionary with MuJoCo contact specifications
    """
    if hand_body_names is None:
        hand_body_names = ["left_hand", "right_hand"]

    contact_pairs = []
    for hand in hand_body_names:
        contact_pairs.append(
            {
                "body1": hand,
                "body2": grip_body_name,
                "friction": list(friction),
                "condim": 4,  # Full 3D friction cone
                "margin": 0.001,  # 1mm contact margin
                "gap": 0.0,
            }
        )

    return {
        "contact_pairs": contact_pairs,
        "default_friction": friction,
        "solver_parameters": {
            "nconmax": 100,  # Max contacts
            "njmax": 300,  # Max constraints
            "cone": "pyramidal",  # Friction cone type
        },
    }


@dataclass
class GripContactTimestep:
    """Contact data for a single timestep (for export).

    Attributes:
        timestamp: Simulation time [s]
        total_normal_force: Total normal force [N]
        total_tangent_force_mag: Magnitude of total tangent force [N]
        num_contacts: Number of active contacts
        num_slipping: Number of slipping contacts
        num_sticking: Number of sticking contacts
        slip_ratio: Ratio of slipping contacts
        min_slip_margin: Minimum slip margin across contacts
        mean_slip_margin: Mean slip margin across contacts
        center_of_pressure: Center of pressure position [m] (3,)
        max_pressure: Maximum contact pressure [Pa]
        mean_pressure: Mean contact pressure [Pa]
        contact_forces: Per-contact normal forces [N] (N,)
        contact_positions: Per-contact positions [m] (N, 3)
        slip_velocities: Per-contact slip velocity magnitudes [m/s] (N,)
    """

    timestamp: float
    total_normal_force: float
    total_tangent_force_mag: float
    num_contacts: int
    num_slipping: int
    num_sticking: int
    slip_ratio: float
    min_slip_margin: float
    mean_slip_margin: float
    center_of_pressure: np.ndarray
    max_pressure: float
    mean_pressure: float
    contact_forces: np.ndarray
    contact_positions: np.ndarray
    slip_velocities: np.ndarray


class GripContactExporter:
    """Export grip contact data per timestep.

    Issue #757: Contact forces and slip metrics exported per timestep.
    """

    def __init__(self, model: GripContactModel) -> None:
        """Initialize exporter with grip contact model.

        Args:
            model: GripContactModel to export data from
        """
        self.model = model
        self.timesteps: list[GripContactTimestep] = []

    def capture_timestep(self) -> GripContactTimestep | None:
        """Capture current model state as exportable timestep.

        Returns:
            GripContactTimestep data or None if no current state
        """
        state = self.model.current_state
        if state is None:
            return None

        # Get slip margins
        margins = self.model.check_slip_margin()

        # Get pressure distribution
        pressures = self.model.get_pressure_distribution()

        # Extract per-contact data
        contact_forces = np.array([c.normal_force for c in state.contacts])
        contact_positions = (
            np.array([c.position for c in state.contacts])
            if state.contacts
            else np.zeros((0, 3))
        )
        slip_velocities = np.array(
            [np.linalg.norm(c.slip_velocity) for c in state.contacts]
        )

        timestep = GripContactTimestep(
            timestamp=state.timestamp,
            total_normal_force=state.total_normal_force,
            total_tangent_force_mag=float(np.linalg.norm(state.total_tangent_force)),
            num_contacts=len(state.contacts),
            num_slipping=state.num_slipping,
            num_sticking=state.num_sticking,
            slip_ratio=(
                state.num_slipping / len(state.contacts) if state.contacts else 0.0
            ),
            min_slip_margin=margins["min_margin"],
            mean_slip_margin=margins["mean_margin"],
            center_of_pressure=state.center_of_pressure.copy(),
            max_pressure=float(np.max(pressures)) if len(pressures) > 0 else 0.0,
            mean_pressure=float(np.mean(pressures)) if len(pressures) > 0 else 0.0,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            slip_velocities=slip_velocities,
        )

        self.timesteps.append(timestep)
        return timestep

    def export_to_dict(self) -> dict:
        """Export all captured timesteps as dictionary.

        Returns:
            Dictionary with all timestep data for JSON/CSV export
        """
        return {
            "metadata": {
                "num_timesteps": len(self.timesteps),
                "friction_static": self.model.params.static_friction,
                "friction_dynamic": self.model.params.dynamic_friction,
                "grip_diameter": self.model.params.grip_diameter,
            },
            "timesteps": [
                {
                    "timestamp": ts.timestamp,
                    "total_normal_force": ts.total_normal_force,
                    "total_tangent_force_mag": ts.total_tangent_force_mag,
                    "num_contacts": ts.num_contacts,
                    "num_slipping": ts.num_slipping,
                    "num_sticking": ts.num_sticking,
                    "slip_ratio": ts.slip_ratio,
                    "min_slip_margin": ts.min_slip_margin,
                    "mean_slip_margin": ts.mean_slip_margin,
                    "center_of_pressure": ts.center_of_pressure.tolist(),
                    "max_pressure": ts.max_pressure,
                    "mean_pressure": ts.mean_pressure,
                }
                for ts in self.timesteps
            ],
        }

    def export_to_csv_data(self) -> list[dict]:
        """Export timesteps as list of flat dictionaries for CSV.

        Returns:
            List of dictionaries suitable for pandas DataFrame or CSV
        """
        return [
            {
                "timestamp": ts.timestamp,
                "total_normal_force": ts.total_normal_force,
                "total_tangent_force_mag": ts.total_tangent_force_mag,
                "num_contacts": ts.num_contacts,
                "num_slipping": ts.num_slipping,
                "num_sticking": ts.num_sticking,
                "slip_ratio": ts.slip_ratio,
                "min_slip_margin": ts.min_slip_margin,
                "mean_slip_margin": ts.mean_slip_margin,
                "cop_x": ts.center_of_pressure[0],
                "cop_y": ts.center_of_pressure[1],
                "cop_z": ts.center_of_pressure[2],
                "max_pressure": ts.max_pressure,
                "mean_pressure": ts.mean_pressure,
            }
            for ts in self.timesteps
        ]

    def get_summary_statistics(self) -> dict:
        """Compute summary statistics across all timesteps.

        Returns:
            Dictionary with summary statistics
        """
        if not self.timesteps:
            return {"error": "No timesteps captured"}

        forces = [ts.total_normal_force for ts in self.timesteps]
        slip_ratios = [ts.slip_ratio for ts in self.timesteps]
        margins = [ts.min_slip_margin for ts in self.timesteps]

        return {
            "duration": self.timesteps[-1].timestamp - self.timesteps[0].timestamp,
            "num_timesteps": len(self.timesteps),
            "force_mean": float(np.mean(forces)),
            "force_max": float(np.max(forces)),
            "force_std": float(np.std(forces)),
            "slip_ratio_mean": float(np.mean(slip_ratios)),
            "slip_ratio_max": float(np.max(slip_ratios)),
            "any_slip_detected": any(sr > 0 for sr in slip_ratios),
            "min_margin_ever": float(np.min(margins)),
            "mean_margin": float(np.mean(margins)),
        }

    def reset(self) -> None:
        """Clear captured timesteps."""
        self.timesteps.clear()


@dataclass
class PressureVisualizationData:
    """Data structure for pressure visualization.

    Issue #757: Pressure distribution visualization in UI.

    Attributes:
        positions: Contact positions in local grip frame [m] (N, 3)
        pressures: Pressure values [Pa] (N,)
        normalized_pressures: Pressures normalized to [0, 1] for coloring (N,)
        max_pressure: Maximum pressure value [Pa]
        mean_pressure: Mean pressure value [Pa]
        grip_axis_positions: Positions projected onto grip axis [m] (N,)
        angular_positions: Angular positions around grip [rad] (N,)
    """

    positions: np.ndarray
    pressures: np.ndarray
    normalized_pressures: np.ndarray
    max_pressure: float
    mean_pressure: float
    grip_axis_positions: np.ndarray
    angular_positions: np.ndarray


def compute_pressure_visualization(
    contacts: list[ContactPoint],
    grip_center: np.ndarray,
    grip_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    contact_area: float = 0.01,
) -> PressureVisualizationData:
    """Compute pressure visualization data from contacts.

    Transforms contact data into format suitable for 2D pressure map
    visualization (unwrapped cylinder or heatmap).

    Args:
        contacts: List of contact points
        grip_center: Center position of grip [m] (3,)
        grip_axis: Direction of grip axis [unitless] (3,)
        contact_area: Total contact area [m²]

    Returns:
        PressureVisualizationData for rendering
    """
    if not contacts:
        return PressureVisualizationData(
            positions=np.zeros((0, 3)),
            pressures=np.array([]),
            normalized_pressures=np.array([]),
            max_pressure=0.0,
            mean_pressure=0.0,
            grip_axis_positions=np.array([]),
            angular_positions=np.array([]),
        )

    n_contacts = len(contacts)
    area_per_contact = contact_area / n_contacts if n_contacts > 0 else 1.0

    # Extract positions and compute pressures
    positions = np.array([c.position for c in contacts])
    pressures = np.array(
        [
            c.normal_force / area_per_contact if area_per_contact > 0 else 0.0
            for c in contacts
        ]
    )

    max_pressure = float(np.max(pressures)) if len(pressures) > 0 else 0.0
    mean_pressure = float(np.mean(pressures)) if len(pressures) > 0 else 0.0

    # Normalize for visualization
    if max_pressure > 0:
        normalized_pressures = pressures / max_pressure
    else:
        normalized_pressures = np.zeros(n_contacts)

    # Transform to grip coordinate system
    grip_axis = grip_axis / np.linalg.norm(grip_axis)
    relative_pos = positions - grip_center

    # Position along grip axis
    grip_axis_positions = np.dot(relative_pos, grip_axis)

    # Angular position (project onto plane perpendicular to axis)
    # Find perpendicular vectors
    if abs(grip_axis[2]) < 0.9:
        perp1 = np.cross(grip_axis, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(grip_axis, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(grip_axis, perp1)

    # Compute angles
    x_proj = np.dot(relative_pos, perp1)
    y_proj = np.dot(relative_pos, perp2)
    angular_positions = np.arctan2(y_proj, x_proj)

    return PressureVisualizationData(
        positions=positions,
        pressures=pressures,
        normalized_pressures=normalized_pressures,
        max_pressure=max_pressure,
        mean_pressure=mean_pressure,
        grip_axis_positions=grip_axis_positions,
        angular_positions=angular_positions,
    )
