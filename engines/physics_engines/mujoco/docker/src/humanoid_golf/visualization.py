"""Enhanced visualization module for MuJoCo humanoid golf simulation.

This module provides:
- Force and torque visualization overlays
- Trajectory tracers that persist through simulation
- Real-time data display
"""

import numpy as np
from collections import deque


class TrajectoryTracer:
    """Manages trajectory traces for bodies in the simulation."""

    def __init__(self, max_points=1000):
        """Initialize trajectory tracer.

        Args:
            max_points: Maximum number of points to store per body
        """
        self.traces = {}  # body_name -> deque of (x, y, z) positions
        self.max_points = max_points

    def add_point(self, body_name, position):
        """Add a point to a body's trajectory.

        Args:
            body_name: Name of the body
            position: 3D position (x, y, z)
        """
        if body_name not in self.traces:
            self.traces[body_name] = deque(maxlen=self.max_points)

        self.traces[body_name].append(position.copy())

    def get_trace(self, body_name):
        """Get trajectory trace for a body.

        Args:
            body_name: Name of the body

        Returns:
            List of positions or empty list if no trace exists
        """
        return list(self.traces.get(body_name, []))

    def clear(self, body_name=None):
        """Clear traces.

        Args:
            body_name: Specific body to clear, or None to clear all
        """
        if body_name:
            self.traces.pop(body_name, None)
        else:
            self.traces.clear()


class ForceVisualizer:
    """Visualizes forces and torques in the simulation."""

    def __init__(self, physics):
        """Initialize force visualizer.

        Args:
            physics: MuJoCo physics object
        """
        self.physics = physics
        self.model = physics.model
        self.data = physics.data

    def get_contact_forces(self):
        """Get all contact forces in the simulation.

        Returns:
            List of dicts with contact force information
        """
        import mujoco

        contacts = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)

            # Get body names
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            # Get body names safely
            try:
                body1_name = self.model.body(body1_id).name
            except Exception:
                body1_name = f"body_{body1_id}"

            try:
                body2_name = self.model.body(body2_id).name
            except Exception:
                body2_name = f"body_{body2_id}"

            contacts.append(
                {
                    "position": contact.pos.copy(),
                    "normal": contact.frame[:3].copy(),  # Contact normal
                    "normal_force": force[0],  # Normal component
                    "friction_force": np.linalg.norm(force[1:3]),  # Tangential
                    "total_force": np.linalg.norm(force[:3]),
                    "body1": body1_name,
                    "body2": body2_name,
                }
            )

        return contacts

    def get_joint_torques(self):
        """Get joint torques for all actuators.

        Returns:
            Dict mapping actuator names to torque values
        """
        torques = {}

        for i in range(self.model.nu):
            try:
                actuator_name = self.model.actuator(i).name
            except Exception:
                actuator_name = f"actuator_{i}"

            torques[actuator_name] = self.data.actuator_force[i]

        return torques

    def get_center_of_mass(self):
        """Get center of mass position and velocity.

        Returns:
            Dict with COM position and velocity
        """
        return {
            "position": self.data.subtree_com[0].copy(),
            "velocity": self.data.cvel[0][:3].copy(),  # Linear velocity only
        }


def add_visualization_overlays(viewer, physics, config, tracer):
    """Add visualization overlays to the MuJoCo viewer.

    Args:
        viewer: MuJoCo viewer object
        physics: Physics object
        config: Configuration dict
        tracer: TrajectoryTracer instance
    """
    import mujoco

    # Get visualization options
    show_forces = config.get("show_contact_forces", True)
    show_torques = config.get("show_joint_torques", True)
    show_tracers = config.get("show_tracers", True)
    tracer_bodies = config.get("tracer_bodies", [])

    visualizer = ForceVisualizer(physics)

    # Update trajectory traces
    if show_tracers:
        for body_name in tracer_bodies:
            try:
                body_id = physics.model.body(body_name).id
                body_pos = physics.data.xpos[body_id].copy()
                tracer.add_point(body_name, body_pos)
            except Exception:
                pass  # Body doesn't exist

    # Add contact force arrows
    if show_forces:
        contacts = visualizer.get_contact_forces()
        for contact in contacts:
            # Only show significant forces
            if contact["total_force"] > 0.1:
                # Add arrow for force direction and magnitude
                # Scale arrow size by force magnitude
                arrow_length = min(contact["total_force"] / 100.0, 0.5)

                # Direction of force (normal)
                direction = contact["normal"]

                # End point of arrow
                end_pos = contact["position"] + direction * arrow_length

                # Note: Actual rendering depends on viewer type
                # This is the data structure for force visualization
                pass

    # Add joint torque indicators
    if show_torques:
        torques = visualizer.get_joint_torques()
        # Torque visualization would show as colored joints or text overlays
        pass

    # Render trajectory traces
    if show_tracers:
        for body_name in tracer_bodies:
            trace = tracer.get_trace(body_name)
            if len(trace) > 1:
                # Render trace as connected line segments
                # Color code by body (e.g., head=red, hands=blue)
                pass


def create_force_arrow_geom(position, direction, magnitude, color):
    """Create geometry data for a force arrow.

    Args:
        position: Start position of arrow
        direction: Direction vector (will be normalized)
        magnitude: Force magnitude (affects arrow length)
        color: RGBA color array

    Returns:
        Dict with arrow geometry data
    """
    # Normalize direction
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Scale length by magnitude (cap at reasonable size)
    length = min(magnitude / 100.0, 0.5)

    # End position
    end_pos = position + direction * length

    # Arrow shaft radius
    radius = 0.005

    return {
        "type": "arrow",
        "start": position,
        "end": end_pos,
        "radius": radius,
        "color": color,
        "magnitude": magnitude,
    }


def create_trace_line_geom(points, color, radius=0.002):
    """Create geometry data for a trajectory trace line.

    Args:
        points: List of 3D positions
        color: RGBA color array
        radius: Line thickness

    Returns:
        List of line segment geometry data
    """
    segments = []

    for i in range(len(points) - 1):
        segments.append(
            {
                "type": "line",
                "start": points[i],
                "end": points[i + 1],
                "radius": radius,
                "color": color,
            }
        )

    return segments


# Color schemes for different visualization elements
FORCE_COLORS = {
    "contact_normal": [1.0, 0.0, 0.0, 0.8],  # Red for normal forces
    "contact_friction": [1.0, 0.5, 0.0, 0.8],  # Orange for friction
    "joint_torque_positive": [0.0, 1.0, 0.0, 0.8],  # Green for positive torque
    "joint_torque_negative": [0.0, 0.0, 1.0, 0.8],  # Blue for negative torque
}

TRACE_COLORS = {
    "pelvis": [1.0, 1.0, 0.0, 0.6],  # Yellow
    "torso": [0.0, 1.0, 1.0, 0.6],  # Cyan
    "head": [1.0, 0.0, 0.0, 0.6],  # Red
    "r_hand": [0.0, 1.0, 0.0, 0.6],  # Green
    "l_hand": [0.0, 0.0, 1.0, 0.6],  # Blue
    "r_foot": [1.0, 0.5, 0.0, 0.6],  # Orange
    "l_foot": [0.5, 0.0, 1.0, 0.6],  # Purple
}


def get_trace_color(body_name):
    """Get color for a body's trajectory trace.

    Args:
        body_name: Name of the body

    Returns:
        RGBA color array
    """
    return TRACE_COLORS.get(body_name, [0.5, 0.5, 0.5, 0.6])  # Default gray
