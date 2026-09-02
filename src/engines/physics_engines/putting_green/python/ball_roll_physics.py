# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Ball Rolling Physics for Putting Simulation.

This module implements the physics of a golf ball rolling on a putting surface,
including the transition from sliding to rolling, spin effects, and surface
interaction.

Physics Model:
    1. Initial impact creates sliding (spin ≠ v/r)
    2. Friction converts sliding to pure rolling
    3. Rolling friction decelerates ball
    4. Slopes add gravitational acceleration
    5. Ball eventually stops

Design by Contract:
    - All velocities in m/s
    - Spin in rad/s
    - Positions in meters
    - Time steps should be small (<= 0.01s) for accuracy

Roll-model provenance (ADR-0045 F1, issue #9343):
    UpstreamDrift preserves two putting roll models, and this module implements
    exactly one of them: ``ud-legacy-roll/1`` (:data:`UD_LEGACY_ROLL_MODEL`),
    the agronomic law ``mu ~= 0.196/stimp`` scaled by height-of-cut, condition,
    and grain factors (see :mod:`.turf_properties`).

    The preserved counterpart is ``usga-stimp-roll/1``
    (:data:`USGA_STIMP_ROLL_MODEL`), the Tools stack's stimpmeter-geometry law
    ``mu ~= 0.559/stimp`` (1.83 m/s USGA release speed) with Holmes/Penner
    speed-dependent hole capture; inside this repository that law is
    implemented by ``src.shared.python.putting_dynamics`` (restated from Tools
    ``swing_sim.putting.roll``) and reached by the ``/simulate-3d`` route.

    The divergence is physics, not a bug: both laws share the ``1/stimp`` form
    and assume different stimpmeter release speeds, which pins the roll-out
    ratio between them at the constant ~2.854 gated by Tools#4819 (P9).
    Because the two models disagree by that fixed factor, **results produced by
    different models must never be compared numerically without their model
    names attached.** Every result document this engine emits therefore carries
    a ``roll_model`` field (:data:`ROLL_MODEL_FIELD`), and readers of those
    documents call :func:`require_roll_model` to refuse an unnamed payload.

References:
    - Cross, R. (2006). Physics of Ball Rolling. American Journal of Physics.
    - Penner, A.R. (2002). The Physics of Putting. Canadian Journal of Physics.
    - ADR-0045 ``docs/adr/0045-putting-integration-one-experience-two-preserved-stacks.md``
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)


#: Provenance name of the roll model implemented in this module (ADR-0045 F1).
UD_LEGACY_ROLL_MODEL = "ud-legacy-roll/1"

#: Provenance name of the preserved counterpart model (Tools stimpmeter law).
USGA_STIMP_ROLL_MODEL = "usga-stimp-roll/1"

#: Key under which every putt result document carries its roll-model name.
ROLL_MODEL_FIELD = "roll_model"

#: The roll models ADR-0045 preserves. Readers accept no other name.
KNOWN_ROLL_MODELS = frozenset({UD_LEGACY_ROLL_MODEL, USGA_STIMP_ROLL_MODEL})


class RollModelProvenanceError(ValueError):
    """A result document is missing, blank, or misnaming its roll model.

    Raised by the fail-closed readers required by ADR-0045: an unnamed putt
    result cannot be compared with anything, because the two preserved models
    differ by the ~2.854 roll-out ratio pinned in Tools#4819.
    """


def require_roll_model(document: Mapping[str, Any], *, source: str) -> str:
    """Read the roll-model name from a result document, fail-closed.

    Design by Contract:
        Preconditions:
            - ``source`` is a non-empty description used in error messages.
        Postconditions:
            - The returned name is a member of :data:`KNOWN_ROLL_MODELS`.

    Args:
        document: Result document (mapping) to inspect.
        source: Human-readable origin of the document, quoted in errors.

    Returns:
        The roll-model name carried by the document.

    Raises:
        ValueError: If ``source`` is empty.
        RollModelProvenanceError: If the document is not a mapping, omits
            :data:`ROLL_MODEL_FIELD`, carries a blank name, or names a model
            this repository does not preserve.
    """
    if not source:
        raise ValueError("source must be a non-empty description")
    if not isinstance(document, Mapping):
        raise RollModelProvenanceError(
            f"{source}: expected a result document mapping carrying "
            f"{ROLL_MODEL_FIELD!r}, got {type(document).__name__}"
        )
    if ROLL_MODEL_FIELD not in document:
        raise RollModelProvenanceError(
            f"{source}: result document has no {ROLL_MODEL_FIELD!r} field; "
            "ADR-0045 requires every putt result to name its roll model "
            f"(expected one of {sorted(KNOWN_ROLL_MODELS)})"
        )
    name = document[ROLL_MODEL_FIELD]
    if not isinstance(name, str) or not name.strip():
        raise RollModelProvenanceError(
            f"{source}: {ROLL_MODEL_FIELD!r} must be a non-empty model name, "
            f"got {name!r}"
        )
    if name not in KNOWN_ROLL_MODELS:
        raise RollModelProvenanceError(
            f"{source}: unknown roll model {name!r}; ADR-0045 preserves "
            f"{sorted(KNOWN_ROLL_MODELS)}"
        )
    return name


def validate_roll_model_name(name: str, *, source: str) -> str:
    """Validate a bare roll-model name (not a whole document), fail-closed.

    Args:
        name: Candidate roll-model name.
        source: Human-readable origin, quoted in errors.

    Returns:
        The validated name.

    Raises:
        RollModelProvenanceError: If the name is blank or unknown.
    """
    return require_roll_model({ROLL_MODEL_FIELD: name}, source=source)


class RollMode(Enum):
    """Rolling mode of the ball."""

    SLIDING = "sliding"  # Ball sliding on surface (spin ≠ v/r)
    ROLLING = "rolling"  # Pure rolling (spin = v/r)
    STOPPED = "stopped"  # Ball at rest


@dataclass
class BallState:
    """Current state of the ball.

    Attributes:
        position: 2D position on green [m, m]
        velocity: 2D velocity [m/s, m/s]
        spin: 3D angular velocity [rad/s] (x=topspin, y=sidespin axis, z=sidespin)
    """

    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        """Ensure arrays are numpy."""
        # Normalize incoming vectors (including column/row vectors) to 1D so
        # scalar math (math.hypot) and downstream indexing stay consistent.
        self.position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(-1)
        self.spin = np.asarray(self.spin, dtype=np.float64).reshape(-1)

    @property
    def speed(self) -> float:
        """Ball speed magnitude."""
        # ⚡ Bolt: math.hypot is ~5x faster than np.linalg.norm for small 2D vectors
        return float(math.hypot(self.velocity[0], self.velocity[1]))

    @property
    def is_moving(self) -> bool:
        """Check if ball is moving (above threshold)."""
        return self.speed > 0.005  # 5mm/s threshold

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector of velocity."""
        if self.speed < 1e-10:
            return np.zeros(2)
        return self.velocity / self.speed

    def copy(self) -> BallState:
        """Create independent copy."""
        return BallState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            spin=self.spin.copy(),
        )


class BallRollPhysics:
    """Physics engine for ball rolling on putting surface.

    Implements realistic ball dynamics including:
    - Sliding-to-rolling transition
    - Spin decay
    - Surface friction
    - Slope effects
    - Grain effects

    Attributes:
        ball_mass: Ball mass [kg]
        ball_radius: Ball radius [m]
        turf: Turf properties
        green: Optional green surface for slopes
    """

    # Sliding friction is typically 1.5-2x rolling friction
    SLIDING_FRICTION_MULTIPLIER = 1.8

    # Velocity threshold for stopping
    STOP_VELOCITY_THRESHOLD = 0.005  # m/s

    # Spin threshold for pure rolling determination
    SPIN_VELOCITY_RATIO_TOLERANCE = 0.05

    def __init__(
        self,
        turf: TurfProperties | None = None,
        green: GreenSurface | None = None,
        ball_mass: float = GOLF_BALL_MASS_KG,
        ball_radius: float = GOLF_BALL_RADIUS_M,
        integrator: str = "euler",
    ) -> None:
        """Initialize ball physics.

        Args:
            turf: Turf properties (uses green's turf if green provided)
            green: Full green surface (optional)
            ball_mass: Ball mass [kg]
            ball_radius: Ball radius [m]
            integrator: Integration method ("euler", "rk4", "verlet")
        """
        if ball_mass is None:
            raise ValueError("ball_mass must be provided")
        self.green = green
        self.turf = turf or (green.turf if green else TurfProperties())
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.integrator = integrator

        # Ball moment of inertia (solid sphere)
        self._moment_of_inertia = (2.0 / 5.0) * ball_mass * ball_radius**2

        # Previous acceleration for Verlet integration
        self._prev_acceleration: np.ndarray | None = None

    @property
    def roll_model(self) -> str:
        """Name of the roll model these dynamics implement (ADR-0045 F1).

        Postcondition: the returned name is in :data:`KNOWN_ROLL_MODELS`.
        """
        return UD_LEGACY_ROLL_MODEL

    def determine_roll_mode(self, state: BallState) -> RollMode:
        """Determine current rolling mode from state.

        Pure rolling occurs when the contact point has zero velocity,
        which means: v = ω × r, or for our 2D case: v = -ω_y * r

        Args:
            state: Current ball state

        Returns:
            Current RollMode
        """
        if state is None:
            raise ValueError("state must be provided")
        speed = state.speed

        if speed < self.STOP_VELOCITY_THRESHOLD:
            return RollMode.STOPPED

        # At very low speeds, numerical errors in spin-velocity ratio
        # dominate; treat as rolling to avoid slip-friction feedback loop
        if speed < 0.05:
            return RollMode.ROLLING

        # For pure rolling: spin_y (about axis perpendicular to velocity) = v / r
        # The spin_y should be negative for forward roll (right-hand rule)
        expected_spin = -speed / self.ball_radius

        # Get spin component about axis perpendicular to velocity
        # This is the y-component of spin when velocity is in x-direction
        # For general direction, we need to project
        if speed > 1e-10:
            v_dir = state.velocity / speed
            # Spin axis for pure rolling is perpendicular to velocity and surface normal
            # For 2D ground: spin_axis = [-v_dir[1], v_dir[0], 0]
            # Component of spin about this axis:
            spin_axis = np.array([-v_dir[1], v_dir[0], 0])
            rolling_spin = np.dot(state.spin, spin_axis)

            # Check if close to pure rolling
            spin_error = abs(rolling_spin - expected_spin) / (
                abs(expected_spin) + 1e-10
            )

            if spin_error < self.SPIN_VELOCITY_RATIO_TOLERANCE:
                return RollMode.ROLLING

        return RollMode.SLIDING

    def compute_rolling_friction(self, state: BallState) -> np.ndarray:
        """Compute rolling friction force.

        Args:
            state: Current ball state

        Returns:
            Friction force vector [N]
        """
        if state is None:
            raise ValueError("state must be provided")
        if state.speed < 1e-10:
            return np.zeros(2)

        # Base friction from turf
        mu = self.turf.effective_friction

        # Apply grain effect
        grain_effect = self.turf.compute_grain_effect(state.direction)
        effective_mu = mu * (1.0 + grain_effect)

        # Friction force = μ * m * g (opposes motion)
        friction_mag = effective_mu * self.ball_mass * GRAVITY_M_S2
        friction_dir = -state.direction

        return friction_mag * friction_dir

    def compute_sliding_friction(self, state: BallState) -> np.ndarray:
        """Compute sliding friction force.

        During sliding, friction is higher and acts to both slow the ball
        and bring it to pure rolling.

        Args:
            state: Current ball state

        Returns:
            Friction force vector [N]
        """
        if state is None:
            raise ValueError("state must be provided")
        if state.speed < 1e-10:
            return np.zeros(2)

        # Sliding friction is higher than rolling
        mu = self.turf.effective_friction * self.SLIDING_FRICTION_MULTIPLIER

        # Friction opposes the slip velocity (contact point velocity)
        # For sliding: slip = v - ω × r
        # In 2D: slip_x = v_x + ω_y * r, slip_y = v_y - ω_x * r
        slip_velocity = np.array(
            [
                state.velocity[0] + state.spin[1] * self.ball_radius,
                state.velocity[1] - state.spin[0] * self.ball_radius,
            ]
        )

        # ⚡ Bolt: math.hypot avoids array allocation overhead
        slip_speed = math.hypot(slip_velocity[0], slip_velocity[1])
        if slip_speed < 1e-10:
            return self.compute_rolling_friction(state)

        slip_dir = slip_velocity / slip_speed

        # Friction force opposes slip
        friction_mag = mu * self.ball_mass * GRAVITY_M_S2
        return -friction_mag * slip_dir

    def compute_slope_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Compute acceleration from slope at position.

        Args:
            position: Ball position [m, m]

        Returns:
            Acceleration vector [m/s²]
        """
        if position is None:
            raise ValueError("position must be provided")
        if self.green is None:
            return np.zeros(2)

        return self.green.get_gravitational_acceleration(position)

    def compute_spin_decay(
        self, state: BallState, dt: float, mode: RollMode
    ) -> np.ndarray:
        """Compute spin decay over time step.

        During sliding, spin changes rapidly to approach pure rolling.
        During rolling, spin decays slowly with velocity.

        Args:
            state: Current ball state
            dt: Time step [s]
            mode: Current roll mode

        Returns:
            New spin vector [rad/s]
        """
        if state is None:
            raise ValueError("state must be provided")
        speed = state.speed

        if mode == RollMode.STOPPED:
            return np.zeros(3)

        if mode == RollMode.ROLLING:
            # Spin is locked to velocity in pure rolling
            # ω = -v / r (negative for forward roll)
            if speed > 1e-10:
                v_dir = state.velocity / speed
                spin_axis = np.array([-v_dir[1], v_dir[0], 0])
                rolling_spin_mag = speed / self.ball_radius
                return -spin_axis * rolling_spin_mag
            return np.zeros(3)

        # Sliding mode: spin decays toward pure rolling condition
        # The friction torque changes spin
        v_dir = state.velocity / (speed + 1e-10)

        # Target spin for pure rolling
        spin_axis = np.array([-v_dir[1], v_dir[0], 0])
        target_spin = -spin_axis * (speed / self.ball_radius)

        # Exponential approach to target (with friction-dependent rate)
        decay_rate = (
            self.turf.effective_friction * GRAVITY_M_S2 / self.ball_radius * 5.0
        )
        alpha = 1.0 - np.exp(-decay_rate * dt)

        new_spin = state.spin + alpha * (target_spin - state.spin)

        # Also decay sidespin (z-component)
        sidespin_decay = 0.9 ** (dt / 0.1)  # Decay 10% per 0.1s
        new_spin[2] *= sidespin_decay

        return new_spin

    def compute_total_acceleration(self, state: BallState) -> np.ndarray:
        """Compute total acceleration on ball.

        Args:
            state: Current ball state

        Returns:
            Acceleration vector [m/s²]
        """
        if state is None:
            raise ValueError("state must be provided")
        mode = self.determine_roll_mode(state)

        if mode == RollMode.STOPPED:
            # Check if on slope (could start moving)
            slope_accel = self.compute_slope_acceleration(state.position)
            return slope_accel

        # Friction force
        if mode == RollMode.ROLLING:
            friction = self.compute_rolling_friction(state)
        else:
            friction = self.compute_sliding_friction(state)

        # Slope acceleration
        slope_accel = self.compute_slope_acceleration(state.position)

        # Total acceleration
        friction_accel = friction / self.ball_mass

        return friction_accel + slope_accel

    def compute_kinetic_energy(self, state: BallState) -> float:
        """Compute total kinetic energy (translational + rotational).

        Args:
            state: Ball state

        Returns:
            Total kinetic energy [J]
        """
        # Translational: 0.5 * m * v²
        if state is None:
            raise ValueError("state must be provided")
        translational = 0.5 * self.ball_mass * state.speed**2

        # Rotational: 0.5 * I * ω²
        # ⚡ Bolt: math.hypot is faster than np.linalg.norm for 3D vectors
        # Normalize array shape before unpacking (fixes #3450)
        spin_vec = np.asarray(state.spin, dtype=float).reshape(-1)
        spin_mag = 0.0 if spin_vec.size == 0 else math.hypot(*spin_vec)
        rotational = 0.5 * self._moment_of_inertia * spin_mag**2

        return float(translational + rotational)

    def step(self, state: BallState, dt: float) -> BallState:
        """Advance ball state by one time step.

        Args:
            state: Current ball state
            dt: Time step [s]

        Returns:
            New ball state
        """
        if state is None:
            raise ValueError("state must be provided")
        if self.integrator == "rk4":
            return self._step_rk4(state, dt)
        if self.integrator == "verlet":
            return self._step_verlet(state, dt)
        return self._step_euler(state, dt)

    def _step_euler(self, state: BallState, dt: float) -> BallState:
        """Euler integration step."""
        if state is None:
            raise ValueError("state must be provided")
        mode = self.determine_roll_mode(state)

        if mode == RollMode.STOPPED:
            # Check if slope would cause movement
            accel = self.compute_slope_acceleration(state.position)
            # ⚡ Bolt: Use math.hypot for 2D magnitude optimization
            if math.hypot(accel[0], accel[1]) < 0.01:  # Threshold for starting
                return state.copy()

        # Compute acceleration
        accel = self.compute_total_acceleration(state)

        # Update velocity
        new_velocity = state.velocity + accel * dt

        # Apply grain curve effect
        new_velocity = self.turf.apply_grain_to_velocity(new_velocity)

        # Apply sidespin curve effect (Magnus-like on ground)
        if abs(state.spin[2]) > 1.0:  # Significant sidespin
            # Sidespin causes lateral acceleration
            spin_curve_accel = state.spin[2] * 0.001  # Coefficient
            perp_dir = np.array([-state.direction[1], state.direction[0]])
            new_velocity += perp_dir * spin_curve_accel * dt

        # Check for stopping
        # ⚡ Bolt: math.hypot for 2D magnitude
        new_speed = math.hypot(new_velocity[0], new_velocity[1])
        if new_speed < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update position
        new_position = state.position + new_velocity * dt

        # Update spin
        new_spin = self.compute_spin_decay(state, dt, mode)

        return BallState(
            position=new_position,
            velocity=new_velocity,
            spin=new_spin,
        )

    def _step_rk4(self, state: BallState, dt: float) -> BallState:
        """4th-order Runge-Kutta integration."""

        if state is None:
            raise ValueError("state must be provided")

        def derivatives(
            pos: np.ndarray, vel: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute velocity and acceleration for the given state."""
            if pos is None:
                raise ValueError("pos must be provided")
            temp_state = BallState(pos, vel, state.spin)
            accel = self.compute_total_acceleration(temp_state)
            return vel, accel

        pos, vel = state.position, state.velocity

        # RK4 stages
        k1_v, k1_a = derivatives(pos, vel)
        k2_v, k2_a = derivatives(pos + 0.5 * dt * k1_v, vel + 0.5 * dt * k1_a)
        k3_v, k3_a = derivatives(pos + 0.5 * dt * k2_v, vel + 0.5 * dt * k2_a)
        k4_v, k4_a = derivatives(pos + dt * k3_v, vel + dt * k3_a)

        # Weighted average
        new_position = pos + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        new_velocity = vel + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)

        # Check stopping
        if math.hypot(new_velocity[0], new_velocity[1]) < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update spin
        mode = self.determine_roll_mode(state)
        new_spin = self.compute_spin_decay(state, dt, mode)

        return BallState(position=new_position, velocity=new_velocity, spin=new_spin)

    def _step_verlet(self, state: BallState, dt: float) -> BallState:
        """Velocity Verlet integration (better energy conservation)."""
        # Current acceleration
        if state is None:
            raise ValueError("state must be provided")
        accel = self.compute_total_acceleration(state)

        # Update position
        if self._prev_acceleration is None:
            self._prev_acceleration = accel

        new_position = state.position + state.velocity * dt + 0.5 * accel * dt**2

        # Compute new acceleration at new position
        temp_state = BallState(new_position, state.velocity, state.spin)
        new_accel = self.compute_total_acceleration(temp_state)

        # Update velocity
        new_velocity = state.velocity + 0.5 * (accel + new_accel) * dt

        # Check stopping
        # ⚡ Bolt: math.hypot for 2D magnitude
        if math.hypot(new_velocity[0], new_velocity[1]) < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update spin
        mode = self.determine_roll_mode(state)
        new_spin = self.compute_spin_decay(state, dt, mode)

        self._prev_acceleration = new_accel

        return BallState(position=new_position, velocity=new_velocity, spin=new_spin)

    def simulate_putt(
        self,
        initial_state: BallState,
        max_time: float = 30.0,
        dt: float = 0.001,
    ) -> dict[str, Any]:
        """Simulate complete putt trajectory.

        Args:
            initial_state: Initial ball state
            max_time: Maximum simulation time [s]
            dt: Time step [s]

        Returns:
            Dictionary with trajectory data
        """
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        positions = [initial_state.position.copy()]
        velocities = [initial_state.velocity.copy()]
        spins = [initial_state.spin.copy()]
        times = [0.0]
        modes = [self.determine_roll_mode(initial_state)]

        state = initial_state.copy()
        t = 0.0
        holed = False

        while t < max_time and state.is_moving:
            state = self.step(state, dt)
            t += dt

            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            spins.append(state.spin.copy())
            times.append(t)
            modes.append(self.determine_roll_mode(state))

            # Check for hole
            if self.green is not None:
                if self.green.is_in_hole(state.position, state.velocity):
                    holed = True
                    break

                # Check if off green
                if not self.green.is_on_green(state.position):
                    break

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "spins": np.array(spins),
            "times": np.array(times),
            "modes": modes,
            "holed": holed,
            "final_position": state.position.copy(),
            "final_velocity": state.velocity.copy(),
            ROLL_MODEL_FIELD: self.roll_model,
        }
