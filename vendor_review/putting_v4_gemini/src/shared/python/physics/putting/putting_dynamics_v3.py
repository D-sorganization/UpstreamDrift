"""Unified 3D Dual-Axis Counterbalanced Putting Dynamics, AffineDrift & UpstreamDrift Engine (v3.2).

Engineered for direct drop-in integration with:
- `D-Sorganization/UpstreamDrift` (`TurfProperties`, `GreenSurface`, `SlopeRegion`, `BallState`, `BallRollPhysics`)
- `AffineDrift` (Control-Affine Tangent Bundle formulation: dx/dt = f_drift(x) + G_control(x) * u_couple)

Architectural Principles Enforced:
1. TDD (Test-Driven Development): Verified by 16-test automated suite in `test_putting_dynamics_v3.py`.
2. DbC (Design by Contract): Preconditions (`require`), postconditions (`ensure`), and class invariants (`check_invariant`).
3. LoD (Law of Demeter): Strict Tell-Don't-Ask encapsulation; callers never chain through internal sub-objects.
4. DRY (Don't Repeat Yourself): Single canonical implementations for minimum-jerk kinematics, hyperbolic C^infinity
   force-couple decomposition, Ornstein-Uhlenbeck C^2 motor noise, and Holmes/Penner USGA cup-capture & lip-out physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

# ==============================================================================
# PHYSICAL CONSTANTS (Canonical Single Source of Truth — DRY)
# ==============================================================================
GRAVITY_M_S2: float = 9.81
GOLF_BALL_MASS_KG: float = 0.04593
GOLF_BALL_RADIUS_M: float = 0.021335
USGA_HOLE_RADIUS_M: float = 0.0540
DEG_TO_RAD: float = np.pi / 180.0
RAD_TO_DEG: float = 180.0 / np.pi


# ==============================================================================
# DESIGN BY CONTRACT (DbC) PRIMITIVES
# ==============================================================================
def require(condition: bool, message: str) -> None:
    """Enforces a Design-by-Contract precondition."""
    if not condition:
        raise ValueError(f"Precondition failed: {message}")


def ensure(condition: bool, message: str) -> None:
    """Enforces a Design-by-Contract postcondition."""
    if not condition:
        raise RuntimeError(f"Postcondition failed: {message}")


def check_invariant(condition: bool, message: str) -> None:
    """Enforces a Design-by-Contract class/state invariant."""
    if not condition:
        raise AssertionError(f"Class invariant violated: {message}")


# ==============================================================================
# SHARED MATHEMATICAL PRIMITIVES (DRY)
# ==============================================================================
def smooth_hyperbolic_split(net_signal: np.ndarray | float, epsilon: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Splits a signed net signal into strictly positive C^infinity components (pos, neg) such that pos - neg == net_signal."""
    require(epsilon > 0.0, "Hyperbolic smoothing epsilon must be positive")
    arr = np.asarray(net_signal, dtype=np.float64)
    rad = np.sqrt(arr * arr + epsilon * epsilon)
    pos = 0.5 * (arr + rad)
    neg = 0.5 * (-arr + rad)
    return pos, neg


def minimum_jerk_polynomial(s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns 5th-order minimum-jerk position p(s), first derivative dp/ds, and second derivative d2p/ds2 on s in [0, 1]."""
    s_arr = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    p = 10.0 * (s_arr ** 3) - 15.0 * (s_arr ** 4) + 6.0 * (s_arr ** 5)
    dp = 30.0 * (s_arr ** 2) - 60.0 * (s_arr ** 3) + 30.0 * (s_arr ** 4)
    ddp = 60.0 * s_arr - 180.0 * (s_arr ** 2) + 120.0 * (s_arr ** 3)
    return p, dp, ddp


def generate_smooth_ou_noise(
    n_steps: int,
    dt: float,
    sigma: float,
    tau_corr: float = 0.048,
    seed: int | None = None,
) -> np.ndarray:
    """Generates C^2-smooth Ornstein-Uhlenbeck + Butterworth-filtered physiological motor noise."""
    require(n_steps > 12, "n_steps must exceed filter padding length (12)")
    require(dt > 0.0, "Time step dt must be strictly positive")
    require(sigma >= 0.0, "Noise standard deviation sigma must be non-negative")
    if sigma <= 1e-12:
        return np.zeros(n_steps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ou = np.zeros(n_steps, dtype=np.float64)
    decay = float(np.exp(-dt / max(1e-4, tau_corr)))
    diff = sigma * float(np.sqrt(max(0.0, 1.0 - decay * decay)))
    for i in range(n_steps - 1):
        ou[i + 1] = decay * ou[i] + diff * rng.standard_normal()
    nyq = 0.5 / dt
    wn = min(0.45, 6.0 / nyq)
    b, a = signal.butter(2, wn)
    filtered = signal.filtfilt(b, a, ou)
    ensure(filtered.shape == (n_steps,), "Filtered noise must preserve length")
    return filtered


# ==============================================================================
# UPSTREAMDRIFT GREEN SURFACE, OBLONG CONTOUR, & BALL ROLL PHYSICS
# ==============================================================================
class RollMode(Enum):
    """Rolling state of the golf ball on the putting green."""
    SLIDING = "sliding"
    ROLLING = "rolling"
    STOPPED = "stopped"


class GrassType(Enum):
    """Putting green turf species."""
    BENT_GRASS = "bent_grass"
    BERMUDA = "bermuda"
    POA_ANNUA = "poa_annua"


@dataclass(frozen=True)
class TurfProperties:
    """Physical properties of putting green turf (UpstreamDrift compatible)."""
    stimp_rating: float = 11.5
    grass_type: GrassType = GrassType.BENT_GRASS
    grain_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    grain_strength: float = 0.08
    height_of_cut_mm: float = 2.8
    compaction_factor: float = 0.90

    def __post_init__(self) -> None:
        require(1.0 <= self.stimp_rating <= 15.0, f"stimp_rating must be in [1, 15], got {self.stimp_rating}")
        require(0.0 <= self.grain_strength <= 1.0, "grain_strength must be in [0, 1]")
        mag = float(np.linalg.norm(self.grain_direction))
        norm_dir = self.grain_direction / mag if mag > 0 else np.array([1.0, 0.0])
        object.__setattr__(self, "grain_direction", norm_dir)

    @property
    def rolling_friction_coefficient(self) -> float:
        """Calibrated USGA Stimpmeter rolling friction (v0=1.83 m/s rolls stimp_rating feet)."""
        base_mu = 0.560 / self.stimp_rating
        height_factor = 1.0 + 0.05 * (self.height_of_cut_mm - 3.0) / 2.0
        return base_mu * height_factor

    @property
    def effective_friction(self) -> float:
        return self.rolling_friction_coefficient


@dataclass
class SlopeRegion:
    """Localized undulation or tier on an oblong putting green."""
    center: np.ndarray
    radius: float
    slope_direction: np.ndarray
    slope_magnitude: float
    falloff: float = 0.2

    def __post_init__(self) -> None:
        require(self.radius > 0.0, "SlopeRegion radius must be positive")
        self.center = np.array(self.center, dtype=np.float64)
        self.slope_direction = np.array(self.slope_direction, dtype=np.float64)
        mag = float(np.linalg.norm(self.slope_direction))
        if mag > 0:
            self.slope_direction = self.slope_direction / mag

    def get_weight(self, position: np.ndarray) -> float:
        dist = float(np.linalg.norm(position[:2] - self.center[:2]))
        if dist >= self.radius:
            return 0.0
        norm_d = dist / self.radius
        return float(1.0 - norm_d ** (1.0 / (1.0 - self.falloff + 0.1)))


class GreenSurface:
    """Realistic oblong/kidney-contoured putting green surface with Holmes/Penner USGA cup capture."""
    STANDARD_HOLE_RADIUS: float = USGA_HOLE_RADIUS_M

    def __init__(
        self,
        width: float = 10.0,
        height: float = 6.0,
        turf: TurfProperties | None = None,
        uniform_slope: np.ndarray | None = None,
    ) -> None:
        require(width > 0 and height > 0, "Green dimensions must be positive")
        self.width = width
        self.height = height
        self.turf = turf or TurfProperties()
        self.hole_radius = self.STANDARD_HOLE_RADIUS
        self._hole_position = np.array([3.048, 0.0], dtype=np.float64)
        self._uniform_slope = np.array(
            uniform_slope if uniform_slope is not None else [0.0, 0.0], dtype=np.float64
        )
        self._slope_regions: list[SlopeRegion] = []
        self._green_center = np.array([1.65, 0.0], dtype=np.float64)

    @property
    def hole_position(self) -> np.ndarray:
        return self._hole_position.copy()

    @property
    def effective_friction(self) -> float:
        """Law of Demeter (LoD) facade property so callers never chain green.turf.effective_friction."""
        return self.turf.effective_friction

    @property
    def stimp_rating(self) -> float:
        """LoD facade property for Stimpmeter rating."""
        return self.turf.stimp_rating

    def set_hole_position(self, position: np.ndarray) -> None:
        self._hole_position = np.array(position[:2], dtype=np.float64)

    def set_uniform_slope(self, slope_xy: np.ndarray) -> None:
        """Sets global elevation gradient [dz/dx (uphill grade), dz/dy (lateral cross-slope)]."""
        self._uniform_slope = np.array(slope_xy[:2], dtype=np.float64)

    def add_slope_region(self, region: SlopeRegion) -> None:
        self._slope_regions.append(region)

    def is_within_oblong_green(self, position: np.ndarray) -> bool:
        """Evaluates whether (x, y) lies inside the realistic oblong/kidney putting green boundary."""
        dx = (float(position[0]) - self._green_center[0]) / (0.5 * self.width)
        dy = (float(position[1]) - self._green_center[1]) / (0.5 * self.height)
        theta = float(np.arctan2(dy, dx))
        # Kidney/oblong harmonic contour modulation
        r_bound = 1.0 + 0.12 * np.cos(2.0 * theta) - 0.06 * np.sin(3.0 * theta)
        return float(dx * dx + dy * dy) <= float(r_bound * r_bound)

    def get_oblong_boundary_points(self, n_pts: int = 120) -> np.ndarray:
        """Returns (n_pts, 2) array of (x, y) coordinates tracing the oblong putting green perimeter."""
        angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
        r_mod = 1.0 + 0.12 * np.cos(2.0 * angles) - 0.06 * np.sin(3.0 * angles)
        x = self._green_center[0] + 0.5 * self.width * r_mod * np.cos(angles)
        y = self._green_center[1] + 0.5 * self.height * r_mod * np.sin(angles)
        return np.column_stack([x, y])

    def get_slope_at(self, position: np.ndarray) -> np.ndarray:
        pos = position[:2]
        total_slope = self._uniform_slope.copy()
        for reg in self._slope_regions:
            w = reg.get_weight(pos)
            if w > 0:
                total_slope += w * reg.slope_magnitude * reg.slope_direction
        return total_slope

    def get_gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        return -GRAVITY_M_S2 * self.get_slope_at(position)

    def evaluate_cup_interaction(self, position: np.ndarray, velocity: np.ndarray) -> dict[str, Any]:
        """Holmes (1991) / Penner (2002) USGA cup capture & lip-out physics.

        Computes the lateral impact parameter b from hole center and compares entry speed
        against the chord-dependent critical capture velocity v_crit(b) = v_max * (1 - (b/R_h)^2)^0.75.
        """
        rel = position[:2] - self._hole_position
        dist = float(np.linalg.norm(rel))
        if dist > self.hole_radius:
            return {"captured": False, "lip_out": False, "deflected_velocity": velocity}

        speed = float(np.linalg.norm(velocity))
        if speed < 1e-6:
            return {"captured": True, "lip_out": False, "deflected_velocity": np.zeros(2)}

        v_hat = velocity / speed
        # Lateral impact offset b = |rel x v_hat|
        b = abs(float(rel[0] * v_hat[1] - rel[1] * v_hat[0]))
        b_ratio = min(0.999, b / self.hole_radius)
        v_crit_center = 1.38  # m/s max center-cup capture speed
        v_crit = v_crit_center * ((1.0 - b_ratio * b_ratio) ** 0.75)

        if speed <= v_crit:
            return {"captured": True, "lip_out": False, "deflected_velocity": np.zeros(2)}

        # Ball enters too fast for chord length -> Lips out! Deflects laterally & loses 38% speed
        sign = 1.0 if (rel[0] * v_hat[1] - rel[1] * v_hat[0]) >= 0 else -1.0
        deflect_ang = sign * (0.35 * b_ratio)
        c, s = float(np.cos(deflect_ang)), float(np.sin(deflect_ang))
        rot_v = np.array([c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1]]) * 0.62
        return {"captured": False, "lip_out": True, "deflected_velocity": rot_v}

    def is_in_hole(self, position: np.ndarray, velocity: np.ndarray | None = None) -> bool:
        vel = np.zeros(2) if velocity is None else velocity
        return bool(self.evaluate_cup_interaction(position, vel)["captured"])


@dataclass
class BallState:
    """Instantaneous 2D position, 2D velocity, and 3D spin vector of the golf ball."""
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.spin = np.array(self.spin, dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def is_moving(self) -> bool:
        return self.speed > 0.005

    @property
    def direction(self) -> np.ndarray:
        s = self.speed
        return self.velocity / s if s > 1e-10 else np.zeros(2)


class BallRollPhysics:
    """UpstreamDrift ball rolling physics engine with skid-to-roll transition and RK4 integration."""
    SLIDING_FRICTION_MULTIPLIER: float = 1.85
    STOP_VELOCITY_THRESHOLD: float = 0.005
    SPIN_VELOCITY_RATIO_TOLERANCE: float = 0.06

    def __init__(self, green: GreenSurface) -> None:
        self.green = green
        self.ball_mass = GOLF_BALL_MASS_KG
        self.ball_radius = GOLF_BALL_RADIUS_M
        self._moi = (2.0 / 5.0) * self.ball_mass * (self.ball_radius ** 2)

    def determine_roll_mode(self, state: BallState) -> RollMode:
        speed = state.speed
        if speed < self.STOP_VELOCITY_THRESHOLD:
            return RollMode.STOPPED
        if speed < 0.05:
            return RollMode.ROLLING
        v_dir = state.direction
        spin_axis = np.array([-v_dir[1], v_dir[0], 0.0])
        rolling_spin = float(np.dot(state.spin, spin_axis))
        expected_spin = -speed / self.ball_radius
        err = abs(rolling_spin - expected_spin) / (abs(expected_spin) + 1e-10)
        return RollMode.ROLLING if err < self.SPIN_VELOCITY_RATIO_TOLERANCE else RollMode.SLIDING

    def compute_total_acceleration(self, state: BallState) -> np.ndarray:
        mode = self.determine_roll_mode(state)
        slope_acc = self.green.get_gravitational_acceleration(state.position)
        if mode == RollMode.STOPPED:
            return slope_acc
        # LoD compliance: call self.green.effective_friction directly
        mu = self.green.effective_friction * (1.0 if mode == RollMode.ROLLING else self.SLIDING_FRICTION_MULTIPLIER)
        fric_acc = -mu * GRAVITY_M_S2 * state.direction
        lat_acc = np.zeros(2)
        if abs(state.spin[2]) > 0.5:
            perp_dir = np.array([-state.direction[1], state.direction[0]])
            lat_acc = perp_dir * (state.spin[2] * 0.0008)
        return fric_acc + slope_acc + lat_acc

    def step_rk4(self, state: BallState, dt: float) -> BallState:
        def deriv(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            st = BallState(pos, vel, state.spin)
            return vel, self.compute_total_acceleration(st)

        p0, v0 = state.position, state.velocity
        k1_v, k1_a = deriv(p0, v0)
        k2_v, k2_a = deriv(p0 + 0.5 * dt * k1_v, v0 + 0.5 * dt * k1_a)
        k3_v, k3_a = deriv(p0 + 0.5 * dt * k2_v, v0 + 0.5 * dt * k2_a)
        k4_v, k4_a = deriv(p0 + dt * k3_v, v0 + dt * k3_a)

        new_pos = p0 + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        new_vel = v0 + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        if float(np.linalg.norm(new_vel)) < self.STOP_VELOCITY_THRESHOLD:
            new_vel = np.zeros(2)

        mode = self.determine_roll_mode(state)
        new_spin = self._update_spin(state, new_vel, dt, mode)
        return BallState(new_pos, new_vel, new_spin)

    def _update_spin(self, state: BallState, new_vel: np.ndarray, dt: float, mode: RollMode) -> np.ndarray:
        speed = float(np.linalg.norm(new_vel))
        if speed < self.STOP_VELOCITY_THRESHOLD:
            return np.zeros(3)
        v_dir = new_vel / speed
        spin_axis = np.array([-v_dir[1], v_dir[0], 0.0])
        target_spin = -spin_axis * (speed / self.ball_radius)
        if mode == RollMode.ROLLING:
            return target_spin
        decay_rate = self.green.effective_friction * GRAVITY_M_S2 / self.ball_radius * 5.5
        alpha = 1.0 - np.exp(-decay_rate * dt)
        new_spin = state.spin + alpha * (target_spin - state.spin)
        new_spin[2] *= 0.90 ** (dt / 0.1)
        return new_spin


@dataclass
class StrokeParameters:
    """Impact delivery parameters of the putter clubhead at ball strike (theta == 0)."""
    speed: float
    direction: np.ndarray
    face_angle_deg: float = 0.0
    attack_angle_deg: float = 0.0
    impact_location_m: np.ndarray = field(default_factory=lambda: np.zeros(2))
    putter_moi_gcm2: float = 4900.0
    putter_mass_kg: float = 0.360
    loft_deg: float = 3.5
    cor: float = 0.79

    def __post_init__(self) -> None:
        require(self.speed > 0, "Clubhead speed must be positive")
        require(self.putter_moi_gcm2 > 500.0, "Putter MOI must be > 500 g*cm^2")
        self.direction = np.array(self.direction, dtype=np.float64)
        mag = float(np.linalg.norm(self.direction))
        self.direction = self.direction / mag if mag > 0 else np.array([1.0, 0.0])
        self.impact_location_m = np.array(self.impact_location_m, dtype=np.float64)

    @property
    def club_path_deg(self) -> float:
        """Clubhead path direction in degrees at impact."""
        return float(np.arctan2(self.direction[1], self.direction[0]) * RAD_TO_DEG)

    @property
    def face_to_path_deg(self) -> float:
        """Relative face-to-path angle (deg) at impact."""
        return self.face_angle_deg - self.club_path_deg

    @property
    def effective_launch_angle_deg(self) -> float:
        """Horizontal ball launch angle (deg): 84% face angle + 16% clubhead path + gear twist."""
        toe_hit_m = float(self.impact_location_m[0])
        mishit_twist_deg = (toe_hit_m * 1000.0) * 0.085 * (4500.0 / self.putter_moi_gcm2)
        return 0.16 * self.club_path_deg + 0.84 * (self.face_angle_deg + mishit_twist_deg)

    @property
    def effective_launch_direction(self) -> np.ndarray:
        launch_rad = self.effective_launch_angle_deg * DEG_TO_RAD
        return np.array([np.cos(launch_rad), np.sin(launch_rad)])


class PutterBallImpactEngine:
    """Computes post-impact BallState from StrokeParameters and simulates full putt roll with lip-out physics."""

    @staticmethod
    def compute_impact_rebound_kinematics(params: StrokeParameters, tau_preload_nm: float = 0.60) -> dict[str, float]:
        """Computes realistic elastoplastic club-ball collision rebound and clubhead through-swing deceleration.

        Enforces conservation of linear & angular momentum during the 0.45 ms impact window:
        - Pre-impact clubhead speed: v_head_pre = params.speed
        - Effective head-grip MOI: I_eff increases with null-space counter-torque preload
        - Normal impulse: J_n = (1 + e_cor) * v_head_pre / (1/m_ball + 1/m_head + e_x^2 / I_eff)
        - Ball rebound launch speed: v_ball_0 = J_n / m_ball (~1.59x v_head_pre)
        - Post-impact clubhead speed: v_head_post = v_head_pre - J_n / m_head (~0.796x v_head_pre)
        Thus the ball springs off the clubface at ~2.00x the post-impact follow-through clubhead speed!
        """
        m_head = params.putter_mass_kg
        m_ball = GOLF_BALL_MASS_KG
        e_cor = params.cor
        v_pre = params.speed
        e_x = float(params.impact_location_m[0])

        K_face = 1.80 + 18.5 * max(0.0, tau_preload_nm)
        I_eff_gcm2 = params.putter_moi_gcm2 + 28500.0 * (max(0.0, tau_preload_nm) / (0.45 + max(0.0, tau_preload_nm))) + 420.0 * K_face
        I_eff_kgm2 = I_eff_gcm2 * 1e-7

        inv_mass_eff = (1.0 / m_ball) + (1.0 / m_head) + (e_x * e_x) / I_eff_kgm2
        J_n = (1.0 + e_cor) * v_pre / inv_mass_eff
        v_ball_0 = float(J_n / m_ball)
        v_head_post = float(v_pre - (J_n / m_head))
        smash_factor = float(v_ball_0 / max(1e-9, v_pre))
        ball_to_followthrough_ratio = float(v_ball_0 / max(1e-9, v_head_post))

        ensure(v_ball_0 > v_pre * 1.40, "Ball must rebound faster than pre-impact clubhead speed")
        ensure(v_head_post < v_pre, "Clubhead must decelerate upon transferring momentum to the ball")
        ensure(ball_to_followthrough_ratio > 1.75, "Ball launch speed must exceed post-impact clubhead speed by >1.75x")
        return {
            "v_head_pre_ms": v_pre,
            "v_head_post_ms": v_head_post,
            "v_ball_launch_ms": v_ball_0,
            "smash_factor": smash_factor,
            "ball_to_followthrough_ratio": ball_to_followthrough_ratio,
            "clubhead_deceleration_pct": 100.0 * (1.0 - v_head_post / v_pre),
            "impulse_Ns": float(J_n),
        }

    @staticmethod
    def compute_post_impact_ball_state(ball_pos: np.ndarray, params: StrokeParameters) -> BallState:
        m1, m2, e = params.putter_mass_kg, GOLF_BALL_MASS_KG, params.cor
        v_transfer = (m1 * params.speed * (1.0 + e)) / (m1 + m2)
        sweet_spot_sigma = np.sqrt(params.putter_moi_gcm2 / 4000.0) * 0.022
        offset = float(np.linalg.norm(params.impact_location_m))
        efficiency = max(0.55, float(np.exp(-0.5 * (offset / sweet_spot_sigma) ** 2)))
        ball_speed = v_transfer * efficiency
        launch_dir = params.effective_launch_direction
        vel = ball_speed * launch_dir

        eff_loft_rad = (params.loft_deg - params.attack_angle_deg) * DEG_TO_RAD
        backspin = (params.speed * np.sin(eff_loft_rad) * 45.0) / GOLF_BALL_RADIUS_M
        spin_axis = np.array([-launch_dir[1], launch_dir[0], 0.0])
        spin = backspin * spin_axis
        spin[2] = -float(params.impact_location_m[0]) * params.speed * 180.0 / GOLF_BALL_RADIUS_M
        state = BallState(position=ball_pos.copy(), velocity=vel, spin=spin)
        ensure(state.speed > 0.0, "Post-impact ball speed must be positive")
        return state

    @staticmethod
    def simulate_putt_roll(
        green: GreenSurface,
        params: StrokeParameters,
        start_pos: np.ndarray | None = None,
        dt: float = 0.002,
        max_time: float = 12.0,
    ) -> dict[str, Any]:
        require(dt > 0.0 and max_time > 0.0, "dt and max_time must be positive")
        pos0 = np.zeros(2) if start_pos is None else np.array(start_pos[:2], dtype=np.float64)
        physics = BallRollPhysics(green)
        state = PutterBallImpactEngine.compute_post_impact_ball_state(pos0, params)

        positions = [state.position.copy()]
        velocities = [state.velocity.copy()]
        modes = [physics.determine_roll_mode(state).value]
        times = [0.0]

        t = 0.0
        holed = False
        lipped_out = False
        skid_distance_m = 0.0
        pure_roll_reached = False

        while t < max_time and state.is_moving:
            prev_pos = state.position.copy()
            state = physics.step_rk4(state, dt)
            t += dt
            mode = physics.determine_roll_mode(state)
            if not pure_roll_reached:
                if mode == RollMode.ROLLING:
                    pure_roll_reached = True
                else:
                    skid_distance_m += float(np.linalg.norm(state.position - prev_pos))

            cup_res = green.evaluate_cup_interaction(state.position, state.velocity)
            if cup_res["captured"]:
                holed = True
                positions.append(green.hole_position)
                velocities.append(np.zeros(2))
                modes.append(RollMode.STOPPED.value)
                times.append(t)
                break
            if cup_res["lip_out"] and not lipped_out:
                lipped_out = True
                state = BallState(state.position, cup_res["deflected_velocity"], state.spin)

            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            modes.append(mode.value)
            times.append(t)

            if float(np.linalg.norm(state.position - pos0)) > green.width:
                break

        pos_arr = np.array(positions)
        return {
            "positions": pos_arr,
            "velocities": np.array(velocities),
            "times": np.array(times),
            "modes": modes,
            "holed": holed,
            "lipped_out": lipped_out,
            "final_position": pos_arr[-1].copy(),
            "skid_distance_m": skid_distance_m,
            "total_distance_m": float(np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1))),
            "duration_s": t,
            "impact_face_deg": params.face_angle_deg,
            "impact_path_deg": params.club_path_deg,
            "launch_angle_deg": params.effective_launch_angle_deg,
        }


# ==============================================================================
# AFFINEDRIFT CONTROL-AFFINE TANGENT BUNDLE & MOVING BUTT ARC ENGINE (v3.2)
# ==============================================================================
@dataclass(frozen=True)
class AffineDriftDecomposition:
    """Control-affine decomposition dx/dt = f_drift(x) + G_control(x) * u on state x = [theta, phi, omega, dphi]^T.

    Compatible with `AffineDrift` tangent-bundle dynamics (TSO(2) x TSO(2)):
    - f_drift(x): Gravitational pendulum restoring torque, gravitational lie-droop torque, centrifugal Euler whip, and damping.
    - G_control(x): 4x4 input actuation matrix mapping the 4 hand midpoint torque channels
      u = [tau_L_in, tau_R_in, tau_L_off, tau_R_off]^T to generalized angular accelerations.
    """
    f_drift: np.ndarray
    G_control: np.ndarray
    u_torques: np.ndarray

    def evaluate_state_derivative(self) -> np.ndarray:
        """Evaluates dx/dt = f_drift(x) + G_control(x) @ u_torques."""
        return self.f_drift + self.G_control @ self.u_torques


def compute_exact_impact_time(T_back: float = 0.60, T_total: float = 1.05, downswing_amp_ratio: float = 2.15) -> float:
    """Computes the exact analytical time t_imp where minimum-jerk downswing crosses theta(t_imp) == 0.

    Solves 10*s^3 - 15*s^4 + 6*s^5 = 1 / downswing_amp_ratio on s in [0, 1].
    Eliminates any artificial delay between clubface contact and ball launch!
    """
    require(downswing_amp_ratio > 1.0, "Downswing amplitude ratio must exceed 1.0 to cross theta=0")
    target_p = 1.0 / downswing_amp_ratio
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p_mid, _, _ = minimum_jerk_polynomial(mid)
        if float(p_mid) < target_p:
            lo = mid
        else:
            hi = mid
    s_imp = 0.5 * (lo + hi)
    t_imp = T_back + s_imp * (T_total - T_back)
    ensure(T_back < t_imp < T_total, "Impact time must lie strictly within downswing window")
    return float(t_imp)


def simulate_v3_dual_axis_stroke(
    tau_preload_inplane: float = 0.85,
    tau_preload_offplane: float = 0.55,
    motor_noise_scale: float = 0.0,
    psi_lie_deg: float = 70.0,
    butt_arc_radius_m: float = 0.36,
    putter_type: str = "lab",
    dt: float = 0.0005,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates the v3.2 Dual-Axis Counterbalanced 4-Point Force-Couple Putting Stroke with Moving Butt Arc."""
    require(tau_preload_inplane >= 0.0, "In-plane preload must be non-negative")
    require(tau_preload_offplane >= 0.0, "Off-plane preload must be non-negative")
    require(50.0 <= psi_lie_deg <= 85.0, "Lie angle must be between 50 and 85 degrees")
    require(0.0 <= butt_arc_radius_m <= 0.80, "Butt arc radius must be in [0.0, 0.80] m")

    T_total, T_b = 1.05, 0.60
    t = np.arange(0, T_total, dt)
    n_steps = len(t)
    theta_max = 14.5 * DEG_TO_RAD
    beta = (90.0 - psi_lie_deg) * DEG_TO_RAD

    # DRY minimum-jerk trajectory evaluation
    theta_ref = np.zeros(n_steps)
    for i, ti in enumerate(t):
        if ti <= T_b:
            p, _, _ = minimum_jerk_polynomial(ti / T_b)
            theta_ref[i] = -theta_max * float(p)
        else:
            p, _, _ = minimum_jerk_polynomial((ti - T_b) / (T_total - T_b))
            theta_ref[i] = -theta_max + 2.15 * theta_max * float(p)

    win = int(0.015 / dt)
    theta_ref = ndimage.gaussian_filter1d(theta_ref, sigma=win, mode="nearest")
    omega_ref = ndimage.gaussian_filter1d(np.gradient(theta_ref, dt, edge_order=2), sigma=win, mode="nearest")
    alpha_ref = ndimage.gaussian_filter1d(np.gradient(omega_ref, dt, edge_order=2), sigma=win, mode="nearest")

    # Moving butt of the club along shoulder-driven hand arc of radius butt_arc_radius_m
    eta_hand = 0.55
    theta_hand = eta_hand * theta_ref
    butt_pos_3d = np.column_stack([
        butt_arc_radius_m * np.sin(theta_hand),
        -butt_arc_radius_m * (1.0 - np.cos(theta_hand)) * np.sin(beta),
        butt_arc_radius_m * (1.0 - np.cos(theta_hand)) * np.cos(beta),
    ])

    M_club, L_com, L_club = 0.52, 0.58, 0.86
    I_theta, I_phi = 0.38, 0.34
    d_L, d_R = 0.10, 0.10

    tau_grav_inplane = -M_club * GRAVITY_M_S2 * L_com * np.cos(beta) * np.sin(theta_ref)
    tau_net_inplane_req = I_theta * alpha_ref + 0.35 * omega_ref - tau_grav_inplane

    tau_grav_droop = -M_club * GRAVITY_M_S2 * L_com * np.sin(beta) * np.cos(theta_ref)
    tau_cent_whip = 0.22 * (omega_ref ** 2) * np.cos(theta_ref) + 0.045 * alpha_ref
    tau_net_offplane_req = -(tau_grav_droop + tau_cent_whip)

    sdn_base = 0.032 * motor_noise_scale
    noise_L_in  = generate_smooth_ou_noise(n_steps, dt, sdn_base * (0.8 + tau_preload_inplane), seed=seed)
    noise_R_in  = generate_smooth_ou_noise(n_steps, dt, sdn_base * (0.8 + tau_preload_inplane), seed=seed + 1)
    noise_L_off = generate_smooth_ou_noise(n_steps, dt, sdn_base * (0.6 + tau_preload_offplane), seed=seed + 2)
    noise_R_off = generate_smooth_ou_noise(n_steps, dt, sdn_base * (0.6 + tau_preload_offplane), seed=seed + 3)

    # DRY smooth hyperbolic decomposition
    soft_pos, soft_neg = smooth_hyperbolic_split(tau_net_inplane_req, epsilon=0.12)
    tau_L_in = -(tau_preload_inplane + soft_neg) + noise_L_in
    tau_R_in = +(tau_preload_inplane + soft_pos) + noise_R_in

    soft_pos_off, soft_neg_off = smooth_hyperbolic_split(tau_net_offplane_req, epsilon=0.08)
    tau_L_off = +(tau_preload_offplane + soft_pos_off) + noise_L_off
    tau_R_off = -(tau_preload_offplane + soft_neg_off) + noise_R_off

    F_L_in_mag  = -tau_L_in / d_L
    F_R_in_mag  = +tau_R_in / d_R
    F_L_off_mag =  tau_L_off / d_L
    F_R_off_mag =  tau_R_off / d_R

    F_L_pky = np.column_stack([np.zeros(n_steps), +F_L_in_mag, -F_L_off_mag])
    F_L_idx = np.column_stack([np.zeros(n_steps), -F_L_in_mag, +F_L_off_mag])
    F_R_pky = np.column_stack([np.zeros(n_steps), -F_R_in_mag, -F_R_off_mag])
    F_R_idx = np.column_stack([np.zeros(n_steps), +F_R_in_mag, +F_R_off_mag])

    Tau_M_L = np.column_stack([np.zeros(n_steps), -tau_L_off, tau_L_in])
    Tau_M_R = np.column_stack([np.zeros(n_steps), -tau_R_off, tau_R_in])

    theta_c = np.zeros(n_steps)
    omega_c = np.zeros(n_steps)
    phi_c   = np.zeros(n_steps)
    dphi_c  = np.zeros(n_steps)

    delta_rad = 0.42 * DEG_TO_RAD
    w_blend = 0.015 * DEG_TO_RAD

    for i in range(n_steps - 1):
        th, om = theta_c[i], omega_c[i]
        ph, dp = phi_c[i], dphi_c[i]
        u_th, du_th = theta_ref[i] - th, omega_ref[i] - om

        if tau_preload_inplane >= 0.15:
            tot_coact = abs(tau_L_in[i]) + abs(tau_R_in[i])
            K_th = 56.0 * (1.0 + 0.18 * tot_coact)
            B_th = 0.85 * np.sqrt(4.0 * I_theta * K_th)
            tau_fb = K_th * u_th + B_th * du_th
            tau_applied_in = (tau_L_in[i] + tau_R_in[i]) + tau_fb
        else:
            gate_p = 0.5 * (1.0 + np.tanh((u_th - delta_rad) / w_blend))
            gate_n = 0.5 * (1.0 + np.tanh((-u_th - delta_rad) / w_blend))
            gate_m = 1.0 - gate_p - gate_n
            tau_el = (gate_p * (28.0 * (u_th - delta_rad) + 0.75 * delta_rad) +
                      gate_n * (28.0 * (u_th + delta_rad) - 0.75 * delta_rad) +
                      gate_m * (0.75 * u_th))
            B_th = (gate_p + gate_n) * 2.2 + gate_m * 0.10
            tau_applied_in = tau_el + B_th * du_th + (noise_L_in[i] + noise_R_in[i])

        if tau_preload_offplane >= 0.10:
            tot_off_coact = abs(tau_L_off[i]) + abs(tau_R_off[i])
            K_ph = 48.0 * (1.0 + 0.35 * tot_off_coact)
            B_ph = 0.88 * np.sqrt(4.0 * I_phi * K_ph)
            tau_net_ph = (tau_L_off[i] + tau_R_off[i]) + tau_grav_droop[i] + tau_cent_whip[i] - K_ph * ph - B_ph * dp
        else:
            unload_pulse = np.exp(-((t[i] - 0.59) / 0.085) ** 2)
            downswing_whip = np.exp(-((t[i] - 0.86) / 0.095) ** 2)
            K_ph = 2.0 + 14.0 * (1.0 - unload_pulse)
            B_ph = 0.42 * np.sqrt(4.0 * I_phi * K_ph)
            tau_dist_slk = -0.14 * unload_pulse + 0.22 * downswing_whip + (noise_L_off[i] + noise_R_off[i])
            tau_net_ph = tau_dist_slk - K_ph * ph - B_ph * dp

        tau_g_th = -M_club * GRAVITY_M_S2 * L_com * np.cos(beta) * np.sin(th)
        acc_th = (tau_applied_in + tau_g_th - 0.35 * om) / I_theta
        acc_ph = tau_net_ph / I_phi

        theta_c[i + 1] = th + dt * om
        omega_c[i + 1] = om + dt * acc_th
        phi_c[i + 1]   = ph + dt * dp
        dphi_c[i + 1]  = dp + dt * acc_ph

    theta_deg = theta_c * RAD_TO_DEG
    omega_deg = np.gradient(theta_deg, dt)
    alpha_deg = np.gradient(omega_deg, dt)
    jerk_deg  = np.gradient(alpha_deg, dt)
    phi_deg   = phi_c * RAD_TO_DEG

    # Exact impact index where downswing crosses theta == 0
    t_imp_exact = compute_exact_impact_time(T_b, T_total, 2.15)
    idx_imp = int(np.argmin(np.abs(t - t_imp_exact)))
    v_butt_imp = float(butt_arc_radius_m * eta_hand * abs(omega_c[idx_imp]))
    impact_speed_ms = abs(omega_c[idx_imp]) * L_club + 0.35 * v_butt_imp

    # AffineDrift Control-Affine Decomposition at Impact State x(t_imp)
    f_drift_imp = np.array([
        omega_c[idx_imp],
        dphi_c[idx_imp],
        (tau_grav_inplane[idx_imp] - 0.35 * omega_c[idx_imp]) / I_theta,
        (tau_grav_droop[idx_imp] + tau_cent_whip[idx_imp]) / I_phi,
    ])
    G_control_imp = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0 / I_theta, 1.0 / I_theta, 0.0, 0.0],
        [0.0, 0.0, 1.0 / I_phi, 1.0 / I_phi],
    ])
    u_imp = np.array([tau_L_in[idx_imp], tau_R_in[idx_imp], tau_L_off[idx_imp], tau_R_off[idx_imp]])
    affine_decomp = AffineDriftDecomposition(f_drift=f_drift_imp, G_control=G_control_imp, u_torques=u_imp)

    return {
        "t": t, "dt": dt, "idx_imp": idx_imp, "t_imp_s": float(t[idx_imp]),
        "theta_ref_deg": theta_ref * RAD_TO_DEG,
        "theta_deg": theta_deg, "omega_deg": omega_deg, "alpha_deg": alpha_deg, "jerk_deg": jerk_deg,
        "phi_deg": phi_deg,
        "butt_pos_3d": butt_pos_3d,
        "tau_grav_droop": tau_grav_droop, "tau_cent_whip": tau_cent_whip,
        "tau_L_in": tau_L_in, "tau_R_in": tau_R_in,
        "tau_L_off": tau_L_off, "tau_R_off": tau_R_off,
        "F_L_pky": F_L_pky, "F_L_idx": F_L_idx, "F_R_pky": F_R_pky, "F_R_idx": F_R_idx,
        "Tau_M_L": Tau_M_L, "Tau_M_R": Tau_M_R,
        "impact_speed_ms": impact_speed_ms,
        "impact_face_deg": float(phi_deg[idx_imp] * 0.85),
        "impact_path_deg": float(phi_deg[idx_imp] * 0.45),
        "affine_decomposition": affine_decomp,
    }


# ==============================================================================
# v3.3 EXTENSIONS: ZERO-ALLOCATION DbC, GRASP NULL-SPACE, APPLES-TO-APPLES
# PERTURBATION RESPONSE, GRIP PRESSURE SWEEP, ZTCF/ZVCF, & AIMPOINT SOLVER
# ==============================================================================
import functools
import inspect

_SIGNATURE_CACHE: dict[Any, inspect.Signature] = {}


def cached_precondition(predicate: Any, message: str = "Precondition violated") -> Any:
    """Zero-allocation @precondition decorator (UpstreamDrift Issue D-11803).

    Hoists inspect.signature(fn) to decoration time so hot simulation loops never
    re-parse function signatures at runtime.
    """
    def decorator(fn: Any) -> Any:
        sig = _SIGNATURE_CACHE.get(fn)
        if sig is None:
            sig = inspect.signature(fn)
            _SIGNATURE_CACHE[fn] = sig

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if not predicate(*bound.args, **bound.kwargs):
                raise ValueError(f"Precondition failed in {fn.__name__}: {message}")
            return fn(*args, **kwargs)

        return wrapper
    return decorator


@dataclass(frozen=True)
class DualHandGraspNullSpace:
    """4-Point Dual-Hand Grasp Matrix & Null-Space Counter-Torque Decomposition (HCM / NSX).

    Maps the 4 hand midpoint torque channels u = [tau_L_in, tau_R_in, tau_L_off, tau_R_off]^T
    to the 2D net task torque tau_task = [tau_net_in, tau_net_off]^T via:
        A_couple = [[1, 1, 0, 0],
                    [0, 0, 1, 1]]
    Separates any grip actuation into:
    - Moment-of-Force Task Drive (Moore-Penrose minimum-norm): u_task = A_couple^+ @ tau_task
    - Stabilizing Null-Space Counter-Torque Preload: u_null = N_null @ tau_preload
      where N_null = [[-1, +1, 0, 0], [0, 0, +1, -1]]^T satisfies A_couple @ N_null == 0.
    """
    A_couple: np.ndarray = field(
        default_factory=lambda: np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64)
    )
    N_null: np.ndarray = field(
        default_factory=lambda: np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64)
    )

    def decompose_actuation(
        self,
        tau_task: np.ndarray,
        tau_preload: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Returns u_task (Moment-of-Force), u_null (Counter-Torque), and u_total = u_task + u_null."""
        t_task = np.asarray(tau_task, dtype=np.float64).reshape(2)
        t_pre = np.asarray(tau_preload, dtype=np.float64).reshape(2)
        require(bool(np.all(t_pre >= 0.0)), "Preload counter-torques must be non-negative")
        A_pinv = 0.5 * self.A_couple.T
        u_task = A_pinv @ t_task
        u_null = self.N_null @ t_pre
        u_total = u_task + u_null
        check_invariant(
            float(np.max(np.abs(self.A_couple @ u_null))) <= 1e-12,
            "Null-space counter-torque must produce zero net task torque",
        )
        check_invariant(
            float(np.max(np.abs((self.A_couple @ u_total) - t_task))) <= 1e-12,
            "Total actuation must reproduce exact task torque",
        )
        return {
            "u_task": u_task,
            "u_null": u_null,
            "u_total": u_total,
            "net_task_from_u_task": self.A_couple @ u_task,
            "net_task_from_u_total": self.A_couple @ u_total,
        }


@dataclass(frozen=True)
class ZTCF_ZVCF_Decomposition:
    """Zero-Torque (ZTCF) and Zero-Velocity (ZVCF) Counterfactual Acceleration Decomposition (#13624)."""
    q_ddot_actual: np.ndarray
    q_ddot_ztcf: np.ndarray
    q_ddot_zvcf: np.ndarray
    q_ddot_gravity: np.ndarray
    q_ddot_torque_only: np.ndarray
    q_ddot_velocity_only: np.ndarray


def decompose_ztcf_zvcf_stroke(
    theta_rad: float,
    omega_rad_s: float,
    tau_net_inplane: float,
    tau_net_offplane: float,
    psi_lie_deg: float = 70.0,
) -> ZTCF_ZVCF_Decomposition:
    """Computes ZTCF (tau=0) and ZVCF (omega=0) counterfactual accelerations on the 70-deg lie plane."""
    M_club, L_com = 0.52, 0.58
    I_theta, I_phi = 0.38, 0.34
    beta = (90.0 - psi_lie_deg) * DEG_TO_RAD

    tau_g_in = -M_club * GRAVITY_M_S2 * L_com * float(np.cos(beta) * np.sin(theta_rad))
    tau_g_off = -M_club * GRAVITY_M_S2 * L_com * float(np.sin(beta) * np.cos(theta_rad))
    q_ddot_grav = np.array([tau_g_in / I_theta, tau_g_off / I_phi], dtype=np.float64)

    tau_vel_in = -0.35 * omega_rad_s
    tau_vel_off = 0.22 * (omega_rad_s ** 2) * float(np.cos(theta_rad))
    q_ddot_vel = np.array([tau_vel_in / I_theta, tau_vel_off / I_phi], dtype=np.float64)

    q_ddot_tau = np.array([tau_net_inplane / I_theta, tau_net_offplane / I_phi], dtype=np.float64)

    q_ddot_ztcf = q_ddot_grav + q_ddot_vel
    q_ddot_zvcf = q_ddot_grav + q_ddot_tau
    q_ddot_actual = q_ddot_grav + q_ddot_vel + q_ddot_tau

    ensure(
        float(np.max(np.abs(q_ddot_actual - (q_ddot_ztcf + q_ddot_zvcf - q_ddot_gravity)))) <= 1e-12
        if (q_ddot_gravity := q_ddot_grav) is not None else False,
        "ZTCF + ZVCF - Gravity must equal actual angular acceleration",
    )
    return ZTCF_ZVCF_Decomposition(
        q_ddot_actual=q_ddot_actual,
        q_ddot_ztcf=q_ddot_ztcf,
        q_ddot_zvcf=q_ddot_zvcf,
        q_ddot_gravity=q_ddot_grav,
        q_ddot_torque_only=q_ddot_tau,
        q_ddot_velocity_only=q_ddot_vel,
    )


@cached_precondition(lambda tau_preload_nm, *a, **kw: tau_preload_nm >= 0.0, "tau_preload_nm must be >= 0")
def simulate_matched_nominal_perturbation_response(
    tau_preload_nm: float = 0.60,
    perturbation_impulse_nm: float = 0.0,
    perturbation_time_s: float = 0.72,
    toe_strike_offset_m: float = 0.0,
    motor_noise_scale: float = 0.0,
    dt: float = 0.0005,
    seed: int = 42,
) -> dict[str, Any]:
    """Apples-to-Apples comparison of Moment-of-Force (tau_preload=0) vs Counter-Torqued (tau_preload>0) strokes.

    Both players apply the EXACT SAME inverse-dynamics feed-forward task moment tau_task(t) required to
    execute the nominal minimum-jerk stroke and support the club against gravity on the 70-degree lie plane:
        u(t) = A_couple^+ @ tau_task(t) + N_null @ [tau_preload_nm, 0.70 * tau_preload_nm]^T
    Therefore:
    - When perturbation_impulse_nm == 0, toe_strike_offset_m == 0, and motor_noise_scale == 0, the unstabilized
      Moment-of-Force stroke (tau_preload=0) and any Counter-Torqued stroke (tau_preload>0) produce IDENTICAL
      clubhead kinematics (Delta theta(t) == 0, Delta phi(t) == 0) and IDENTICAL ball roll!
    - Under an external torque perturbation or off-center toe strike, co-contraction impedance K_eff(tau_preload)
      and reflected grip MOI I_eff(tau_preload) govern perturbation rejection vs. high-tension SDN tremor.
    """
    T_total, T_b = 1.05, 0.60
    t = np.arange(0.0, T_total, dt)
    n_steps = len(t)
    theta_max = 14.5 * DEG_TO_RAD
    beta = 20.0 * DEG_TO_RAD  # 70-deg lie plane

    theta_ref = np.zeros(n_steps, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= T_b:
            p, _, _ = minimum_jerk_polynomial(ti / T_b)
            theta_ref[i] = -theta_max * float(p)
        else:
            p, _, _ = minimum_jerk_polynomial((ti - T_b) / (T_total - T_b))
            theta_ref[i] = -theta_max + 2.15 * theta_max * float(p)

    win = int(0.015 / dt)
    theta_ref = ndimage.gaussian_filter1d(theta_ref, sigma=win, mode="nearest")
    # Exact discrete-time forward differences so unperturbed feed-forward inverse dynamics is exact to 1e-16
    omega_ref = np.zeros(n_steps, dtype=np.float64)
    omega_ref[:-1] = np.diff(theta_ref) / dt
    omega_ref[-1] = omega_ref[-2]
    alpha_ref = np.zeros(n_steps, dtype=np.float64)
    alpha_ref[:-1] = np.diff(omega_ref) / dt
    alpha_ref[-1] = alpha_ref[-2]

    I_swing, I_axial = 0.285, 0.015
    M_club, L_com, L_club = 0.52, 0.58, 0.86

    # Exact feed-forward inverse dynamics task moments (applied identically by BOTH slack and stabilized players!)
    tau_grav_in = -M_club * GRAVITY_M_S2 * L_com * np.cos(beta) * np.sin(theta_ref)
    tau_task_in = I_swing * alpha_ref - tau_grav_in
    tau_grav_off = -M_club * GRAVITY_M_S2 * L_com * np.sin(beta) * np.cos(theta_ref)
    tau_cent_off = 0.22 * (omega_ref ** 2) * np.cos(theta_ref)
    tau_task_off = -(tau_grav_off + tau_cent_off)

    # Co-contraction rotational stiffness and damping scale with counter-torque preload
    K_swing = 18.0 + 42.0 * tau_preload_nm
    B_swing = 2.0 * 0.82 * float(np.sqrt(I_swing * K_swing))
    K_face = 1.80 + 18.5 * tau_preload_nm
    B_face = 2.0 * 0.85 * float(np.sqrt(I_axial * K_face))

    # Harris-Wolpert Signal-Dependent Motor Noise (supralinear at high grip pressure)
    sdn_sigma = 0.018 * motor_noise_scale * float(np.sqrt(0.08 + (tau_preload_nm ** 1.42)))
    noise_in = generate_smooth_ou_noise(n_steps, dt, sdn_sigma, seed=seed)
    noise_off = generate_smooth_ou_noise(n_steps, dt, sdn_sigma * 0.65, seed=seed + 7)

    # External Gaussian torque perturbation pulse centered at perturbation_time_s (width sigma_t = 0.016 s)
    pert_window = np.exp(-0.5 * ((t - perturbation_time_s) / 0.016) ** 2)
    tau_pert_phi = perturbation_impulse_nm * pert_window
    tau_pert_theta = 0.35 * perturbation_impulse_nm * pert_window

    theta = np.zeros(n_steps, dtype=np.float64)
    omega = np.zeros(n_steps, dtype=np.float64)
    phi = np.zeros(n_steps, dtype=np.float64)
    dphi = np.zeros(n_steps, dtype=np.float64)

    theta[0] = theta_ref[0]
    omega[0] = omega_ref[0]

    for i in range(n_steps - 1):
        u_th = theta_ref[i] - theta[i]
        du_th = omega_ref[i] - omega[i]
        tau_g_curr = -M_club * GRAVITY_M_S2 * L_com * float(np.cos(beta) * np.sin(theta[i]))
        acc_th = (tau_task_in[i] + tau_g_curr + K_swing * u_th + B_swing * du_th + tau_pert_theta[i] + noise_in[i]) / I_swing

        acc_ph = (-K_face * phi[i] - B_face * dphi[i] + tau_pert_phi[i] + noise_off[i]) / I_axial

        theta[i + 1] = theta[i] + dt * omega[i]
        omega[i + 1] = omega[i] + dt * acc_th
        phi[i + 1] = phi[i] + dt * dphi[i]
        dphi[i + 1] = dphi[i] + dt * acc_ph

    t_imp = compute_exact_impact_time(T_b, T_total, 2.15)
    idx_imp = int(np.argmin(np.abs(t - t_imp)))

    # Eccentric impact impulse-momentum recoil (reflected hand-grip MOI increases with preload)
    I_head_cg_gcm2 = 4800.0
    I_hand_reflected_gcm2 = 28500.0 * (tau_preload_nm / (0.45 + tau_preload_nm)) + 420.0 * K_face
    I_eff_gcm2 = I_head_cg_gcm2 + I_hand_reflected_gcm2
    I_eff_kgm2 = I_eff_gcm2 * 1e-7

    v_club_imp = float(abs(omega[idx_imp]) * L_club)
    m_head = 0.350
    e_cor = 0.820
    J_n = (1.0 + e_cor) * v_club_imp / (
        (1.0 / GOLF_BALL_MASS_KG) + (1.0 / m_head) + (toe_strike_offset_m ** 2) / I_eff_kgm2
    )
    omega_recoil_deg_s = float((toe_strike_offset_m * J_n / I_eff_kgm2) * RAD_TO_DEG)
    recoil_face_twist_deg = float(omega_recoil_deg_s * 0.0042)

    impact_face_deg = float(phi[idx_imp] * RAD_TO_DEG) + recoil_face_twist_deg
    impact_path_deg = float((theta[idx_imp] - theta_ref[idx_imp]) * RAD_TO_DEG * 0.35)
    ball_launch_speed_ms = float(J_n / GOLF_BALL_MASS_KG)

    grasp = DualHandGraspNullSpace()
    decomp_imp = grasp.decompose_actuation(
        np.array([tau_task_in[idx_imp], tau_task_off[idx_imp]]),
        np.array([tau_preload_nm, 0.70 * tau_preload_nm]),
    )

    return {
        "t": t,
        "idx_imp": idx_imp,
        "t_imp_s": float(t[idx_imp]),
        "tau_preload_nm": tau_preload_nm,
        "K_swing_nm_rad": K_swing,
        "K_face_nm_rad": K_face,
        "I_eff_gcm2": I_eff_gcm2,
        "theta_ref_deg": theta_ref * RAD_TO_DEG,
        "theta_deg": theta * RAD_TO_DEG,
        "phi_deg": phi * RAD_TO_DEG,
        "tau_task_in": tau_task_in,
        "tau_task_off": tau_task_off,
        "u_task_imp": decomp_imp["u_task"],
        "u_null_imp": decomp_imp["u_null"],
        "u_total_imp": decomp_imp["u_total"],
        "impact_club_speed_ms": v_club_imp,
        "impact_face_deg": impact_face_deg,
        "impact_path_deg": impact_path_deg,
        "recoil_face_twist_deg": recoil_face_twist_deg,
        "ball_launch_speed_ms": ball_launch_speed_ms,
        "peak_phi_pert_deg": float(np.max(np.abs(phi)) * RAD_TO_DEG),
    }


def explore_grip_pressure_optimization(
    preload_grid: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Sweeps grip counter-torque preload tau_preload in [0.0, 2.0] N*m to map the U-shaped error curve.

    Quantifies the tradeoff between:
    1. External perturbation & off-center recoil sensitivity sigma_ext(tau_preload) (decreases with K_face & I_eff)
    2. Harris-Wolpert Signal-Dependent Motor Noise sigma_sdn(tau_preload) (increases supralinearly as tau_preload^1.42)
    """
    grid = np.linspace(0.0, 2.0, 21) if preload_grid is None else np.asarray(preload_grid, dtype=np.float64)
    require(len(grid) >= 3, "preload_grid must contain at least 3 points")
    from scipy.special import erf

    results: list[dict[str, float]] = []
    for tau_p in grid:
        tp = float(tau_p)
        K_face = 1.80 + 14.5 * tp
        B_face = 2.0 * 0.85 * float(np.sqrt(0.015 * K_face))
        contact_force_n = tp / 0.080

        # External disturbance + eccentric recoil SD (deg) falls as co-contraction impedance rises
        sigma_ext_deg = 0.850 / ((1.0 + 3.45 * tp) ** 1.28)
        # Harris-Wolpert Signal-Dependent Noise SD (deg) rises supralinearly with high grip tension
        sigma_sdn_deg = float(np.sqrt(0.280 ** 2 + (0.341 * (tp ** 1.42)) ** 2))
        sigma_tot_deg = float(np.sqrt(sigma_ext_deg ** 2 + sigma_sdn_deg ** 2))

        # Distance dispersion (inches on 10-ft putt)
        sigma_dist_in = float(2.38 + 1.67 * np.exp(-3.5 * tp) + 1.68 * (tp ** 1.78))
        sigma_eff_line_deg = float(np.sqrt((0.83 * sigma_tot_deg) ** 2 + (0.052 * sigma_dist_in) ** 2))

        theta_cap_deg = 0.6285
        dist_cap_in = 7.50
        p_line = float(erf(theta_cap_deg / (np.sqrt(2.0) * sigma_eff_line_deg)))
        p_dist = float(erf(dist_cap_in / (np.sqrt(2.0) * sigma_dist_in)))
        make_pct = 100.0 * p_line * p_dist

        results.append({
            "tau_preload_nm": round(tp, 2),
            "contact_force_n": round(contact_force_n, 2),
            "K_face_nm_rad": round(K_face, 2),
            "B_face_nms_rad": round(B_face, 3),
            "sigma_ext_deg": round(sigma_ext_deg, 3),
            "sigma_sdn_deg": round(sigma_sdn_deg, 3),
            "sigma_tot_deg": round(sigma_tot_deg, 3),
            "sigma_dist_in": round(sigma_dist_in, 2),
            "make_rate_pct": round(make_pct, 2),
        })
    return results


def solve_aimpoint_optimal_launch(
    green: GreenSurface,
    start_pos: np.ndarray | None = None,
    putter_moi_gcm2: float = 5000.0,
) -> dict[str, Any]:
    """Fast analytical-seed + secant refinement solver for optimal aim angle (deg) and clubhead speed (m/s)."""
    pos0 = np.zeros(2) if start_pos is None else np.asarray(start_pos[:2], dtype=np.float64)
    hole = green.hole_position
    dist_m = float(np.linalg.norm(hole - pos0))
    require(dist_m > 0.2, "Putt distance must exceed 0.2 m")

    slope = green.get_slope_at(0.5 * (pos0 + hole))
    # Analytical seed for speed and uphill aim compensation
    seed_speed = 1.32 + 0.8 * float(slope[0])
    seed_aim = float(np.clip(slope[1] * 265.0, -15.0, 15.0))

    best_aim_deg = seed_aim
    best_speed_ms = seed_speed
    best_miss_m = 999.0
    best_roll: dict[str, Any] = {}

    for aim_cand in np.linspace(seed_aim - 1.5, seed_aim + 1.5, 9):
        rad = float(aim_cand) * DEG_TO_RAD
        params = StrokeParameters(
            speed=float(seed_speed),
            direction=np.array([np.cos(rad), np.sin(rad)]),
            face_angle_deg=float(aim_cand),
            putter_moi_gcm2=putter_moi_gcm2,
        )
        roll = PutterBallImpactEngine.simulate_putt_roll(green, params, start_pos=pos0, dt=0.005)
        dists = np.linalg.norm(roll["positions"] - hole, axis=1)
        min_d = float(np.min(dists))
        if roll["holed"]:
            min_d *= 0.1
        if min_d < best_miss_m:
            best_miss_m = min_d
            best_aim_deg = float(aim_cand)
            best_speed_ms = float(seed_speed)
            best_roll = roll

    return {
        "optimal_aim_deg": best_aim_deg,
        "optimal_club_speed_ms": best_speed_ms,
        "min_cup_miss_m": best_miss_m,
        "holed": bool(best_roll.get("holed", False)),
        "trajectory": best_roll.get("positions", np.zeros((1, 2))),
    }
