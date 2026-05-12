//! Aerodynamics calculations for golf ball flight.
//!
//! Computes drag, lift, and Magnus forces using empirical correlations
//! from golf ball aerodynamics literature.
//!
//! # Design by Contract
//! - All velocity/spin inputs must be finite (no NaN/Inf)
//! - Air density must be positive
//! - Ball radius and area must be positive
//! - All force outputs are guaranteed finite
//!
//! # References
//! - Smits, A.J., & Ogg, S. (2004). Golf Ball Aerodynamics. Physics Today.
//! - Jorgensen, T. (1999). The Physics of Golf. Springer.

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

const MIN_VALID_REYNOLDS: f64 = 1.0e3;
const MAX_VALID_REYNOLDS: f64 = 1.0e7;
const CD_PRECRITICAL: f64 = 0.50;
const RE_MIN_CD: f64 = 7.0e4;
const CD_HIGH: f64 = 0.30;
const TRANSITION_WIDTH: f64 = 0.16;
const DRAG_CRISIS_RECOVERY_RE: f64 = 7.0e5;

/// Air properties at given atmospheric conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AirProperties {
    /// Air density [kg/m³]
    pub density: f64,
    /// Dynamic viscosity [Pa·s]
    pub viscosity: f64,
    /// Temperature [K]
    pub temperature: f64,
    /// Pressure [Pa]
    pub pressure: f64,
}

impl Default for AirProperties {
    /// Sea level, 15°C standard conditions.
    fn default() -> Self {
        Self {
            density: 1.225,
            viscosity: 1.81e-5,
            temperature: 288.15,
            pressure: 101_325.0,
        }
    }
}

impl AirProperties {
    /// Validate public air properties in all build modes.
    pub fn validate(&self) -> Result<(), String> {
        if !self.density.is_finite() || self.density < 0.0 {
            return Err(format!(
                "AirProperties.density must be finite and non-negative, got {}",
                self.density
            ));
        }
        if !self.viscosity.is_finite() || self.viscosity <= 0.0 {
            return Err(format!(
                "AirProperties.viscosity must be finite and positive, got {}",
                self.viscosity
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(format!(
                "AirProperties.temperature must be finite and positive, got {}",
                self.temperature
            ));
        }
        if !self.pressure.is_finite() || self.pressure < 0.0 {
            return Err(format!(
                "AirProperties.pressure must be finite and non-negative, got {}",
                self.pressure
            ));
        }
        Ok(())
    }
}

/// Golf ball physical properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AeroBallProperties {
    /// Ball mass [kg]
    pub mass: f64,
    /// Ball radius [m]
    pub radius: f64,
    /// Cross-sectional area [m²]
    pub area: f64,
    /// Baseline drag coefficient (turbulent regime)
    pub drag_coefficient: f64,
    /// Spin decay time constant [1/s]
    pub spin_decay_rate: f64,
}

impl Default for AeroBallProperties {
    /// Standard golf ball (45.93 g, 42.7 mm diameter).
    fn default() -> Self {
        let radius = 0.02135;
        Self {
            mass: 0.04593,
            radius,
            area: std::f64::consts::PI * radius * radius,
            drag_coefficient: 0.25,
            spin_decay_rate: 0.1,
        }
    }
}

impl AeroBallProperties {
    /// Validate public golf-ball aerodynamic properties in all build modes.
    pub fn validate(&self) -> Result<(), String> {
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(format!(
                "AeroBallProperties.mass must be finite and positive, got {}",
                self.mass
            ));
        }
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(format!(
                "AeroBallProperties.radius must be finite and positive, got {}",
                self.radius
            ));
        }
        if !self.area.is_finite() || self.area <= 0.0 {
            return Err(format!(
                "AeroBallProperties.area must be finite and positive, got {}",
                self.area
            ));
        }
        if !self.drag_coefficient.is_finite() || self.drag_coefficient < 0.0 {
            return Err(format!(
                "AeroBallProperties.drag_coefficient must be finite and non-negative, got {}",
                self.drag_coefficient
            ));
        }
        if !self.spin_decay_rate.is_finite() || self.spin_decay_rate < 0.0 {
            return Err(format!(
                "AeroBallProperties.spin_decay_rate must be finite and non-negative, got {}",
                self.spin_decay_rate
            ));
        }
        Ok(())
    }
}

/// Result of an aerodynamics force computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AeroForces {
    /// Drag force vector [N] (opposes velocity)
    pub drag: Vector3,
    /// Lift force vector [N] (perpendicular to velocity, in spin plane)
    pub lift: Vector3,
    /// Magnus force vector [N] (ω × v direction)
    pub magnus: Vector3,
}

/// Compute all aerodynamic forces on a golf ball.
///
/// # Design by Contract
/// ## Preconditions
/// - `velocity` components must be finite
/// - `spin` components must be finite
/// - `air.density` must be positive
/// - `ball.area` must be positive
///
/// ## Postconditions
/// - All returned force components are finite
#[must_use]
pub fn compute_aero_forces(
    velocity: &Vector3,
    spin: &Vector3,
    ball: &AeroBallProperties,
    air: &AirProperties,
) -> AeroForces {
    assert!(
        velocity.x.is_finite() && velocity.y.is_finite() && velocity.z.is_finite(),
        "DbC: velocity must be finite"
    );
    assert!(
        spin.x.is_finite() && spin.y.is_finite() && spin.z.is_finite(),
        "DbC: spin must be finite"
    );
    ball.validate()
        .expect("invalid aerodynamic ball properties");
    air.validate().expect("invalid air properties");

    let drag = compute_drag(velocity, ball, air);
    let lift = compute_lift(velocity, spin, ball, air);
    let magnus = compute_magnus(velocity, spin, ball, air);

    AeroForces { drag, lift, magnus }
}

/// Compute drag force opposing motion.
///
/// F_drag = -0.5 * ρ * Cd * A * |v|² * v̂
#[must_use]
pub fn compute_drag(velocity: &Vector3, ball: &AeroBallProperties, air: &AirProperties) -> Vector3 {
    let speed = velocity.magnitude();
    if speed < 1e-6 {
        return Vector3::new(0.0, 0.0, 0.0);
    }

    let cd = compute_drag_coefficient(speed, ball, air);
    let f_mag = 0.5 * air.density * cd * ball.area * speed * speed;

    // Opposite to velocity direction
    let inv_speed = 1.0 / speed;
    Vector3::new(
        -f_mag * velocity.x * inv_speed,
        -f_mag * velocity.y * inv_speed,
        -f_mag * velocity.z * inv_speed,
    )
}

/// Compute lift force from backspin.
///
/// Lift direction: spin_axis × velocity (normalized)
#[must_use]
pub fn compute_lift(
    velocity: &Vector3,
    spin: &Vector3,
    ball: &AeroBallProperties,
    air: &AirProperties,
) -> Vector3 {
    let speed = velocity.magnitude();
    if speed < 1e-6 {
        return Vector3::new(0.0, 0.0, 0.0);
    }

    let spin_mag = spin.magnitude();
    let spin_ratio = ball.radius * spin_mag / (speed + 1e-10);
    let cl = compute_lift_coefficient(spin_ratio);

    // Spin axis (unit vector)
    let spin_inv = 1.0 / (spin_mag + 1e-10);
    let spin_axis = Vector3::new(spin.x * spin_inv, spin.y * spin_inv, spin.z * spin_inv);

    // Lift direction: spin_axis × velocity
    let lift_dir = spin_axis.cross(velocity);
    let lift_norm = lift_dir.magnitude();

    if lift_norm < 1e-6 {
        return Vector3::new(0.0, 0.0, 0.0);
    }

    let f_mag = 0.5 * air.density * cl * ball.area * speed * speed;
    let inv_norm = 1.0 / lift_norm;

    Vector3::new(
        f_mag * lift_dir.x * inv_norm,
        f_mag * lift_dir.y * inv_norm,
        f_mag * lift_dir.z * inv_norm,
    )
}

/// Compute Magnus force from spin-induced pressure differential.
///
/// F_magnus = 0.5 * ρ * Cm * A * |v|² * (ω × v) / |ω × v|
#[must_use]
pub fn compute_magnus(
    velocity: &Vector3,
    spin: &Vector3,
    ball: &AeroBallProperties,
    air: &AirProperties,
) -> Vector3 {
    let speed = velocity.magnitude();
    let spin_mag = spin.magnitude();

    if speed < 1e-6 || spin_mag < 1e-6 {
        return Vector3::new(0.0, 0.0, 0.0);
    }

    let magnus_dir = spin.cross(velocity);
    let magnus_norm = magnus_dir.magnitude();

    if magnus_norm < 1e-6 {
        return Vector3::new(0.0, 0.0, 0.0);
    }

    let spin_param = ball.radius * spin_mag / speed;
    let cm = compute_magnus_coefficient(spin_param);
    let f_mag = 0.5 * air.density * cm * ball.area * speed * speed;
    let inv_norm = 1.0 / magnus_norm;

    Vector3::new(
        f_mag * magnus_dir.x * inv_norm,
        f_mag * magnus_dir.y * inv_norm,
        f_mag * magnus_dir.z * inv_norm,
    )
}

/// Compute drag coefficient based on Reynolds number.
///
/// Golf ball dimples reduce drag at high Re through turbulent boundary
/// layer transition.
#[must_use]
pub fn compute_drag_coefficient(speed: f64, ball: &AeroBallProperties, air: &AirProperties) -> f64 {
    let re = air.density * speed * (2.0 * ball.radius) / air.viscosity;
    let clamped_re = re.clamp(MIN_VALID_REYNOLDS, MAX_VALID_REYNOLDS);
    cd_dimpled_sphere(clamped_re, ball.drag_coefficient)
        .expect("clamped Reynolds number and validated ball Cd must be accepted")
}

fn smoothstep_tanh(x: f64, x0: f64, width: f64) -> f64 {
    0.5 * (1.0 + ((x - x0) / width).tanh())
}

/// Return the dimpled-sphere drag coefficient for a Reynolds number.
///
/// This ports `src.shared.python.physics.atmosphere.cd_dimpled_sphere` into
/// Rust so the timestep kernel uses the same smooth drag-crisis model as the
/// Python reference path.
///
/// # Design by Contract
/// ## Preconditions
/// - `reynolds` must be finite and in `[1e3, 1e7]`
/// - `base_cd` must be finite
///
/// ## Postconditions
/// - Returned Cd is finite and clamped to `[0.15, 0.55]`
pub fn cd_dimpled_sphere(reynolds: f64, base_cd: f64) -> Result<f64, String> {
    if !reynolds.is_finite() || reynolds <= 0.0 {
        return Err("reynolds must be a positive finite number".to_string());
    }
    if !(MIN_VALID_REYNOLDS..=MAX_VALID_REYNOLDS).contains(&reynolds) {
        return Err(format!(
            "reynolds={reynolds} is outside the supported range [{MIN_VALID_REYNOLDS}, {MAX_VALID_REYNOLDS}]"
        ));
    }
    if !base_cd.is_finite() {
        return Err("base_cd must be finite".to_string());
    }

    let cd_floor = base_cd.clamp(0.18, 0.30);
    let log_re = reynolds.log10();
    let log_re_crisis = RE_MIN_CD.log10();
    let log_re_recovery = DRAG_CRISIS_RECOVERY_RE.log10();

    let crisis_blend = smoothstep_tanh(log_re, log_re_crisis, TRANSITION_WIDTH);
    let cd_crisis = CD_PRECRITICAL + (cd_floor - CD_PRECRITICAL) * crisis_blend;

    let recovery_blend = smoothstep_tanh(log_re, log_re_recovery, 0.45);
    let cd = cd_crisis + (CD_HIGH - cd_floor) * recovery_blend;

    Ok(cd.clamp(0.15, 0.55))
}

/// Compute lift coefficient from dimensionless spin ratio.
///
/// Empirical relationship (Smits & Ogg): Cl saturates at ~0.4.
#[must_use]
pub fn compute_lift_coefficient(spin_ratio: f64) -> f64 {
    let cl_max = 0.4;
    cl_max * (1.0 - (-spin_ratio / 0.1).exp())
}

/// Compute Magnus coefficient from spin parameter ωR/v.
///
/// Approximately linear for small spin_param, capped at 0.2.
#[must_use]
pub fn compute_magnus_coefficient(spin_param: f64) -> f64 {
    0.4 * spin_param.min(0.5)
}

/// Compute spin decay over a time step.
///
/// Spin decays exponentially: ω(t+dt) = ω(t) * exp(-λ * dt)
///
/// # Preconditions
/// - `dt` must be positive
#[must_use]
pub fn compute_spin_decay(spin: &Vector3, dt: f64, spin_decay_rate: f64) -> Vector3 {
    assert!(
        dt.is_finite() && dt > 0.0,
        "DbC: dt must be finite and positive"
    );
    assert!(
        spin_decay_rate.is_finite() && spin_decay_rate >= 0.0,
        "DbC: spin decay rate must be finite and non-negative"
    );
    let decay = (-spin_decay_rate * dt).exp();
    Vector3::new(spin.x * decay, spin.y * decay, spin.z * decay)
}

/// Create air properties for a given altitude using ISA model.
///
/// # DRY
/// Delegates to `tools_core::atmosphere::atmosphere_at_altitude` —
/// the canonical ISA implementation.
///
/// # Preconditions
/// - `altitude_m` should be in [0, 11000] for accuracy (troposphere)
#[must_use]
pub fn air_from_altitude(altitude_m: f64) -> AirProperties {
    let atm = tools_core::atmosphere::atmosphere_at_altitude(altitude_m);
    AirProperties {
        density: atm.density,
        viscosity: atm.viscosity,
        temperature: atm.temperature,
        pressure: atm.pressure,
    }
}

// NOTE: cross product uses Vector3::cross() from tools-core (DRY).

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AirProperties {
    #[new]
    #[pyo3(signature = (density=1.225, viscosity=1.81e-5, temperature=288.15, pressure=101325.0))]
    fn py_new(
        density: f64,
        viscosity: f64,
        temperature: f64,
        pressure: f64,
    ) -> pyo3::PyResult<Self> {
        let air = Self {
            density,
            viscosity,
            temperature,
            pressure,
        };
        air.validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(air)
    }

    #[staticmethod]
    fn from_altitude(altitude_m: f64) -> Self {
        air_from_altitude(altitude_m)
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AeroBallProperties {
    #[new]
    #[pyo3(signature = (mass=0.04593, radius=0.02135, drag_coefficient=0.25, spin_decay_rate=0.1))]
    fn py_new(
        mass: f64,
        radius: f64,
        drag_coefficient: f64,
        spin_decay_rate: f64,
    ) -> pyo3::PyResult<Self> {
        let area = std::f64::consts::PI * radius * radius;
        let ball = Self {
            mass,
            radius,
            area,
            drag_coefficient,
            spin_decay_rate,
        };
        ball.validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(ball)
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AeroForces {
    /// Get drag as [x, y, z] list.
    #[getter]
    fn drag_vec(&self) -> [f64; 3] {
        [self.drag.x, self.drag.y, self.drag.z]
    }

    /// Get lift as [x, y, z] list.
    #[getter]
    fn lift_vec(&self) -> [f64; 3] {
        [self.lift.x, self.lift.y, self.lift.z]
    }

    /// Get magnus as [x, y, z] list.
    #[getter]
    fn magnus_vec(&self) -> [f64; 3] {
        [self.magnus.x, self.magnus.y, self.magnus.z]
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests (TDD — written alongside implementation)
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ball() -> AeroBallProperties {
        AeroBallProperties::default()
    }

    fn default_air() -> AirProperties {
        AirProperties::default()
    }

    #[test]
    fn test_air_properties_reject_invalid_public_values() {
        assert!(AirProperties {
            density: f64::NAN,
            ..AirProperties::default()
        }
        .validate()
        .is_err());
        assert!(AirProperties {
            viscosity: 0.0,
            ..AirProperties::default()
        }
        .validate()
        .is_err());
        assert!(AirProperties {
            temperature: -1.0,
            ..AirProperties::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_aero_ball_properties_reject_invalid_public_values() {
        assert!(AeroBallProperties {
            mass: 0.0,
            ..AeroBallProperties::default()
        }
        .validate()
        .is_err());
        assert!(AeroBallProperties {
            radius: -0.1,
            area: 0.01,
            ..AeroBallProperties::default()
        }
        .validate()
        .is_err());
        assert!(AeroBallProperties {
            spin_decay_rate: f64::INFINITY,
            ..AeroBallProperties::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    #[should_panic(expected = "invalid aerodynamic ball properties")]
    fn test_compute_aero_forces_rejects_invalid_ball_in_release_mode() {
        let velocity = Vector3::new(10.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 0.0, 100.0);
        let ball = AeroBallProperties {
            mass: 0.0,
            ..AeroBallProperties::default()
        };
        let _ = compute_aero_forces(&velocity, &spin, &ball, &AirProperties::default());
    }

    // ── Drag Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_drag_opposes_velocity() {
        let v = Vector3::new(50.0, 0.0, 0.0);
        let ball = default_ball();
        let air = default_air();
        let drag = compute_drag(&v, &ball, &air);

        // Drag must oppose velocity (negative x)
        assert!(drag.x < 0.0, "Drag must oppose velocity: got {}", drag.x);
        assert!(drag.y.abs() < 1e-10, "No lateral drag for x-only velocity");
        assert!(drag.z.abs() < 1e-10, "No vertical drag for x-only velocity");
    }

    #[test]
    fn test_drag_zero_velocity() {
        let v = Vector3::new(0.0, 0.0, 0.0);
        let ball = default_ball();
        let air = default_air();
        let drag = compute_drag(&v, &ball, &air);

        assert!(drag.x.abs() < 1e-10);
        assert!(drag.y.abs() < 1e-10);
        assert!(drag.z.abs() < 1e-10);
    }

    #[test]
    fn test_drag_scales_with_speed_squared() {
        let ball = default_ball();
        let air = default_air();

        // Both speeds must be fully turbulent (Re > 2e5)
        // Re = rho * v * d / mu; for Re=2e5 → v ≈ 69 m/s
        // Use 80 and 160 to stay well above transition
        let v1 = Vector3::new(80.0, 0.0, 0.0);
        let v2 = Vector3::new(160.0, 0.0, 0.0);

        let drag1 = compute_drag(&v1, &ball, &air);
        let drag2 = compute_drag(&v2, &ball, &air);

        // Both fully turbulent → same Cd → ratio should be (160/80)^2 = 4
        let ratio = drag2.x.abs() / drag1.x.abs();
        assert!(
            (ratio - 4.15).abs() < 0.1,
            "Drag ratio should include v^2 plus smooth Cd recovery, got {}",
            ratio
        );
    }

    // ── Lift Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_lift_with_backspin() {
        // Backspin (positive y-spin) with forward velocity (x direction)
        // Should produce upward lift (positive z)
        let v = Vector3::new(50.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 300.0, 0.0);
        let ball = default_ball();
        let air = default_air();

        let lift = compute_lift(&v, &spin, &ball, &air);

        // Lift should have a significant z-component
        // spin_axis × velocity = (0,1,0) × (50,0,0) = (0,0,-50) — negative z
        // Actually: (0,1,0) × (50,0,0) = (1*0 - 0*0, 0*50 - 0*0, 0*0 - 1*50) = (0, 0, -50)
        // The magnitude matters, sign depends on cross product direction
        assert!(
            lift.magnitude() > 0.01,
            "Backspin should produce non-zero lift"
        );
    }

    #[test]
    fn test_lift_zero_spin() {
        let v = Vector3::new(50.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 0.0, 0.0);
        let ball = default_ball();
        let air = default_air();

        let lift = compute_lift(&v, &spin, &ball, &air);
        assert!(lift.magnitude() < 1e-6, "No spin should mean no lift");
    }

    // ── Magnus Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_magnus_with_sidespin() {
        // Sidespin (z-axis) with forward velocity (x direction)
        // Should produce lateral force (y direction)
        let v = Vector3::new(50.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 0.0, 200.0);
        let ball = default_ball();
        let air = default_air();

        let magnus = compute_magnus(&v, &spin, &ball, &air);

        // ω × v = (0,0,200) × (50,0,0) = (0*0 - 200*0, 200*50 - 0*0, 0*0 - 0*50)
        //       = (0, 10000, 0)
        assert!(
            magnus.y.abs() > 0.01,
            "Sidespin should produce lateral Magnus force"
        );
    }

    #[test]
    fn test_magnus_zero_speed() {
        let v = Vector3::new(0.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 0.0, 200.0);
        let ball = default_ball();
        let air = default_air();

        let magnus = compute_magnus(&v, &spin, &ball, &air);
        assert!(magnus.magnitude() < 1e-10);
    }

    #[test]
    fn test_magnus_zero_spin() {
        let v = Vector3::new(50.0, 0.0, 0.0);
        let spin = Vector3::new(0.0, 0.0, 0.0);
        let ball = default_ball();
        let air = default_air();

        let magnus = compute_magnus(&v, &spin, &ball, &air);
        assert!(magnus.magnitude() < 1e-10);
    }

    // ── Combined Forces Tests ────────────────────────────────────────────

    #[test]
    fn test_compute_aero_forces_all_finite() {
        let v = Vector3::new(70.0, 5.0, 20.0);
        let spin = Vector3::new(10.0, 300.0, 50.0);
        let ball = default_ball();
        let air = default_air();

        let forces = compute_aero_forces(&v, &spin, &ball, &air);

        assert!(forces.drag.x.is_finite());
        assert!(forces.drag.y.is_finite());
        assert!(forces.drag.z.is_finite());
        assert!(forces.lift.x.is_finite());
        assert!(forces.lift.y.is_finite());
        assert!(forces.lift.z.is_finite());
        assert!(forces.magnus.x.is_finite());
        assert!(forces.magnus.y.is_finite());
        assert!(forces.magnus.z.is_finite());
    }

    #[test]
    fn test_drag_always_opposes_velocity_diagonal() {
        let v = Vector3::new(30.0, 20.0, 10.0);
        let ball = default_ball();
        let air = default_air();

        let drag = compute_drag(&v, &ball, &air);

        // Dot product of drag and velocity should be negative (opposing)
        let dot = drag.x * v.x + drag.y * v.y + drag.z * v.z;
        assert!(dot < 0.0, "Drag must oppose velocity: dot = {}", dot);
    }

    // ── Coefficient Tests ────────────────────────────────────────────────

    #[test]
    fn test_drag_coefficient_laminar() {
        let ball = default_ball();
        let air = default_air();
        // Very low speed → laminar → Cd = 0.5
        let cd = compute_drag_coefficient(1.0, &ball, &air);
        assert!(
            (cd - 0.500_001_242_489_331_1).abs() < 1e-12,
            "Laminar Cd should match the Python dimpled-sphere curve, got {}",
            cd
        );
    }

    #[test]
    fn test_drag_coefficient_matches_python_drag_crisis_curve() {
        let ball = default_ball();
        let air = default_air();
        let cd = compute_drag_coefficient(100.0, &ball, &air);
        assert!(
            (cd - 0.257_787_143_359_211_3).abs() < 1e-12,
            "Rust Cd should match Python cd_dimpled_sphere at golf launch Re, got {}",
            cd
        );
    }

    #[test]
    fn test_cd_dimpled_sphere_python_parity_anchors() {
        let cases = [
            (1.0e3, 0.500_000_161_176_627_4),
            (1.0e4, 0.500_007_263_990_912_3),
            (4.0e4, 0.488_763_716_427_621_5),
            (7.0e4, 0.375_580_365_822_265_2),
            (7.0e5, 0.275_000_931_659_821_1),
            (1.0e7, 0.299_706_757_892_032_6),
        ];

        for (reynolds, expected) in cases {
            let actual = cd_dimpled_sphere(reynolds, 0.25).unwrap();
            assert!(
                (actual - expected).abs() < 1e-12,
                "Cd parity failed for Re={reynolds}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_cd_dimpled_sphere_rejects_invalid_inputs() {
        assert!(cd_dimpled_sphere(0.0, 0.25).is_err());
        assert!(cd_dimpled_sphere(f64::NAN, 0.25).is_err());
        assert!(cd_dimpled_sphere(999.0, 0.25).is_err());
        assert!(cd_dimpled_sphere(1.0e8, 0.25).is_err());
        assert!(cd_dimpled_sphere(1.0e5, f64::INFINITY).is_err());
    }

    #[test]
    fn test_lift_coefficient_increases_with_spin() {
        let cl_low = compute_lift_coefficient(0.01);
        let cl_high = compute_lift_coefficient(0.5);
        assert!(cl_high > cl_low, "Higher spin ratio → higher Cl");
    }

    #[test]
    fn test_lift_coefficient_saturates() {
        let cl_very_high = compute_lift_coefficient(10.0);
        assert!(
            (cl_very_high - 0.4).abs() < 0.01,
            "Cl should saturate near 0.4, got {}",
            cl_very_high
        );
    }

    #[test]
    fn test_magnus_coefficient_capped() {
        let cm = compute_magnus_coefficient(1.0);
        assert!(
            (cm - 0.2).abs() < 1e-10,
            "Magnus coeff caps at 0.4*0.5=0.2, got {}",
            cm
        );
    }

    // ── Spin Decay Tests ─────────────────────────────────────────────────

    #[test]
    fn test_spin_decay_reduces_magnitude() {
        let spin = Vector3::new(0.0, 300.0, 0.0);
        let decayed = compute_spin_decay(&spin, 0.01, 0.1);
        assert!(
            decayed.magnitude() < spin.magnitude(),
            "Spin should decrease after decay"
        );
    }

    #[test]
    fn test_spin_decay_preserves_direction() {
        let spin = Vector3::new(10.0, 300.0, 50.0);
        let decayed = compute_spin_decay(&spin, 0.01, 0.1);

        // Direction should be preserved (ratio of components same)
        let ratio_x = decayed.x / spin.x;
        let ratio_y = decayed.y / spin.y;
        let ratio_z = decayed.z / spin.z;

        assert!((ratio_x - ratio_y).abs() < 1e-10, "Decay should be uniform");
        assert!((ratio_y - ratio_z).abs() < 1e-10, "Decay should be uniform");
    }

    // ── Air from Altitude ────────────────────────────────────────────────

    #[test]
    fn test_air_from_altitude_sea_level() {
        let air = air_from_altitude(0.0);
        assert!(
            (air.density - 1.225).abs() < 0.01,
            "Sea level density should be ~1.225"
        );
        assert!(
            (air.temperature - 288.15).abs() < 0.1,
            "Sea level temp should be ~288.15 K"
        );
    }

    #[test]
    fn test_air_from_altitude_lower_density_at_height() {
        let sea = air_from_altitude(0.0);
        let high = air_from_altitude(2000.0);
        assert!(
            high.density < sea.density,
            "Higher altitude → lower density"
        );
        assert!(
            high.temperature < sea.temperature,
            "Higher altitude → lower temperature"
        );
    }

    // ── Edge case tests (TDD) ───────────────────────────────────────────

    /// Zero velocity should produce zero aerodynamic forces.
    #[test]
    fn test_aero_forces_zero_velocity() {
        let ball = default_ball();
        let air = AirProperties::default();
        let velocity = Vector3::zero();
        let spin = Vector3::new(0.0, 0.0, 300.0);

        let forces = compute_aero_forces(&velocity, &spin, &ball, &air);
        assert!(forces.drag.magnitude() < 1e-10, "Zero velocity → zero drag");
    }

    /// Tropopause altitude (11000m) should give valid but thin air.
    #[test]
    fn test_air_from_altitude_tropopause() {
        let air = air_from_altitude(11000.0);
        assert!(air.density > 0.0, "Density must be positive at 11km");
        assert!(
            air.density < 0.5,
            "Density at 11km should be < 0.5 kg/m³, got {}",
            air.density
        );
        assert!(
            air.temperature > 200.0 && air.temperature < 230.0,
            "Tropopause temp ~216.65 K, got {}",
            air.temperature
        );
    }

    /// Zero spin should produce zero Magnus/lift forces.
    #[test]
    fn test_aero_forces_zero_spin() {
        let ball = default_ball();
        let air = AirProperties::default();
        let velocity = Vector3::new(40.0, 0.0, 0.0);
        let spin = Vector3::zero();

        let forces = compute_aero_forces(&velocity, &spin, &ball, &air);
        assert!(forces.magnus.magnitude() < 1e-10, "Zero spin → zero Magnus");
    }
}
