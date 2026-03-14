#!/usr/bin/env python3
"""Fitting loss coefficients (K-factors) for pressure drop calculations.

Resistance coefficients for valves, fittings, and pipe components based on
established engineering references.

IMPORTANT NOTE ON K-FACTORS:
    The K-factor (resistance coefficient) is used directly in:
        ΔP = K × (ρV²/2)

    K-factors are dimensionless and typically range from:
        - Elbows: 0.2 - 1.5
        - Tees: 0.2 - 2.0
        - Valves (open): 0.1 - 10.0
        - Entrances: 0.04 - 1.0
        - Exits: 1.0

    DO NOT confuse with L/D (equivalent length in diameters), which requires
    multiplication by friction factor: K = f × (L/D)

References:
    - Crane Technical Paper No. 410 (TP-410), Flow of Fluids Through Valves, Fittings, and Pipe
    - Idelchik, I.E. (2007): Handbook of Hydraulic Resistance, 4th Edition
    - Hooper, W.B. (1981): "The Two-K Method", Chemical Engineering
    - Darby, R. (1999): "Correlate pressure drops through fittings", Chemical Engineering
    - Miller, D.S. (1990): Internal Flow Systems, 2nd Edition
"""

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARD FITTING K-FACTORS (TRUE K VALUES)
# ============================================================================

# TRUE K-factors from Crane TP-410 and Idelchik
# These are dimensionless resistance coefficients for direct use in: ΔP = K × (ρV²/2)
# Values are for fully turbulent flow in typical pipe sizes (2-6 inch)
FITTING_K_FACTORS: dict[str, float] = {
    # ===== ELBOWS =====
    # Standard radius elbows (r/D ≈ 1.0) - K values from Crane TP-410 Table A-29
    "90_elbow_std": 0.75,  # 90° standard elbow, threaded (K ≈ 0.7-0.9)
    "90_elbow_long": 0.45,  # 90° long radius elbow, r/D=1.5 (K ≈ 0.4-0.5)
    "90_elbow_welded_std": 0.60,  # 90° standard elbow, welded (K ≈ 0.5-0.7)
    "90_elbow_welded_long": 0.40,  # 90° long radius elbow, welded (K ≈ 0.35-0.45)
    "45_elbow_std": 0.35,  # 45° standard elbow, threaded (K ≈ 0.3-0.4)
    "45_elbow_long": 0.20,  # 45° long radius elbow, threaded (K ≈ 0.18-0.25)
    "45_elbow_welded": 0.30,  # 45° elbow, welded (K ≈ 0.25-0.35)
    # Mitered elbows (without vanes) - higher losses
    "90_miter_1weld": 1.30,  # 90° miter, 1 weld (α = 90°)
    "90_miter_2weld": 0.60,  # 90° miter, 2 welds (α = 45°)
    "90_miter_3weld": 0.45,  # 90° miter, 3 welds (α = 30°)
    "45_miter": 0.35,  # 45° miter bend
    # ===== TEES =====
    # K-factors depend on flow direction and velocity ratios
    "tee_through_branch": 1.00,  # Flow through branch (90° turn)
    "tee_through_run": 0.30,  # Flow straight through run
    "tee_branching_flow": 1.00,  # Branching flow (dividing)
    "tee_combining_flow": 1.50,  # Combining flow (merging, higher K)
    # ===== VALVES =====
    # Gate valves - low resistance when fully open
    "gate_valve_open": 0.15,  # Fully open (K ≈ 0.1-0.2)
    "gate_valve_3/4_open": 0.85,  # 3/4 open
    "gate_valve_1/2_open": 4.50,  # 1/2 open (significant restriction)
    "gate_valve_1/4_open": 22.0,  # 1/4 open (severe restriction)
    # Globe valves - high resistance even when open
    "globe_valve_open": 6.50,  # Fully open (K ≈ 6-10)
    "globe_valve_angle": 3.00,  # Angle globe valve, open (K ≈ 2-4)
    # Ball valves - very low resistance when open
    "ball_valve_open": 0.05,  # Fully open (K ≈ 0.04-0.1)
    "ball_valve_1/3_open": 5.50,  # 1/3 open
    "ball_valve_2/3_open": 0.30,  # 2/3 open
    # Butterfly valves - moderate resistance
    "butterfly_valve_open": 0.35,  # Fully open (θ = 0°)
    "butterfly_valve_10deg": 0.50,  # 10° from fully open
    "butterfly_valve_20deg": 1.50,  # 20° from fully open
    "butterfly_valve_30deg": 3.50,  # 30° from fully open
    "butterfly_valve_40deg": 9.00,  # 40° from fully open
    "butterfly_valve_50deg": 25.0,  # 50° from fully open
    "butterfly_valve_60deg": 70.0,  # 60° from fully open (nearly closed)
    # Check valves - resistance depends on type
    "check_valve_swing": 2.00,  # Swing check valve (K ≈ 1.5-2.5)
    "check_valve_lift": 12.0,  # Lift check valve (K ≈ 10-15)
    "check_valve_ball": 4.50,  # Ball check valve
    "check_valve_tilting": 1.00,  # Tilting disk check valve (low loss)
    # Plug valves
    "plug_valve_straight": 0.40,  # Straight-through plug valve
    "plug_valve_3way": 0.90,  # 3-way plug valve
    "plug_valve_branch": 1.50,  # Branch flow
    # Diaphragm valve
    "diaphragm_valve_open": 2.30,  # Fully open (K ≈ 2-3)
    # ===== REDUCERS/EXPANDERS =====
    # K based on velocity in smaller diameter
    "reducer_gradual": 0.04,  # Gradual reducer (θ ≈ 20°, K ≈ 0.03-0.05)
    "reducer_sudden": 0.50,  # Sudden reducer (contraction, K ≈ 0.4-0.5)
    "expander_gradual": 0.30,  # Gradual expander (θ ≈ 20°, K ≈ 0.2-0.4)
    "expander_sudden": 1.00,  # Sudden expander (Borda-Carnot, K = 1.0)
    # ===== ENTRANCES/EXITS =====
    # Entrance K-factors (based on velocity in pipe)
    "entrance_sharp": 0.50,  # Sharp-edged entrance
    "entrance_rounded": 0.04,  # Well-rounded entrance (r/D = 0.15)
    "entrance_bellmouth": 0.03,  # Bellmouth entrance (r/D ≥ 0.25)
    "entrance_inward": 0.78,  # Inward projecting (Borda) entrance
    "entrance_chamfered": 0.25,  # Chamfered entrance
    # Exit K-factors (always K = 1.0 for all geometries)
    "exit_sharp": 1.00,  # Sharp exit to reservoir
    "exit_rounded": 1.00,  # Rounded exit (still K = 1.0)
    "exit_submerged": 1.00,  # Submerged exit
    # ===== BENDS =====
    "bend_close_return": 2.20,  # Close return bend (180°)
    "bend_return_flanged": 1.50,  # Return bend, flanged
    "bend_90_r1d": 0.35,  # 90° bend, r/D = 1
    "bend_90_r2d": 0.19,  # 90° bend, r/D = 2
    "bend_90_r4d": 0.17,  # 90° bend, r/D = 4
    "bend_90_r6d": 0.22,  # 90° bend, r/D = 6 (increasing due to length)
}


# ============================================================================
# EQUIVALENT LENGTH IN PIPE DIAMETERS (L/D)
# ============================================================================

# L/D values from Crane TP-410 - Alternative representation
# To convert L/D to K-factor: K = f_T × (L/D) where f_T ≈ 0.015-0.020 for turbulent flow
# Note: L/D method is less accurate than K-factors for variable flow conditions
FITTING_EQUIVALENT_LENGTH: dict[str, float] = {
    # Format: fitting_type: L/D (equivalent length in diameters)
    # Elbows
    "90_elbow_std": 30,  # K ≈ 0.02 × 30 = 0.6
    "90_elbow_long": 20,  # K ≈ 0.02 × 20 = 0.4
    "45_elbow_std": 16,  # K ≈ 0.02 × 16 = 0.32
    # Tees
    "tee_through_branch": 60,  # K ≈ 0.02 × 60 = 1.2
    "tee_through_run": 20,  # K ≈ 0.02 × 20 = 0.4
    # Valves
    "gate_valve_open": 8,  # K ≈ 0.02 × 8 = 0.16
    "globe_valve_open": 340,  # K ≈ 0.02 × 340 = 6.8
    "ball_valve_open": 3,  # K ≈ 0.02 × 3 = 0.06
    "check_valve_swing": 100,  # K ≈ 0.02 × 100 = 2.0
    # Entrances/Exits - use K-factors instead (not L/D dependent)
    "entrance_sharp": 25,  # K = 0.5 (use K-factor directly)
    "exit_sharp": 50,  # K = 1.0 (use K-factor directly)
}


def get_fitting_k_factor(fitting_type: str) -> float:
    """Get resistance coefficient (K-factor) for a fitting.

    The pressure drop is calculated as:
        ΔP = K × (ρV²/2)

    where:
        K = resistance coefficient (dimensionless)
        ρ = fluid density (kg/m³)
        V = flow velocity (m/s)

    Args:
        fitting_type: Fitting type identifier (see FITTING_K_FACTORS keys)

    Returns:
        K-factor (dimensionless resistance coefficient)

    Raises:
        ValueError: If fitting type not found

    References:
        Crane TP-410, Table A-29: Resistance Coefficients

    Example:
        >>> k = get_fitting_k_factor('90_elbow_std')
        >>> print(k)
        30
    """
    if fitting_type not in FITTING_K_FACTORS:
        available = ", ".join(sorted(FITTING_K_FACTORS.keys()))
        raise ValueError(
            f"Fitting type '{fitting_type}' not found.\nAvailable types: {available}"
        )

    return FITTING_K_FACTORS[fitting_type]


def get_multiple_fittings_k(fittings: dict[str, int]) -> float:
    """Calculate total K-factor for multiple fittings.

    Args:
        fittings: Dictionary mapping fitting_type to quantity

    Returns:
        Total K-factor for all fittings

    Example:
        >>> fittings = {
        ...     '90_elbow_std': 4,
        ...     'gate_valve_open': 2,
        ...     'tee_through_run': 1
        ... }
        >>> total_k = get_multiple_fittings_k(fittings)
        >>> print(total_k)  # 4*30 + 2*8 + 1*20 = 156
        156
    """
    total_k = 0.0
    for fitting_type, quantity in fittings.items():
        k = get_fitting_k_factor(fitting_type)
        total_k += k * quantity
        logger.debug(f"Added {quantity} × {fitting_type}: K = {k * quantity}")

    return total_k


def k_to_equivalent_length(k_factor: float, friction_factor: float) -> float:
    """Convert K-factor to equivalent length in pipe diameters.

    Relationship: K = f × (L/D)
    Therefore: L/D = K/f

    Args:
        k_factor: Resistance coefficient
        friction_factor: Darcy friction factor

    Returns:
        Equivalent length in pipe diameters (L/D)

    Note:
        This requires knowing the friction factor, which varies with Reynolds number.
        Typically, use f ≈ 0.015-0.025 for turbulent flow in commercial pipe.

    Example:
        >>> # 90° elbow with K = 30, assuming f = 0.02
        >>> L_over_D = k_to_equivalent_length(30, 0.02)
        >>> print(f"Equivalent to {L_over_D:.0f} diameters of straight pipe")
        Equivalent to 1500 diameters of straight pipe
    """
    if friction_factor <= 0:
        raise ValueError("Friction factor must be positive")

    return k_factor / friction_factor


def equivalent_length_to_k(L_over_D: float, friction_factor: float) -> float:
    """Convert equivalent length to K-factor.

    Args:
        L_over_D: Equivalent length in pipe diameters
        friction_factor: Darcy friction factor

    Returns:
        Resistance coefficient (K-factor)

    Example:
        >>> # 30 diameters equivalent length with f = 0.02
        >>> k = equivalent_length_to_k(30, 0.02)
        >>> print(f"K-factor = {k:.1f}")
        K-factor = 0.6
    """
    if friction_factor <= 0:
        raise ValueError("Friction factor must be positive")

    return L_over_D * friction_factor


# ============================================================================
# TWO-K METHOD (Hooper 1981, 1988)
# ============================================================================

# More accurate method accounting for Reynolds number and pipe size effects
# K = K1/Re + K∞ × (1 + Kd/ID^0.3)
#
# where:
#   K1 = coefficient for laminar flow contribution
#   K∞ = base turbulent flow K-factor for large pipes
#   Kd = diameter correction factor (accounts for smaller pipe sizes having higher K)
#   ID = internal diameter in inches
#   Re = Reynolds number
#
# Reference: Hooper, W.B. (1981): "The Two-K Method", Chemical Engineering, Nov 1981

TWO_K_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    # Format: fitting_type: (K1, K∞, Kd)
    # Values from Hooper (1981) and Darby (2001)
    #
    # ===== ELBOWS =====
    "90_elbow_std_2k": (800, 0.25, 4.0),  # Standard 90° elbow, threaded
    "90_elbow_long_2k": (800, 0.20, 4.0),  # Long radius 90° elbow, r/D=1.5
    "90_elbow_welded_2k": (800, 0.18, 4.0),  # 90° elbow, welded/flanged
    "45_elbow_std_2k": (500, 0.15, 4.0),  # 45° elbow
    "90_miter_1weld_2k": (1000, 0.60, 4.0),  # 90° miter, single weld
    "90_miter_2weld_2k": (800, 0.35, 4.0),  # 90° miter, two welds
    #
    # ===== TEES =====
    "tee_through_branch_2k": (500, 0.70, 4.0),  # Flow through branch (90° turn)
    "tee_through_run_2k": (200, 0.10, 4.0),  # Flow straight through run
    "tee_branching_2k": (500, 0.70, 4.0),  # Branching/dividing flow
    "tee_combining_2k": (800, 1.00, 4.0),  # Combining/merging flow
    #
    # ===== VALVES =====
    "gate_valve_open_2k": (300, 0.10, 4.0),  # Gate valve, fully open
    "gate_valve_3/4_open_2k": (500, 0.60, 4.0),  # Gate valve, 3/4 open
    "gate_valve_1/2_open_2k": (1000, 3.50, 4.0),  # Gate valve, 1/2 open
    "globe_valve_open_2k": (1500, 4.00, 4.0),  # Globe valve, fully open
    "globe_valve_angle_2k": (1000, 2.00, 4.0),  # Angle globe valve
    "ball_valve_open_2k": (300, 0.017, 3.5),  # Ball valve, fully open
    "butterfly_valve_2k": (800, 0.25, 4.0),  # Butterfly valve, fully open
    "check_valve_swing_2k": (1500, 1.50, 4.0),  # Swing check valve
    "check_valve_lift_2k": (2000, 10.0, 4.0),  # Lift check valve
    #
    # ===== ENTRANCES/EXITS =====
    "entrance_sharp_2k": (160, 0.50, 0.0),  # Sharp entrance (K independent of size)
    "entrance_inward_2k": (160, 0.78, 0.0),  # Inward projecting entrance
    "exit_2k": (0, 1.00, 0.0),  # Exit to tank (always K=1.0)
    #
    # ===== REDUCERS/EXPANDERS =====
    "reducer_sudden_2k": (800, 0.50, 4.0),  # Sudden contraction
    "expander_sudden_2k": (0, 1.00, 0.0),  # Sudden expansion (Borda-Carnot)
}


def calculate_two_k_factor(
    fitting_type: str, reynolds_number: float, diameter_inches: float
) -> float:
    """Calculate K-factor using Two-K method (Hooper).

    More accurate than constant K-factors, especially for:
    - Laminar and transitional flow (Re < 10,000)
    - Small diameter pipes (< 2 inches)

    K = K1/Re + K∞ × (1 + Kd/ID^0.3)

    Args:
        fitting_type: Fitting type with '_2k' suffix
        reynolds_number: Reynolds number
        diameter_inches: Internal diameter in inches

    Returns:
        K-factor accounting for Re and diameter effects

    References:
        Hooper, W.B. (1981): "The Two-K Method for Predicting Pressure Loss"
        Hooper, W.B. (1988): "Calculate Head Loss Caused by Change in Pipe Size"

    Example:
        >>> k = calculate_two_k_factor('90_elbow_std_2k', 50000, 4.0)
        >>> print(f"K = {k:.2f}")
    """
    if fitting_type not in TWO_K_COEFFICIENTS:
        raise ValueError(f"Fitting '{fitting_type}' not in Two-K database")

    K1, K_inf, Kd = TWO_K_COEFFICIENTS[fitting_type]

    # Two-K correlation
    k_laminar = K1 / reynolds_number
    k_turbulent = K_inf * (1.0 + Kd / (diameter_inches**0.3))

    total_k = k_laminar + k_turbulent

    logger.debug(
        f"{fitting_type}: K_lam={k_laminar:.3f}, K_turb={k_turbulent:.3f}, Total={total_k:.3f}"
    )

    return float(total_k)


# ============================================================================
# DARBY 3-K METHOD
# ============================================================================

# Even more accurate: K = K1/Re + K∞(1 + Kd/DN^0.3)
# where DN = nominal diameter in mm

DARBY_3K_COEFFICIENTS = {
    # Format: (K1, K∞, Kd)
    "90_elbow_threaded": (800, 0.40, 4),
    "90_elbow_flanged": (800, 0.25, 4),
    "gate_valve": (300, 0.10, 4),
    "globe_valve": (1500, 4.0, 4),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def list_available_fittings() -> dict[str, float]:
    """List all available fittings with their K-factors.

    Returns:
        Dictionary of fitting types and K-factors
    """
    return FITTING_K_FACTORS.copy()


def print_fitting_database() -> None:
    """Print formatted table of all fittings."""
    logger.info("\n" + "=" * 80)
    logger.info("FITTING RESISTANCE COEFFICIENTS (K-FACTORS)")
    logger.info("Source: Crane TP-410, Idelchik Handbook")
    logger.info("=" * 80)

    categories = {
        "ELBOWS": ["elbow", "miter", "bend"],
        "TEES": ["tee"],
        "VALVES": ["valve"],
        "REDUCERS/EXPANDERS": ["reducer", "expander"],
        "ENTRANCES/EXITS": ["entrance", "exit"],
    }

    for category, keywords in categories.items():
        logger.info(f"\n{category}:")
        logger.info("-" * 80)
        for fitting_type, k_factor in sorted(FITTING_K_FACTORS.items()):
            if any(kw in fitting_type for kw in keywords):
                # Format the name nicely
                name = fitting_type.replace("_", " ").title()
                logger.info(f"  {name:50s} K = {k_factor:6.0f}")


def calculate_fitting_pressure_drop(
    k_factor: float, density: float, velocity: float
) -> float:
    """Calculate pressure drop across a fitting.

    ΔP = K × (ρV²/2)

    Args:
        k_factor: Resistance coefficient
        density: Fluid density (kg/m³)
        velocity: Flow velocity (m/s)

    Returns:
        Pressure drop (Pa)

    Example:
        >>> # 90° elbow, water at 5 m/s
        >>> k = get_fitting_k_factor('90_elbow_std')
        >>> dp = calculate_fitting_pressure_drop(k, 1000, 5)
        >>> print(f"ΔP = {dp:.0f} Pa = {dp/1e5:.3f} bar")
    """
    assert k_factor is not None, "k_factor must be provided"
    velocity_pressure = 0.5 * density * velocity**2
    return k_factor * velocity_pressure


if __name__ == "__main__":
    # Demonstrate usage
    logging.basicConfig(level=logging.INFO)

    print_fitting_database()

    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE CALCULATIONS")
    logger.info("=" * 80)

    # Example 1: Single fitting
    logger.info("\nExample 1: Pressure drop across 90° elbow")
    logger.info("-" * 80)
    k = get_fitting_k_factor("90_elbow_std")
    rho = 1.2  # kg/m³ (air)
    v = 15  # m/s
    dp = calculate_fitting_pressure_drop(k, rho, v)
    logger.info("  Fitting: 90° standard elbow")
    logger.info(f"  K-factor: {k}")
    logger.info(f"  Density: {rho} kg/m³")
    logger.info(f"  Velocity: {v} m/s")
    logger.info(f"  Pressure drop: {dp:.1f} Pa = {dp / 100:.2f} mbar")

    # Example 2: Multiple fittings
    logger.info("\nExample 2: Total K-factor for piping system")
    logger.info("-" * 80)
    fittings = {
        "90_elbow_std": 6,
        "45_elbow_std": 2,
        "gate_valve_open": 3,
        "tee_through_run": 2,
    }
    total_k = get_multiple_fittings_k(fittings)
    logger.info("  System components:")
    for fitting, qty in fittings.items():
        logger.info(f"    - {qty} × {fitting}")
    logger.info(f"  Total K-factor: {total_k}")

    # Example 3: Two-K method
    logger.info("\nExample 3: Two-K method for small pipe")
    logger.info("-" * 80)
    re = 10000
    d_inch = 1.0
    k_std = get_fitting_k_factor("90_elbow_std")
    k_2k = calculate_two_k_factor("90_elbow_std_2k", re, d_inch)
    logger.info(f'  90° elbow in 1" pipe at Re = {re}')
    logger.info(f"  Standard K-factor: {k_std}")
    logger.info(f"  Two-K method: {k_2k:.2f}")
    logger.info(f"  Difference: {((k_2k / k_std - 1) * 100):.1f}%")
