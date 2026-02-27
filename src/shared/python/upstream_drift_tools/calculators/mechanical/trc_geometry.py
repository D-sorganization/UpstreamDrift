"""
TRC Geometry Engine
===================

Core calculation engine for the Thermal Reactor Controller (TRC) vessel geometry.
Handles volume, mass, surface area, and residence time calculations.

Performance optimizations:
- Pre-computed constants (PI, PI/3, unit conversion factors)
- Reduced redundant calculations in layer loops
- Early validation for invalid inputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Pre-computed constants for performance
_PI = math.pi
_PI_OVER_3 = math.pi / 3.0
_CUBIC_INCHES_TO_CUBIC_FEET = 1.0 / 1728.0  # 12^3
_SQUARE_INCHES_TO_SQUARE_FEET = 1.0 / 144.0  # 12^2


@dataclass
class LayerConfig:
    """Configuration for a single vessel layer"""

    name: str
    thickness: float  # inches
    density: float  # lb/ft^3
    color: str
    visible: bool = True
    transparency: float = 0.3
    top_section_name: str = ""

    def __post_init__(self) -> None:
        if not self.top_section_name:
            self.top_section_name = f"{self.name} Top"
        # Ensure numeric types
        self.thickness = float(self.thickness)
        self.density = float(self.density)
        self.transparency = float(self.transparency)


@dataclass
class VesselDimensions:
    """Dimensions of the vessel geometry"""

    cylinder_height: float  # inches
    cylinder_diameter: float  # inches
    cone_height: float  # inches
    cone_bottom_diameter: float  # inches
    cone_interior_hole: float  # inches
    top_refractory_thickness: float  # inches

    # Feature flags affecting calculation
    display_lid: bool = True
    display_cylinder: bool = True
    display_cone: bool = True


@dataclass
class LayerResult:
    """Calculation results for a single layer"""

    name: str
    volume_ft3: float
    mass_lb: float
    density: float
    outer_surface_area_ft2: float = 0.0


@dataclass
class VesselGeometryResult:
    """Total vessel geometry results"""

    layers: list[LayerResult] = field(default_factory=list)
    total_volume_ft3: float = 0.0
    total_mass_lb: float = 0.0
    outside_surface_area_ft2: float = 0.0
    interior_volume_ft3: float = 0.0  # Void volume for residence time
    void_radius_inches: float = 0.0
    void_diameter_inches: float = 0.0
    interior_height_inches: float = 0.0


def _calculate_layer_cylinder_volume(
    current_radius_sq: float,
    inner_radius_sq: float,
    interior_height: float,
) -> float:
    """Cylinder side-wall annular volume: π(R² − r²)h."""
    return _PI * (current_radius_sq - inner_radius_sq) * interior_height


def _calculate_layer_cone_volume(
    current_radius: float,
    current_radius_sq: float,
    inner_radius: float,
    inner_radius_sq: float,
    cone_bottom_radius: float,
    radius_offset: float,
    layer_thickness: float,
    interior_hole_radius: float,
    cone_height_factor: float,
) -> float:
    """Differential annular truncated-cone volume for one layer."""
    # Outer frustum at bottom for this layer
    outer_bottom_r = max(cone_bottom_radius - radius_offset, interior_hole_radius)
    outer_bottom_sq = outer_bottom_r * outer_bottom_r

    # Inner frustum at bottom for this layer
    inner_bottom_r = max(outer_bottom_r - layer_thickness, interior_hole_radius)
    inner_bottom_sq = inner_bottom_r * inner_bottom_r

    # V = (π/3)h(R² + Rr + r²)  for each frustum, take difference
    cone_outer = cone_height_factor * (
        current_radius_sq + current_radius * outer_bottom_r + outer_bottom_sq
    )
    cone_inner = cone_height_factor * (
        inner_radius_sq + inner_radius * inner_bottom_r + inner_bottom_sq
    )
    return cone_outer - cone_inner


def _calculate_layer_surface_area(
    current_radius: float,
    cylinder_height: float,
    cone_bottom_radius: float,
    radius_offset: float,
    interior_hole_radius: float,
    cone_height: float,
    *,
    display_cylinder: bool,
    display_cone: bool,
) -> float:
    """Outer surface area for the Metal Shell layer."""
    area = 0.0
    if display_cylinder:
        area += (
            2.0 * _PI * current_radius * cylinder_height * _SQUARE_INCHES_TO_SQUARE_FEET
        )
    if display_cone:
        bottom_r = max(cone_bottom_radius - radius_offset, interior_hole_radius)
        radius_diff = current_radius - bottom_r
        slant = math.sqrt(cone_height * cone_height + radius_diff * radius_diff)
        area += (
            _PI * (current_radius + bottom_r) * slant * _SQUARE_INCHES_TO_SQUARE_FEET
        )
    return area


def _calculate_interior_void(
    last_inner_radius: float,
    half_cylinder_diameter: float,
    cone_bottom_radius: float,
    interior_hole_radius: float,
    interior_height: float,
    cone_height_factor: float,
    *,
    display_cylinder: bool,
    display_cone: bool,
) -> float:
    """Interior void volume in cubic inches (cylinder + cone)."""
    r_sq = last_inner_radius * last_inner_radius

    void_cyl = _PI * r_sq * interior_height if display_cylinder else 0.0

    if display_cone:
        total_thickness = half_cylinder_diameter - last_inner_radius
        void_bottom_r = max(cone_bottom_radius - total_thickness, interior_hole_radius)
        void_bottom_sq = void_bottom_r * void_bottom_r
        void_cone = cone_height_factor * (
            r_sq + last_inner_radius * void_bottom_r + void_bottom_sq
        )
    else:
        void_cone = 0.0

    return void_cyl + void_cone


class TRCGeometryEngine:
    """Engine for calculating TRC vessel geometry and physics.

    Performance optimizations applied:
    - Pre-computed math constants avoid repeated function calls
    - Consolidated unit conversions
    - Reduced redundant radius calculations
    """

    @staticmethod
    def _calculate_layer_contributions(
        layer: LayerConfig,
        current_radius: float,
        hole_r: float,
        cone_bot_r: float,
        radius_offset: float,
        interior_h: float,
        ch_factor: float,
        dimensions: VesselDimensions,
    ) -> tuple[LayerResult, float, float]:
        """Calculate volume, mass, and surface area for a single layer.

        Returns:
            Tuple of (LayerResult, new_radius_offset_delta, inner_radius).
        """
        t = layer.thickness
        inner_r = max(current_radius - t, hole_r)
        r_sq = current_radius * current_radius
        ir_sq = inner_r * inner_r

        cyl_vol = (
            _calculate_layer_cylinder_volume(r_sq, ir_sq, interior_h)
            if dimensions.display_cylinder
            else 0.0
        )
        cone_vol = (
            _calculate_layer_cone_volume(
                current_radius,
                r_sq,
                inner_r,
                ir_sq,
                cone_bot_r,
                radius_offset,
                t,
                hole_r,
                ch_factor,
            )
            if dimensions.display_cone
            else 0.0
        )
        top_disk = _PI * ir_sq * t if dimensions.display_lid else 0.0

        vol_ft3 = (cyl_vol + cone_vol + top_disk) * _CUBIC_INCHES_TO_CUBIC_FEET
        mass_lb = vol_ft3 * layer.density

        layer_sa = 0.0
        if layer.name == "Metal Shell":
            layer_sa = _calculate_layer_surface_area(
                current_radius,
                dimensions.cylinder_height,
                cone_bot_r,
                radius_offset,
                hole_r,
                dimensions.cone_height,
                display_cylinder=dimensions.display_cylinder,
                display_cone=dimensions.display_cone,
            )

        result = LayerResult(
            name=layer.name,
            volume_ft3=vol_ft3,
            mass_lb=mass_lb,
            density=layer.density,
            outer_surface_area_ft2=layer_sa,
        )
        return result, current_radius - inner_r, inner_r

    @staticmethod
    def _finalize_geometry_results(
        results: VesselGeometryResult,
        current_radius: float,
        half_cyl_d: float,
        cone_bot_r: float,
        hole_r: float,
        interior_h: float,
        ch_factor: float,
        dimensions: VesselDimensions,
    ) -> None:
        """Compute interior void and final geometry dimensions."""
        void_in3 = _calculate_interior_void(
            current_radius,
            half_cyl_d,
            cone_bot_r,
            hole_r,
            interior_h,
            ch_factor,
            display_cylinder=dimensions.display_cylinder,
            display_cone=dimensions.display_cone,
        )
        results.interior_volume_ft3 = void_in3 * _CUBIC_INCHES_TO_CUBIC_FEET
        results.void_radius_inches = current_radius
        results.void_diameter_inches = current_radius * 2.0
        results.interior_height_inches = interior_h

    def calculate_geometry(
        self, dimensions: VesselDimensions, layers: list[LayerConfig]
    ) -> VesselGeometryResult:
        """Calculate vessel geometry properties.

        Args:
            dimensions: Vessel dimensions and flags
            layers: List of layer configurations (ordered from outer to inner)

        Returns:
            VesselGeometryResult containing detailed calculations
        """
        # DbC preconditions
        assert (
            dimensions.cylinder_diameter > 0
        ), f"cylinder_diameter must be positive, got {dimensions.cylinder_diameter}"
        assert (
            dimensions.cylinder_height > 0
        ), f"cylinder_height must be positive, got {dimensions.cylinder_height}"

        results = VesselGeometryResult()
        if not layers:
            return results

        half_cyl_d = dimensions.cylinder_diameter * 0.5
        current_radius = half_cyl_d
        hole_r = dimensions.cone_interior_hole * 0.5
        cone_bot_r = dimensions.cone_bottom_diameter * 0.5
        interior_h = dimensions.cylinder_height - (
            dimensions.top_refractory_thickness if dimensions.display_lid else 0
        )
        ch_factor = _PI_OVER_3 * dimensions.cone_height

        total_mass = 0.0
        total_volume = 0.0
        outside_sa = 0.0
        radius_offset = 0.0

        for layer in layers:
            if not layer.visible or layer.thickness <= 0:
                continue

            layer_result, offset_delta, inner_r = self._calculate_layer_contributions(
                layer,
                current_radius,
                hole_r,
                cone_bot_r,
                radius_offset,
                interior_h,
                ch_factor,
                dimensions,
            )
            total_mass += layer_result.mass_lb
            total_volume += layer_result.volume_ft3
            outside_sa += layer_result.outer_surface_area_ft2
            results.layers.append(layer_result)

            radius_offset += offset_delta
            current_radius = inner_r

        results.total_volume_ft3 = total_volume
        results.total_mass_lb = total_mass
        results.outside_surface_area_ft2 = outside_sa

        self._finalize_geometry_results(
            results,
            current_radius,
            half_cyl_d,
            cone_bot_r,
            hole_r,
            interior_h,
            ch_factor,
            dimensions,
        )

        return results

    def calculate_residence_time(
        self, volume_ft3: float, gas_flow_acfm: float
    ) -> float:
        """
        Calculate residence time in seconds.

        Args:
            volume_ft3: Interior void volume in cubic feet
            gas_flow_acfm: Gas flow rate in Actual Cubic Feet per Minute

        Returns:
            Residence time in seconds
        """
        if gas_flow_acfm <= 0:
            return 0.0

        # Volume (ft3) / Flow (ft3/min) = min
        # Convert to seconds
        return (volume_ft3 / gas_flow_acfm) * 60.0
